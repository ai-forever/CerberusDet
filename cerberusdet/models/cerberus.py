import itertools
import logging
import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from cerberusdet.models.common import Bottleneck  # noqa: F401
from cerberusdet.models.common import (  # noqa: F401
    C2,
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    BottleneckCSP,
    C2f,
    Concat,
    Contract,
    Conv,
    DWConv,
    Expand,
    Focus,
)
from cerberusdet.models.yolo import Detect, Model, get_next_layer_from_cfg, initialize_weights
from cerberusdet.utils.torch_utils import de_parallel, fuse_conv_and_bn
from loguru import logger

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", -1))


class Controller:
    """
    CerberusDet's block controller. Stores information about its index in the
    blocks list, the execution chain (blocks that should be executed in
    order before this block), and the children blocks of this block.

    Attributes:
      index:             the index of this block in the CerberusDet.blocks
      execution_chain:   indices of blocks to be executed prior to this
      parent_index:      index (in CerberusDet.blocks) of the parent block
      children_indices:  indices (in CerberusDet.blocks) of the childrens
      task_id:           if this block is a head, stores the task_id
      serving_tasks:     a dict {task_id: idk_what_this_is}
    """

    def __init__(self, index=None):
        self.index = index
        self.execution_chain = [index]
        self.parent_index = None
        self.children_indices = []
        self.task_id = None
        self.serving_tasks = dict()

    def stack_on(self, controller):
        """Stacks current controller on top of another controller"""
        prev_chain = controller.execution_chain.copy()
        self.execution_chain = prev_chain + [self.index]
        self.parent_index = controller.index
        controller.children_indices.append(self.index)
        return self

    def add_parent(self, controller, controllers):
        """Extend parents for the current controller"""
        if self.parent_index == controller.index:
            return
        if isinstance(self.parent_index, list) and controller.index in self.parent_index:
            return
        if self.parent_index is None:
            return self.stack_on(controller)

        new_chain = controller.execution_chain.copy()
        if isinstance(self.parent_index, int):
            self.parent_index = [self.parent_index, controller.index]
        else:
            self.parent_index = [*self.parent_index, controller.index]

        if self.index not in controller.children_indices:
            controller.children_indices.append(self.index)
        new_chain.append(self.index)

        # Merge chains to have correct execution order
        n_elemnts = len(np.unique(new_chain + self.execution_chain))
        merged_chain = []
        index_left = index_right = 0
        while len(merged_chain) < n_elemnts:
            new_ind = new_chain[index_left]
            old_ind = self.execution_chain[index_right]
            if old_ind == new_ind:
                index_right += 1
                index_left += 1
                merged_chain.append(old_ind)
            else:
                if old_ind in controllers[new_ind].execution_chain:
                    if old_ind not in merged_chain:
                        merged_chain.append(old_ind)
                    index_right += 1
                else:
                    if new_ind not in merged_chain:
                        merged_chain.append(new_ind)
                    index_left += 1

            if index_right == len(self.execution_chain):
                merged_chain += new_chain[index_left:]
                break
            elif index_left == len(new_chain):
                merged_chain += self.execution_chain[index_right:]
                break

        self.execution_chain = merged_chain

        return self

    def __str__(self):
        return "({}): parent={}, children={}, serving=[{}]".format(
            self.index,
            self.parent_index,
            self.children_indices,
            ", ".join(str(task_id) for task_id in self.serving_tasks),
        )

    def __repr__(self):
        return str(self)

    def serialize(self):
        """Serialize to ordinary python's dict object"""
        return self.__dict__

    def deserialize(self, serialized_controller):
        """Deserialize from a python's dict object"""
        for k, v in serialized_controller.items():
            setattr(self, k, v)
        return self


class CerberusDet(nn.Module):
    def __init__(self, task_ids, nc, cfg="yolov5s.yaml", ch=3, verbose=True):
        """
        Base configuration will be:

          (neck_n+1) (..) (N)  <-  heads
           |           |   |
           +---+---|---+---+
                   |
                (neck_n)
                   |
                  ...
                   |
                  (1)            <-  neck
                   |
                  (0)            <-  backbone
                   |
                  (*)            <-  input

        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.controllers = list()
        self.heads = dict()
        self.rep_tensors = dict()
        self.branching_points = set()
        self.verbose = verbose

        backbone = Model(cfg=cfg, ch=ch, nc=nc, without_head=True, verbose=self.verbose)
        model = self.add_block(backbone)

        self.gd, self.gw = backbone.yaml["depth_multiple"], backbone.yaml["width_multiple"]
        self.max_channels = backbone.yaml.get("max_channels", 1024)
        nc = backbone.yaml["nc"]  # has been set during backbone building

        self.neck_head_save = []  # save for some blocks inputs
        model, layer_ind_map = self.parse_neck(model, deepcopy(backbone.yaml), backbone.saved_ch, nc)
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print("Finish neck parsing")

        self.parse_heads(
            model, deepcopy(backbone.yaml), task_ids, backbone.saved_ch, backbone.inplace, layer_ind_map, deepcopy(nc)
        )
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print("Finish heads parsing")

        del backbone.saved_ch
        for block in self.blocks:
            initialize_weights(block)

        self.yaml = backbone.yaml
        self.build()
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print(self.info())

    def test_forward(self, device=None):
        # test forward for all tasks: not use during training (leads to error during ema creation)
        s = 256
        test_input = torch.ones(1, 3, s, s)
        if device is not None:
            test_input = test_input.to(device)
        self.forward(test_input)

    def parse_neck(self, prev_model, cfg, ch, nc):

        ind = len(cfg["backbone"])
        layer_ind_map = {}  # map for initial layer inds

        for i, layer in enumerate(cfg["neck"], start=1):

            f, n, m, args = layer  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            args, _, n, c2, m_ = get_next_layer_from_cfg(self.gd, ch, self.gw, nc, m, n, f, args, self.max_channels)

            ch.append(c2)

            t = str(m)[8:-2].replace("__main__.", "")  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params

            next_block = self.add_block(m_).stack_on(prev_model)
            f = [f] if isinstance(f, int) else f
            assert f[0] == -1 or len(f) == 1, "Unsupported config"

            new_input_idx = []
            for x in f:
                if x != -1 and x >= len(cfg["backbone"]):
                    x = layer_ind_map[x]  # middle of the neck
                    self.neck_head_save.append(x)
                    next_block.add_parent(self.controllers[x], self.controllers)
                elif x == -1 and i == 1:
                    raise ValueError("Input for first cerbernet block must be defined")
                elif x != -1 and x < len(cfg["backbone"]):
                    x = (0, x)  # input from the middle of backbone
                    next_block.add_parent(self.controllers[0], self.controllers)
                new_input_idx.append(x)

            m_.i, m_.f, m_.type, m_.np = ind, new_input_idx, t, np  # original index, 'from' index, type, number params
            if self.verbose and LOCAL_RANK in [-1, 0]:
                LOGGER.info("%3s-%3s%18s%3s%10.0f  %-40s%-30s" % (i, ind, f, n, np, t, args))  # print

            prev_model = next_block
            layer_ind_map[ind] = next_block.index

            ind += 1

        return prev_model, layer_ind_map

    def parse_heads(self, prev_model, cfg, task_ids, ch, inplace, layer_ind_map, nc):

        ind = len(cfg["backbone"]) + len(cfg["neck"])
        if len(cfg["head"]) != 1:
            raise NotImplementedError

        for task_id in task_ids:

            for i, layer in enumerate(cfg["head"]):
                f, n, m, args = layer  # from, number, module, args

                m = eval(m) if isinstance(m, str) else m  # eval strings
                if self.verbose and LOCAL_RANK in [-1, 0]:
                    print(f"Adding head for {task_id}")
                args_, nc, n, _, m_ = get_next_layer_from_cfg(
                    self.gd, ch, self.gw, nc, m, n, f, deepcopy(args), self.max_channels
                )
                h = self.add_head(m_, task_id)

                t = str(m)[8:-2].replace("__main__.", "")  # module type
                np = sum([x.numel() for x in m_.parameters()])  # number params
                f = [f] if isinstance(f, int) else f

                if f[0] == -1:
                    h.stack_on(prev_model)

                new_input_idx = []
                for x in f:
                    if x != -1 and x >= len(cfg["backbone"]):
                        x = layer_ind_map[x]
                        self.neck_head_save.append(x)
                        h.add_parent(self.controllers[x], self.controllers)
                    elif x != -1 and x < len(cfg["backbone"]):
                        raise ValueError("Input for the head must be from neck")
                    new_input_idx.append(x)

                m_.i, m_.f, m_.type, m_.np = (
                    ind,
                    new_input_idx,
                    t,
                    np,
                )  # original index, 'from' index, type, number params

                if m in [Detect]:
                    s = 256  # 2x min stride
                    m_.inplace = inplace
                    m_.stride = torch.tensor(
                        [s / x.shape[-2] for x in self.forward(torch.zeros(1, cfg["ch"], s, s), task_id)]
                    )  # forward

                    if not hasattr(self, "stride"):
                        self.stride = m_.stride
                    else:
                        assert torch.equal(m_.stride, self.stride)

                    m_.bias_init()  # only run once

                if self.verbose and LOCAL_RANK in [-1, 0]:
                    LOGGER.info("%3s-%3s%18s%3s%10.0f  %-40s%-30s" % (i, ind, f, n, np, t, args_))  # print

    def set_task(self, task_id):
        # DO NOT USE: pass task_id as argument into forward
        self.cur_task = task_id

    def get_head(self, task_id) -> Detect:
        indx = self.heads[task_id]
        return self.blocks[indx]

    def add_block(self, module):
        """
        Registers a new CerberusDet block, automatically adds it to the
        self.blocks and the execution graph.

        Args:
          module: a `nn.Module` object

        Returns:
          a Controller object for newly added block
        """
        new_index = len(self.blocks)
        new_controller = Controller(new_index)
        self.blocks.append(module)
        self.controllers.append(new_controller)
        return new_controller

    def add_head(self, module, task_id):
        """
        Registers a new CerberusDet block as a "Head". Same as the method
        `add_block()`, but adds the controller to self.heads.

        Args:
          module:    a `nn.Module` object
          task_id:  an identifier of the task that the head is solving

        Returns:
          a Controller object for newly added block
        """
        new_controller = self.add_block(module)
        new_controller.task_id = task_id
        self.heads[task_id] = new_controller.index
        return new_controller

    def info(self):
        """ """
        items = "\n  ".join(str(c) for c in self.controllers)
        controllers = "(block controllers):\n  " + items
        items = "\n  ".join("({}) -> {}  {}".format(k, str(c), type(self.blocks[c])) for k, c in self.heads.items())
        heads = "(heads):\n  " + items
        return controllers + "\n" + heads

    def execution_plan(self, task_ids: Union[List[str], str]):
        """
        Dynamicaly constructs an execution plan, given the identifiers
        of tasks that we want to perform.

        Args:
          task_ids:  an identifier, or list of identifiers of tasks

        Returns:
          execution_order: a list of indices of modules to be executed
          branching_ids:   indices of branching points
        """
        if not isinstance(task_ids, list):
            task_ids = [task_ids]
        execution_order = []
        branching_ids = set()
        for task_id in task_ids:
            branching_point = None
            controller = self.controllers[self.heads[task_id]]
            task_exec_chain = controller.execution_chain
            for i, index in enumerate(task_exec_chain):
                if index not in execution_order:
                    break
                branching_point = index
            execution_order += task_exec_chain[i:].copy()
            if branching_point is not None:
                parents = self.controllers[index].parent_index
                if isinstance(parents, int):
                    assert parents == branching_point
                    branching_ids.add(branching_point)
                else:
                    branching_ids.update(parents)
        return execution_order, branching_ids

    def control_blocks(self, task_ids=None):
        """
        Yields an iterator over the blocks. If `task_ids` are specified,
        only blocks flowing towards corresponding heads will be yielded.
        """
        if task_ids is None:
            for controller, block in zip(self.controllers, self.blocks):
                yield controller, block
        else:
            execution_order, _ = self.execution_plan(task_ids)
            for index in execution_order:
                yield self.controllers[index], self.blocks[index]

    def parameters(self, recurse=True, task_ids=None, only_trainable=False):
        """
        Returns an iterator over module parameters. If task_ids
        are specified, returns an iterator only over the parameters
        that affects the outputs on those tasks.

        Args:
          recurse:         whether to yield the parameters of submodules
          task_ids:        whether to yield only task-related parameters
          only_trainable:  whether to yield only trainable parameters

        Yields:
          Parameter: module parameter
        """
        if task_ids is None and not only_trainable:
            for param in super().parameters(recurse):
                yield param
        else:
            if task_ids is None:
                task_ids = list(self.heads.keys())
            execution_order, _ = self.execution_plan(task_ids)
            for index in execution_order:
                if only_trainable:
                    if not hasattr(self.blocks[index], "trainable"):
                        continue
                    if self.blocks[index].trainable is not True:
                        continue

                for param in self.blocks[index].parameters():
                    yield param

    def build(self):
        """
        Builds the model.
        """
        for _, head_index in self.heads.items():
            controller = self.controllers[head_index]
            task_id = controller.task_id
            for index in controller.execution_chain:
                idx = len(self.controllers[index].serving_tasks)
                self.controllers[index].serving_tasks[task_id] = idx
        _, self.branching_points = self.execution_plan(list(self.heads.keys()))

    def create_nested_branch(
        self,
        index: int,
        branches: List[int],
        device: Optional[torch.device] = None,
        inds_to_map_per_head: Optional[Dict[int, List[int]]] = None,
        next_ids_map: Optional[Dict[int, Dict[int, int]]] = None,
    ):
        """
        Dynamically clones childrens of `self.blocks[index]`, and stacks the branches
        specified by `branches` on top of the newly cloned branch.

        [Before]                         [After]
                    __ ...........           --1----2- ...........
        index      /                        / index
        --O---1---2--- branches[0]       --O            __ branches[0]
                   \__                      \ clones   /
                       branches[1]           --1----2--- branches[1]

        Args:
          index:      index of the block which childrens to clone
          branches:   indices of block's children to stack on the clones
          device:     device to spawn the clone on, can be decided later
          inds_to_map_per_head: blocks indexes per each head index,
                    which need to be mapped to new modules blocks indexes
                    e.g. {branches[0]: [2]} means that for head `branches[0]`
                        we need to get new index of previous block with index 2
          next_ids_map: dict to save new blocks indexes from inds_to_map_per_head
                    per each head index
                    e.g. {branches[0]: {2: 4}} means that for head `branches[0]`
                        block with index 2 was mapped into block with index 4


        Returns:
          controllers: controllers of the newly created branch modules
          blocks:      modules of the newly created branch
        """
        if index in self.heads:
            raise ValueError("Cannot split 's head.")

        if inds_to_map_per_head is not None:
            assert next_ids_map is not None

        start_controller = self.controllers[index]
        # for b in branches:
        #     if b in start_controller.children_indices:
        #         raise ValueError(f"Ind {b} of branches are already in "
        #                          f"controller's children_indices {start_controller.children_indices}.")

        branches_names = [task_id for task_id, task_ind in self.heads.items() if task_ind in branches]
        if len(branches_names) != len(branches):
            raise ValueError("Indices of branches must be indexes of heads.")

        cloned_blocks = []
        cloned_controllers = []

        exec_order, _ = self.execution_plan(branches_names)
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print(f"\nOld exec paln for {branches_names} : {exec_order}")
        clones_ids = {}

        # clone nn modules and create new controllers
        prev_model = start_controller
        prev_controller = start_controller
        for ind in exec_order:
            if ind <= index:
                continue
            if ind in branches:
                break
            cloned_block = deepcopy(self.blocks[ind])
            if device is not None:
                cloned_block = cloned_block.to(device)

            controller = self.controllers[ind]
            new_index = len(self.controllers)
            cloned_controller = Controller(new_index)
            clones_ids[controller.index] = new_index
            self.controllers.append(cloned_controller)
            self.blocks.append(cloned_block)

            # change parents and childrend
            if isinstance(controller.parent_index, int):
                cloned_controller.stack_on(prev_model)
            elif isinstance(controller.parent_index, list):
                cloned_controller.stack_on(prev_model)
                for parent_ind in controller.parent_index:
                    if parent_ind == prev_controller.index:
                        continue
                    if parent_ind in clones_ids:
                        new_parent_ind = clones_ids[parent_ind]
                        cloned_controller.add_parent(self.controllers[new_parent_ind], self.controllers)
                        if parent_ind in self.neck_head_save:
                            self.neck_head_save.append(new_parent_ind)
                    else:
                        cloned_controller.add_parent(self.controllers[parent_ind], self.controllers)

            else:
                raise ValueError("Unknown parent type")

            for i, from_ind in enumerate(cloned_block.f[:]):
                if from_ind != -1 and from_ind in clones_ids:
                    cloned_block.f[i] = clones_ids[from_ind]

            prev_model = cloned_controller
            prev_controller = controller

            cloned_blocks.append(cloned_block)
            cloned_controllers.append(cloned_controller)

        # stack heads on top of cloned branch
        for head_ind in branches:
            head_controller = self.controllers[head_ind]
            head_controller.execution_chain = [head_controller.index]

            for i, from_ind in enumerate(self.blocks[head_ind].f[:]):
                if from_ind != -1 and from_ind in clones_ids:
                    self.blocks[head_ind].f[i] = clones_ids[from_ind]

            if isinstance(head_controller.parent_index, int):
                parent_ind = head_controller.parent_index
                assert parent_ind in clones_ids
                self.controllers[parent_ind].children_indices.remove(head_ind)
                new_parent_ind = clones_ids[parent_ind]
                head_controller.stack_on(self.controllers[new_parent_ind])

                if parent_ind in self.neck_head_save:
                    self.neck_head_save.append(new_parent_ind)
                continue

            old_parent_inds = head_controller.parent_index
            head_controller.parent_index = None
            for i, parent_ind in enumerate(old_parent_inds):
                old_parent = self.controllers[parent_ind]
                if parent_ind in clones_ids:
                    old_parent.children_indices.remove(head_ind)
                    new_parent_ind = clones_ids[parent_ind]
                    head_controller.add_parent(self.controllers[new_parent_ind], self.controllers)
                    if parent_ind in self.neck_head_save:
                        self.neck_head_save.append(new_parent_ind)
                elif head_controller.parent_index is None:
                    prev_chain = old_parent.execution_chain.copy()
                    head_controller.execution_chain = prev_chain + [head_controller.index]
                    head_controller.parent_index = old_parent.index
                    assert head_controller.index in old_parent.children_indices
                else:
                    assert head_controller.index in old_parent.children_indices
                    head_controller.add_parent(old_parent, self.controllers)

        # update serving tasks and branch indexes
        for controller in self.controllers:
            controller.serving_tasks = dict()
        self.rep_tensors.clear()
        self.build()  # build model again

        exec_order, _ = self.execution_plan(branches_names)
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print(
                f"\nNew exec paln for {branches_names}({branches}) : "
                f"{exec_order}\n Branching ids: {self.branching_points}"
            )

        # map requested indexes
        for old_ind, new_ind in clones_ids.items():
            for task_ind in branches:
                if (
                    inds_to_map_per_head is not None
                    and task_ind in inds_to_map_per_head
                    and old_ind in inds_to_map_per_head[task_ind]
                ):

                    next_ids_map[task_ind][old_ind] = new_ind

        return cloned_controllers, cloned_blocks

    def split(self, index, branching_scheme, device, next_cerber_configs):
        """
        Splits a model's block into several blocks, according to the
        `branching_scheme`. Results of `split(0, [[2], [3, 4], [5, 6]])`:

        | B |  (2) (3) (4) (5) (6)     | A |  (2) (3) (4) (5) (6)
        | E |   |   |   |   |   |      | F |   |   |   |   |   |
        | F |   +---+---|---+---+      | T |   |   +---|   +---|
        | O |          (1)             | E |  (1)     (6)     (7)
        | R |           |              | R |  |        |       |
        | E |          (0)             |   |  +--------|-------+
        |   |           |              |   |          (0)
        |   |          (*)             |   |           |
        |   |                          |   |          (*)

        Args:
          index:            index of the block to split
          branching_scheme: list of list of indices (as example above)
          device:           a device to spawn the new branches on

        Raises:
          ValueError:       in case invalid parameters are specified

        Returns:
          controllers:      list of controllers of splitted branches
          blocks:           list of blocks - the splitted branches
        """

        inds_to_map_per_head: Dict[int, List[int]] = defaultdict(list)
        next_ids_map: Dict[int, Dict[int, int]] = {}

        for sc in next_cerber_configs:  # schedule e.g.: [[4, [[16, 17], [15]]], [12, [[17], [16]]]]
            for head_ind in itertools.chain(*sc[1]):
                inds_to_map_per_head[head_ind].append(sc[0])
                next_ids_map[head_ind] = {sc[0]: None}
                if head_ind in branching_scheme[0]:
                    next_ids_map[head_ind][sc[0]] = sc[0]

        controller = self.controllers[index]
        block = self.blocks[index]

        total_branches = set()
        for branch in branching_scheme:
            total_branches.update(set(branch))

        if not total_branches == set(self.heads.values()):
            missed_inds = [ind for ind in self.heads.values() if ind not in total_branches]
            logger.warning(f"Branching config does not include {missed_inds} head inds")

        for i in range(len(branching_scheme)):
            scheme_a = set(branching_scheme[i])
            for j in range(i + 1, len(branching_scheme)):
                scheme_b = set(branching_scheme[j])
                if not scheme_a.isdisjoint(scheme_b):
                    raise ValueError("The branching schemes should " "be disjoint to each other.")

        if self.verbose and LOCAL_RANK in [-1, 0]:
            logger.info(f"Branching ids: {self.branching_points}")
        new_controllers, new_blocks = [controller], [block]
        for branch in branching_scheme[1:]:
            if self.verbose and LOCAL_RANK in [-1, 0]:
                logger.info(f"Creating branch for {branch} at {index}")
            tmp_ctrl, tmp_block = self.create_nested_branch(index, branch, device, inds_to_map_per_head, next_ids_map)
            new_controllers.append(tmp_ctrl)
            new_blocks.append(tmp_block)

        # self.test_forward(device)
        return new_controllers, new_blocks, next_ids_map

    def sequential_split(self, cerber_schedule, device):
        """
        Sequentially splits a model's block into several blocks, according to the
        each `branching_scheme` in `cerber_schedule`
        """
        schedule_head_ids = [list(itertools.chain(*cerber_conf[-1])) for cerber_conf in cerber_schedule]
        schedule_head_ids = list(itertools.chain(*schedule_head_ids))
        schedule_head_ids = sorted(list(np.unique(schedule_head_ids)))
        model_head_ids = sorted(list(self.heads.values()))

        assert (
            model_head_ids == schedule_head_ids or len(schedule_head_ids) == 0
        ), f"Invalid cerberusNet config {cerber_schedule}"

        for i in range(len(cerber_schedule)):
            branching_scheme = cerber_schedule[i]  # e.g. branching_scheme: [2, [[15], [16, 17]]]
            next_configs = cerber_schedule[i + 1 :]
            # get new indexes for all future configurations

            _, _, ids_map = self.split(*branching_scheme, device, next_configs)
            # ids_map: e.g. {16: {1: 18}, {17: {1: 18}}}

            # update ids in the next configs for next branching
            for ii, next_branching_scheme in enumerate(next_configs):
                mapped_ind = [
                    ids_map[head_ind][next_branching_scheme[0]]
                    for head_ind in itertools.chain(*next_branching_scheme[1])
                ]
                assert None not in mapped_ind
                assert len(np.unique(mapped_ind)) == 1
                cerber_schedule[i + 1 + ii][0] = mapped_ind[0]

            # print(shared_model.info())
            # print("Next configs: ", cerber_schedule[i+1:])

    def fuse(self):
        if not hasattr(self, "verbose"):
            setattr(self, "verbose", True)
        # fuse model Conv2d() + BatchNorm2d() layers in the base yolo model
        if self.verbose and LOCAL_RANK in [-1, 0]:
            LOGGER.info("Fusing layers... ")

        for module in self.blocks:
            if isinstance(module, Model):
                module = module.fuse()
            else:
                for m in module.modules():
                    if type(m) is Conv and hasattr(m, "bn"):
                        m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                        delattr(m, "bn")  # remove batchnorm
                        m.forward = m.fuseforward  # update forward
                        # print("Fuse layer")

        return self

    def _get_one_input(self, block_layer, parent_index, parent_out, branching_ids, rep_tensors):
        if parent_index in branching_ids:
            parent_out = rep_tensors[parent_index]

        if isinstance(parent_out, list):
            # select tensor from backbone output
            assert len(block_layer.f) == 1 and block_layer.f[0][1] != -1
            input_idx = block_layer.f[0][1]
            parent_out = parent_out[input_idx]
            assert parent_out is not None
        return [parent_out]

    def _get_several_inputs(
        self, block_layer, parent_index, x, branching_ids, middle_outputs, rep_tensors, neck_head_save
    ):
        next_input = []
        assert len(block_layer.f) == len(parent_index)

        for input_idx, _parent_index in zip(block_layer.f, parent_index):

            if isinstance(input_idx, tuple):
                # input from the middle of backbone
                assert _parent_index == 0
                parent_out = (
                    middle_outputs[_parent_index] if _parent_index not in branching_ids else rep_tensors[_parent_index]
                )

                assert isinstance(parent_out, list)
                backbone_ind = input_idx[1]
                parent_out = parent_out[backbone_ind]
            elif isinstance(input_idx, int) and input_idx == -1:
                # output from just previous layer
                parent_out = x if _parent_index not in branching_ids else rep_tensors[_parent_index]
            elif isinstance(input_idx, int) and input_idx != -1 and input_idx in neck_head_save:
                # input from the middle of neck
                assert _parent_index == input_idx
                parent_out = (
                    middle_outputs[_parent_index] if _parent_index not in branching_ids else rep_tensors[_parent_index]
                )
            else:
                raise ValueError(f"Unknown input index {input_idx}")

            next_input.append(parent_out)
        return next_input

    def forward(self, input_tensor, task_ids=None, retain_tensors=False, retain_all=False):
        """
        Defines the computation performed at every call. Dynamically
        and automatically decides what to run and in what order.

        Args:
          input_tensor:    a common input for specified tasks
          task_ids:        identifiers of tasks to be executed
          retain_tensors:  if True, save branching tensors to rep_tensors
          retain_all:      if True, save ALL tensors at rep_tensors

        Returns:
          A dictionary {task_id: output} of task-specific outputs
        #"""
        if task_ids is None and hasattr(self, "cur_task"):
            # for stack trace
            print(f"WARN: forcely inference for {self.cur_task} task")
            task_ids = self.cur_task
        elif task_ids is None:
            task_ids = list(self.heads.keys())

        exec_order, branching_ids = self.execution_plan(task_ids)
        # print(f"{task_ids}: Exec order: {exec_order}, branches: {branching_ids}")

        x = input_tensor
        outputs = dict()

        middle_outputs = dict()
        for index in exec_order:

            controller = self.controllers[index]
            parent_index = controller.parent_index

            block_layer = self.blocks[index]

            if parent_index is None:
                # input for backbone
                next_input = [x]
            elif isinstance(parent_index, int):
                # one input
                parent_out = x
                next_input = self._get_one_input(block_layer, parent_index, parent_out, branching_ids, self.rep_tensors)
            else:
                # several inputs (concat or head layer)
                next_input = self._get_several_inputs(
                    block_layer,
                    parent_index,
                    x,
                    branching_ids,
                    middle_outputs,
                    self.rep_tensors,
                    self.neck_head_save,
                )

            if len(next_input) == 1:
                next_input = next_input[0]

            # forward
            x = block_layer(next_input)

            # save tensors
            if retain_all:
                self.rep_tensors[index] = x
            elif retain_tensors and index in self.branching_points:
                self.rep_tensors[index] = x
            elif index in branching_ids:
                self.rep_tensors[index] = x

            if index in self.neck_head_save:
                assert index not in middle_outputs
                middle_outputs[index] = x
            if index == 0 and 0 not in branching_ids:
                # save backbone output
                middle_outputs[0] = x

            if controller.task_id is not None:
                outputs[controller.task_id] = x

        return outputs[task_ids] if isinstance(task_ids, str) else outputs

    @staticmethod
    def freeze_shared_layers(cerberus_model):
        model = de_parallel(cerberus_model)

        model_tasks = len(model.heads)
        if model_tasks == 1:
            return

        if model.verbose and LOCAL_RANK in [-1, 0]:
            logger.info("freeze layers...")

        for idx, (ctrl, block) in enumerate(model.control_blocks()):
            n_branches = max(len(ctrl.serving_tasks), 1.0)
            if n_branches != model_tasks:
                continue
            for _, p in block.named_parameters():
                p.requires_grad = False

            for m in block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.eval()

    @staticmethod
    def unfreeze_shared_layers(cerberus_model):
        model = de_parallel(cerberus_model)

        model_tasks = len(model.heads)
        if model_tasks == 1:
            return

        if model.verbose and LOCAL_RANK in [-1, 0]:
            logger.info("unfreeze layers...")

        for idx, (ctrl, block) in enumerate(model.control_blocks()):
            n_branches = max(len(ctrl.serving_tasks), 1.0)
            if n_branches != model_tasks:
                continue
            for _, p in block.named_parameters():
                p.requires_grad = True

            for m in block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True
                    m.train()
