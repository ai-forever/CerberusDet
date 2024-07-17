import logging
import os
import random

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.trainers.base_trainer import BaseTrainer
from cerberusdet.utils.general import colorstr, one_cycle
from cerberusdet.utils.plots import plot_lr_scheduler
from cerberusdet.utils.torch_utils import de_parallel, get_hyperparameter
from loguru import logger
from torch.cuda import amp
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))


class Averaging(BaseTrainer):
    """
    Gradient averaging
    """

    def __init__(self, device, model, model_manager, train_loaders, val_loaders, dataset, imgsz, gs, loss_weights=None):
        super().__init__(
            device=device,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            model_manager=model_manager,
            imgsz=imgsz,
            gs=gs,
            dataset=dataset,
        )

        self.cuda = device.type != "cpu"
        self.batch_size = self.opt.batch_size

        self.nbs = 64  # nominal batch size
        datasets_len = [len(train_loader) for train_loader in self.train_loaders]
        if LOCAL_RANK in [-1, 0]:
            logger.info(f"datasets sizes: {datasets_len} ")
        self.nb = max(datasets_len)

        # number of warmup iterations, max(3 epochs, 1k iterations)
        self.nw = max(round(get_hyperparameter(self.hyp, "warmup_epochs") * self.nb), 1000)

        self.optimizer = get_optimizer(model, self.hyp)
        self.scheduler, self.lf = get_lr_scedule(self.optimizer, self.hyp, self.opt.linear_lr, model_manager.epochs)
        self.scaler = amp.GradScaler(enabled=self.cuda)

        if loss_weights is None:
            loss_weights = dict(zip(self.task_ids, [1.0] * len(self.task_ids)))
            if LOCAL_RANK in [-1, 0]:
                logger.info(f"NOTE: Use tasks loss weights: {loss_weights} ")

        self.loss_weights = dict((k, torch.tensor(v, device=device)) for k, v in loss_weights.items())
        self.nc = self.model_manager.data_dict["nc"]

    def resume(self, ckpt):

        start_epoch = 0

        if ckpt is None:
            return start_epoch

        # Optimizer
        if ckpt.get("optimizer") is not None:
            if not isinstance(ckpt["optimizer"], list):
                ckpt_optimizer = ckpt["optimizer"]
            else:
                assert len(ckpt["optimizer"]) == 2
                ckpt_optimizer = ckpt["optimizer"]

            self.optimizer.load_state_dict(ckpt_optimizer)
            self.best_fitness = ckpt["best_fitness"]
            self.best_fitness_per_task = ckpt["best_fitness_per_task"]
            self.scheduler.last_epoch = start_epoch

        start_epoch = ckpt.get("epoch", -1) + 1
        return start_epoch

    def get_optimizer_dict(self):
        return self.optimizer.state_dict()

    def train_epoch(self, model, ema, epoch, local_rank, world_size):

        if hasattr(de_parallel(model), "heads"):  # CerberNet
            if epoch < self.opt.freeze_shared_till_epoch:
                CerberusDet.freeze_shared_layers(model)
            elif 0 < self.opt.freeze_shared_till_epoch == epoch:
                CerberusDet.unfreeze_shared_layers(model)

        plots = not self.opt.evolve
        model.train()

        loader_iterators = []
        for task_i, task in enumerate(self.task_ids):
            data_loader = self.train_loaders[task_i]

            if local_rank != -1:
                data_loader.sampler.set_epoch(epoch)
            loader_iterators.append(iter(data_loader))

        # start epoch
        self.reset_print_info(local_rank)
        pbar = enumerate(range(self.nb))

        num_branches = dict()
        for idx, (ctrl, block) in enumerate(de_parallel(model).control_blocks()):
            n_branches = max(len(ctrl.serving_tasks), 1.0)
            num_branches[idx] = torch.tensor(n_branches, device=self.device)

        progress_bar = tqdm(total=self.nb) if local_rank in [-1, 0] else None

        # save model for tests
        # self.model_manager.save_model(epoch, self.best_fitness_per_task, self.best_fitness, model, ema, self.get_optimizer_dict(), is_best=True)  # noqa: E501

        self.optimizer.zero_grad()
        log_step = self.nb // 10 if self.opt.evolve else 1
        for i, batch_idx in pbar:

            # Warmup
            ni = i + self.nb * epoch  # number integrated batches (since train start)
            if ni <= self.nw:
                BaseTrainer.warmup_lr(ni, epoch, self.optimizer, self.nw, self.hyp, self.lf)

            printed_task_i = random.randint(0, len(self.task_ids) - 1)

            # for each task, calculate head grads and accumulate body grads
            for task_i, task_id in enumerate(self.task_ids):

                try:
                    batch = next(loader_iterators[task_i])
                except StopIteration:
                    loader_iterators[task_i] = iter(self.train_loaders[task_i])
                    batch = next(loader_iterators[task_i])

                batch = self.preprocess_batch(batch)

                # Forward
                # do inference with backward
                with amp.autocast(enabled=self.cuda):
                    output = model(batch["img"], task_id)  # forward

                    loss, loss_items = self.compute_loss(output, batch, task_id)  # loss scaled by batch_size
                    if local_rank != -1:
                        loss *= world_size  # gradient averaged between devices in DDP mode

                    wloss = self.loss_weights[task_id] * loss

                # Backward
                self.scaler.scale(wloss).backward()

                n_lbls = batch["bboxes"].shape[0]
                self._log_info(epoch, task_id, local_rank, loss_items, n_lbls, batch["img"].shape[-1])

                if local_rank in [-1, 0]:
                    # print randomly one of the tasks' statistics
                    if printed_task_i == task_i and (i % log_step == 0 and i > 0 or i == self.nb - 1):
                        progress_bar.update(log_step)
                        progress_bar.set_description(self.stat[task_id])

                    # Plot
                    if plots:
                        self.model_manager.plot_train_images(i, task_id, batch, de_parallel(model))

            self.optimizer_step(model, ema, num_branches)

        # Scheduler
        lr = [x["lr"] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()

        # Log
        if local_rank in [-1, 0]:
            for task_i, task in enumerate(self.task_ids):
                self.model_manager.train_log(task, lr, self.mloss_per_task[task_i], epoch, self.stat[task])

    def optimizer_step(self, model, ema, num_branches):

        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients

        # averaging out body gradients
        for idx, (_, block) in enumerate(de_parallel(model).control_blocks()):
            for p_name, p in block.named_parameters():
                if not p.requires_grad:
                    continue
                p.grad /= num_branches[idx]

        self.scaler.step(self.optimizer)  # optimizer.step
        self.scaler.update()
        self.optimizer.zero_grad()
        if ema:
            ema.update(model)


def init_optimizer(g, lr, momentum, name="SGD"):

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    return optimizer


def get_optimizer(model, hyp, name="SGD"):

    # Optimizer
    decay = get_hyperparameter(hyp, "weight_decay")
    lr = get_hyperparameter(hyp, "lr0")
    momentum = get_hyperparameter(hyp, "momentum")

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()

    for _, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    optimizer = init_optimizer(g, lr, momentum, name)

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias"
    )

    return optimizer


def get_lr_scedule(optimizer, hyp, use_linear_lr, epochs):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lrf = get_hyperparameter(hyp, "lrf")
    if use_linear_lr:
        # linear
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf  # noqa: E731
    else:
        lf = one_cycle(1, lrf, epochs)  # cosine 1->lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, epochs)

    return scheduler, lf
