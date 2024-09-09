import argparse
from copy import deepcopy
from typing import List, Union, Dict

import torch
import yaml
from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.utils.ckpt_utils import dict_to_cerber, intersect_dicts
from loguru import logger


def from_ckpt(ckpt, model, exclude=()):
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"].float().state_dict()  # to FP32
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        state_dict = ckpt.state_dict()
    loaded = False

    if "blocks." in list(model.state_dict().keys())[0] and "blocks." not in list(state_dict.keys())[0]:
        # if loading weights from yolov8 to cerberus
        state_dict = dict_to_cerber(state_dict, model)  # intersect
        loaded = True

        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=True)  # load
        logger.info("Transferred %g/%g items" % (len(state_dict), len(model.state_dict())))  # report
    elif "blocks." in list(model.state_dict().keys())[0] and "blocks." in list(state_dict.keys())[0]:
        # Load from pretrained CerberusDet state_dict
        cerber_schedule = model.yaml["cerber"]
        model.sequential_split(deepcopy(cerber_schedule), "cpu")
        model.load_state_dict(state_dict, strict=True)
        loaded = True

    return state_dict, loaded


def save_best_task_model(
        output: str,
        model: CerberusDet,
        nc: Union[int, List[int]],
        categories: Union[List[str], Dict[str, List[str]]],
        task_name: str,
):
    ckpt = _get_ckpt_to_save(model, nc, categories, task_name)

    torch.save(ckpt, output)
    print("Saved", output)


def _get_ckpt_to_save(model, nc, categories, task_name):
    if not isinstance(categories, dict):
        model.names = {task_name: categories}
    else:
        model.names = categories
    if not isinstance(nc, dict):
        model.nc = {task_name: nc}
    else:
        model.nc = nc
    ckpt = {
        "epoch": 0,
        "model": deepcopy(model),
        "names": model.names,
    }
    return ckpt


def convert(ckpt: str, cfg: str, out_path: str, data_cfg_path: str, task_name: str):
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    task_ids = data_cfg["task_ids"]

    if task_name:
        task_ind = task_ids.index(task_name)
        task_nc = data_cfg["nc"][task_ind]
        task_categories = data_cfg["names"][task_ind]

        tasks_names = [task_name]
        tasks_nc = [task_nc]
    else:
        tasks_names = task_ids
        tasks_nc = data_cfg["nc"]

    model = CerberusDet(
        task_ids=tasks_names,
        nc=tasks_nc,
        cfg=cfg,
        ch=3,
        verbose=True,
    )
    ckpt = torch.load(ckpt, map_location="cpu")
    state_dict, loaded = from_ckpt(ckpt, model, exclude=[])
    assert loaded

    if task_name:
        save_best_task_model(out_path, model, task_nc, task_categories, task_name)
    else:
        categories, num_classes = {}, {}
        for task_ind, task_id in enumerate(task_ids): 
            categories[task_id] = data_cfg["names"][task_ind]
            num_classes[task_id] = data_cfg["nc"][task_ind]
        save_best_task_model(out_path, model, num_classes, categories, task_name)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="yolov8 ckpt with state_dict")
    parser.add_argument("--cfg", type=str, default="cerberusdet/models/yolov8x.yaml", help="model.yaml path")
    parser.add_argument("--output", type=str, default="", help="output .pt file path")
    parser.add_argument("--data", type=str, default="data/voc_obj365.yaml", help="dataset.yaml path")
    parser.add_argument("--task_name", type=str, default="", help="task name to use for ckpt from --data file")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    convert(opt.weights, opt.cfg, opt.output, opt.data, opt.task_name)
