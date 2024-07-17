import os
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import mlflow
import pandas as pd
import torch
import yaml
from cerberusdet.data.datasets import LoadImagesAndLabels
from cerberusdet.evolvers.checkpoint_logger import CheckpointLogger
from cerberusdet.evolvers.file_logger import FileLogger
from cerberusdet.utils.general import check_img_size
from cerberusdet.utils.metrics import fitness, overall_fitness
from cerberusdet.utils.mlflow_logging import init_mlflow
from cerberusdet.utils.models_manager import ModelManager
from cerberusdet.utils.plots import plot_evolution
from cerberusdet.utils.train_utils import create_data_loaders
from loguru import logger
from mlflow.tracking import MlflowClient

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", -1))


class BaseEvolver(ABC):
    def __init__(self, opt, device: torch.device):

        self.device = device
        self.opt = opt

        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit, enable/disable)
        self.meta = {
            "lr0": [1, 1e-5, 1e-1, True],  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": [1, 0.01, 1.0, True],  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": [0.3, 0.6, 0.98, True],  # SGD momentum/Adam beta1
            "weight_decay": [1, 0.0, 0.001, True],  # optimizer weight decay
            "warmup_epochs": [1, 0.0, 5.0, True],  # warmup epochs (fractions ok)
            "warmup_momentum": [1, 0.0, 0.95, True],  # warmup initial momentum
            "warmup_bias_lr": [1, 0.0, 0.2, True],  # warmup initial bias lr
            "box": [1, 0.02, 0.2, True],  # box loss gain
            "cls": [1, 0.2, 4.0, True],  # cls loss gain
            "dfl": [1, 0.2, 4.0, True],  # dfl loss gain
            "hsv_h": [1, 0.0, 0.1, True],  # image HSV-Hue augmentation (fraction)
            "hsv_s": [1, 0.0, 0.9, True],  # image HSV-Saturation augmentation (fraction)
            "hsv_v": [1, 0.0, 0.9, True],  # image HSV-Value augmentation (fraction)
            "degrees": [1, 0.0, 45.0, True],  # image rotation (+/- deg)
            "translate": [1, 0.0, 0.9, True],  # image translation (+/- fraction)
            "scale": [1, 0.0, 0.9, True],  # image scale (+/- gain)
            "scaleup": [1, 0.0, 1.0, True],  # image scale (+/- gain)
            "shear": [1, 0.0, 10.0, True],  # image shear (+/- deg)
            "perspective": [0, 0.0, 0.001, True],  # image perspective (+/- fraction), range 0-0.001
            "flipud": [1, 0.0, 1.0, True],  # image flip up-down (probability)
            "fliplr": [0, 0.0, 1.0, True],  # image flip left-right (probability)
            "mosaic": [1, 0.0, 1.0, True],  # image mixup (probability)
            "mixup": [1, 0.0, 1.0, True],  # image mixup (probability)
            "label_smoothing": [1, 0.0, 0.5, True],  # label_smoothing
        }

        if self.opt.params_to_evolve.strip():
            params_to_evolve = [p.strip() for p in self.opt.params_to_evolve.split(",")]
            for k, v in self.meta.items():
                if k not in params_to_evolve:
                    self.meta[k][3] = False

        with open(self.opt.data) as f:
            data_dict = yaml.safe_load(f)
        self.task_ids = data_dict["task_ids"]

        self.params_to_evolve = [k for k, v in self.meta.items() if v[3] is True]
        self.opt.noval, self.opt.nosave = True, True  # only val/save final epoch

        self._mlflow_inited = False
        if self.opt.mlflow_url:
            self._mlflow_inited = True
            init_mlflow(self.opt.mlflow_url)
        self._init_loggers()

    @abstractmethod
    def run_evolution(self, train_func: Callable) -> None:
        pass

    def _init_loggers(self) -> None:
        self.fileLogger = FileLogger(self.opt.save_dir)
        self.checkpointLogger = CheckpointLogger(self.opt.save_dir)
        self.mlflow_update_func = (
            partial(
                update_last_artifacts,
                params_to_evolve=self.params_to_evolve,
                opts=self.opt,
                mlflow_inited=self._mlflow_inited,
            )
            if self._mlflow_inited
            else None
        )

    def load_and_cache_data(self) -> Tuple[List[LoadImagesAndLabels], List[LoadImagesAndLabels]]:
        opt = deepcopy(self.opt)
        # rank is set to 2 to not create loggers
        model_manager = ModelManager(opt.hyp, opt, rank=2, local_rank=LOCAL_RANK)
        model = model_manager.load_model(opt.cfg, self.device, False)[0]
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs)  # verify imgsz is gs-multiple

        _, _, train_datasets, valid_datasets = create_data_loaders(
            model_manager.data_dict, -1, WORLD_SIZE, opt, model_manager.hyp, gs, imgsz
        )
        del model, model_manager

        return train_datasets, valid_datasets

    def load_init_hyp(self) -> Dict[str, Any]:
        with open(self.opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        assert LOCAL_RANK == -1, "DDP mode not implemented for --evolve"
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices

        return hyp

    def plot_evolution(self):
        # Plot results
        plot_evolution(self.fileLogger.yaml_file, self.fileLogger.evolve_results_file, self.params_to_evolve)

        logger.info(
            f"Hyperparameter evolution complete. Best results saved as: {self.fileLogger.yaml_file}\n"
            f"Command to train a new model with these hyperparameters: "
            f"$ python train.py --hyp {self.fileLogger.yaml_file}"
        )
        self._update_best_run_artifacts()

    def _update_best_run_artifacts(self) -> None:
        """
        Save final results (log files, png images etc.) into the best mlflow run
        """
        if not self._mlflow_inited:
            return

        runs = mlflow.search_runs(
            experiment_names=[f"{self.opt.experiment_name}"], order_by=["param.overall_fitness DESC"]
        )

        run_ids = filter_mlflow_runs_by_name(runs, self.opt.name)
        if len(run_ids) == 0:
            return

        best_run_id = run_ids[0]
        logger.info(f"Saving final result into mlflow run {best_run_id}")

        with mlflow.start_run(run_id=best_run_id):
            mlflow.log_artifacts(self.opt.save_dir, artifact_path="final_output")


def update_last_artifacts(results_per_task, params_to_evolve, opts, mlflow_inited) -> None:
    """
    Save artifacts and best model to last mlflow run
    """

    if not mlflow_inited:
        return

    opt = deepcopy(opts)

    runs = mlflow.search_runs(
        experiment_names=[f"{opt.experiment_name}"],
        order_by=["attribute.start_time DESC"],
    )
    run_ids = filter_mlflow_runs_by_name(runs, opts.name)

    if len(run_ids) == 0:
        return

    last_run_id = run_ids[0]
    logger.info(f"Update artifacts for mlflow run {last_run_id}")

    of = overall_fitness(results_per_task)

    save_dir = Path(opt.save_dir)
    to_save = [
        save_dir / "evolve.json",
        save_dir / "hyp_evolved.yaml",
        save_dir / "results.txt",
        save_dir / "weights" / "best.pt",
    ]
    with mlflow.start_run(run_id=last_run_id):

        for f in to_save:
            if f.is_file():
                mlflow.log_artifact(str(f), artifact_path="artifacts")
            elif f.is_dir():
                dir_name = os.path.basename(str(f))
                mlflow.log_artifacts(str(f), artifact_path=f"artifacts/{dir_name}")

        mlflow.log_param("search_params", params_to_evolve)
        mlflow.log_param("overall_fitness", of)

        for task_name, task_metrics in results_per_task.items():
            mlflow.log_param(f"{task_name}_fitness", fitness(task_metrics))


def filter_mlflow_runs_by_name(runs: pd.DataFrame, name: str) -> List[str]:
    """
    Return list of run ids to observe
    """
    if len(runs) == 0:
        return []

    client = MlflowClient()
    run_ids = runs["run_id"]
    run_ids = [
        run_id
        for run_id in run_ids
        if "mlflow.runName" in client.get_run(run_id).data.tags
        and name in client.get_run(run_id).data.tags["mlflow.runName"]
        or "mlflow.runName" not in client.get_run(run_id).data.tags
    ]

    if len(run_ids) == 0:
        logger.error(f"Can not find any experiments with name {name}")

    return run_ids
