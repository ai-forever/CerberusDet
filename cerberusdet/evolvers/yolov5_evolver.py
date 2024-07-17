import logging
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
from cerberusdet.evolvers.base_evolver import BaseEvolver
from cerberusdet.utils.metrics import overall_fitness
from loguru import logger

LOGGER = logging.getLogger(__name__)


class Yolov5Evolver(BaseEvolver):
    def __init__(self, opt, device: torch.device):
        super(Yolov5Evolver, self).__init__(opt, device)
        LOGGER.info(f"[Yolov5Evolver] These parameters will be hypersearched: {self.params_to_evolve}")

    def run_evolution(self, train_func: Callable) -> None:

        hyp = self.load_init_hyp()
        logger.info(f"Start with hyp: {hyp}")
        train_datasets, valid_datasets = self.load_and_cache_data()
        for evolve_it in range(self.opt.evolve):  # generations to evolve

            hyp = self._get_next_hyp(hyp)

            results_per_task, train_epochs = train_func(
                hyp=deepcopy(hyp),
                opt=deepcopy(self.opt),
                device=self.device,
                train_dataset=train_datasets,
                val_dataset=valid_datasets,
            )

            logger.info(f"Done {evolve_it + 1}/{self.opt.evolve}: ")
            self.fileLogger.append_mutation_to_file(deepcopy(hyp), results_per_task, train_epochs, evolve_it)
            if self.fileLogger.is_last_mutation_best():
                logger.info(f"New best mutation at step {evolve_it + 1}")
                self.checkpointLogger.update_best_model()
            else:
                self.checkpointLogger.remove_last_model()

            if self.mlflow_update_func:
                self.mlflow_update_func(results_per_task)

        self.plot_evolution()

    def _mutate_from_prev_result(self, hyp: Dict[str, Any]) -> Dict[str, Any]:

        # Select parent(s)
        parent = "single"  # parent selection method: 'single' or 'weighted'

        mutations_list = self.fileLogger.read_top_5_mutations()
        task_names = mutations_list[0]["results_per_task"].keys()

        of = np.array([overall_fitness(x["results_per_task"]) for x in mutations_list])
        w = of - of.min() + 1e-6  # weights (sum > 0)

        if parent == "single" or len(mutations_list) == 1:
            n = len(mutations_list)
            # x = x[random.randint(0, n - 1)]  # random selection
            x = mutations_list[random.choices(range(n), weights=w)[0]]["hyps"]  # weighted selection
        elif parent == "weighted":
            raise NotImplementedError

        # Mutate
        mp, s = 0.8, 0.2  # mutation probability, sigma
        ng = len(self.meta)

        tasks_values = []
        for _ in task_names:
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([self.meta[k][0] for k in hyp.keys()])  # gains 0-1
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            tasks_values.append(v)

        for i, k in enumerate(hyp.keys()):
            if self.meta[k][3] is False:
                # does not mutate this hyperparam
                continue

            # mutate
            if isinstance(hyp[k], list):
                hyp[k] = [float(x[k][task_i] * tasks_values[task_i][i]) for task_i in range(len(tasks_values))]
            else:
                hyp[k] = float(x[k] * tasks_values[0][i])  # mutate

        return hyp

    def _bound_hyp_values(self, hyp: Dict[str, Any]) -> Dict[str, Any]:
        # Constrain to limits hyp values
        for k, v in self.meta.items():
            if isinstance(hyp[k], list):
                bounded = []
                for el in hyp[k]:
                    el = max(el, v[1])  # lower limit
                    el = min(el, v[2])  # upper limit
                    el = round(el, 5)  # significant digits
                    bounded.append(el)
                hyp[k] = bounded
            else:
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits
        return hyp

    def _get_next_hyp(self, hyp: Dict[str, Any]) -> Dict[str, Any]:

        if Path(self.fileLogger.evolve_results_file).exists():
            hyp = self._mutate_from_prev_result(hyp)

        return self._bound_hyp_values(hyp)
