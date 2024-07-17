import logging
import multiprocessing as mp
import os
from copy import deepcopy
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional

import ray
from cerberusdet.evolvers.base_evolver import BaseEvolver
from cerberusdet.evolvers.checkpoint_logger import CheckpointLogger
from cerberusdet.evolvers.file_logger import FileLogger
from cerberusdet.utils.metrics import fitness, overall_fitness
from loguru import logger
from ray.air import session
from ray.tune import CLIReporter, schedulers
from ray.tune.experiment.trial import Trial
from ray.tune.logger import LoggerCallback as RayLoggerCallback

LOGGER = logging.getLogger(__name__)


def objective(hyp_config, train_func, opt, device):
    ray.data.set_progress_bars(enabled=True)
    results_per_task, train_epochs = train_func(hyp=deepcopy(hyp_config), opt=deepcopy(opt), device=device)
    of = overall_fitness(results_per_task)

    report = {}
    for task_name, task_metrics in results_per_task.items():
        report[f"{task_name}_fitness"] = fitness(task_metrics)
    report["task_names"] = list(results_per_task.keys())
    report["overall_fitness"] = of
    report["train_epochs"] = train_epochs + 1
    report["results_per_task"] = results_per_task
    report["done"] = True

    session.report(report)


class RayEvolver(BaseEvolver):
    def __init__(self, opt, device):
        super(RayEvolver, self).__init__(opt, device)
        self.evolver_type = self.opt.evolver

        # ["fifo", "async_hyperband", "median_stopping_rule", "hyperband", "hb_bohb", "pbt", "pbt_replay", "pb2]
        # https://docs.ray.io/en/latest/tune/api/schedulers.html
        self.scheduler_type = "async_hyperband"

        LOGGER.info(f"[RayEvolver] These parameters will be hypersearched: {self.params_to_evolve}")

    def run_evolution(self, train_func: Callable) -> None:

        tuner = self._init_ray_tuner(train_func)
        results = tuner.fit()
        LOGGER.info(f"Best hyperparameters: {results.get_best_result().config}")

        self.plot_evolution()

    def _init_ray_tuner(self, train_func: Callable) -> ray.tune.Tuner:

        name = os.path.basename(self.opt.save_dir)
        run_name = f"ray_evolve_{name}"

        # STEP 1: init algorithm
        algo = self._init_algo(run_name)

        # STEP 2: init and cache data
        # train_datasets, valid_datasets = self.load_and_cache_data()

        # STEP 3: initialize trainable function
        n_gpus_per_run = 1
        if os.environ["CUDA_VISIBLE_DEVICES"]:
            n_gpus_per_run = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

        trainable = ray.tune.with_resources(
            ray.tune.with_parameters(
                objective,
                train_func=train_func,
                opt=self.opt,
                device=self.device,
            ),
            resources={"gpu": n_gpus_per_run, "cpu": mp.cpu_count()},
        )

        # STEP 4: initialize custom callbacks
        callback_wrapper = LoggerCallback()
        callback_wrapper.set_file_logger(self.fileLogger)
        callback_wrapper.set_checkpoint_logger(self.checkpointLogger)
        callback_wrapper.set_mlflow_update_func(self.mlflow_update_func)

        # STEP 5: initialize tuner
        scheduler = schedulers.create_scheduler(self.scheduler_type)

        tuner = ray.tune.Tuner(
            trainable,
            run_config=ray.air.RunConfig(
                name=run_name,
                local_dir=self.opt.save_dir,
                callbacks=[callback_wrapper],
                progress_reporter=TrialTerminationReporter(),
            ),
            tune_config=ray.tune.TuneConfig(
                metric="overall_fitness",
                mode="max",
                search_alg=algo,
                scheduler=scheduler,
                num_samples=self.opt.evolve,  # Number of times to sample from the hyperparameter space
            ),
            param_space=self._get_hyperparams_space(),
        )

        return tuner

    def _init_algo(self, run_name: str):
        name = os.path.basename(self.opt.save_dir)

        kwargs = {}
        if self.evolver_type == "dragonfly":
            kwargs = {"domain": "euclidean", "optimizer": "bandit"}
        if self.evolver_type == "nevergrad":
            import nevergrad

            kwargs = {"optimizer": nevergrad.optimizers.OnePlusOne}
        if self.evolver_type == "zoopt":
            kwargs = {"budget": self.opt.evolve}

        algo = ray.tune.search.create_searcher(self.evolver_type, **kwargs)
        if self.evolver_type == "optuna":
            algo._study_name = name

        # if (self.opt.exist_ok or self.opt.resume) and os.path.exists(os.path.join(self.opt.save_dir, run_name)):
        #     algo.restore_from_dir(
        #         os.path.join(self.opt.save_dir, run_name)
        #     )
        #     logger.info(f"Algo has been restored from the local dir {self.opt.save_dir}/{run_name}")

        # constrain to 4 concurrent trials
        algo = ray.tune.search.ConcurrencyLimiter(algo, max_concurrent=4)

        return algo

    def _get_hyperparams_space(self) -> Dict[str, Any]:

        hyperparams_space = {}
        hyp = self.load_init_hyp()
        for hyp_name in hyp.keys():

            if self.meta[hyp_name][3] is False:
                # does not mutate this hyperparam
                hyperparams_space[hyp_name] = hyp[hyp_name]
                continue

            min_bound = self.meta[hyp_name][1]  # lower limit
            max_bound = self.meta[hyp_name][2]  # upper limit

            # NOTE: it is assumed here that all parameters are float
            if isinstance(hyp[hyp_name], list):
                for task_name in self.task_ids:
                    hyperparams_space[f"{hyp_name}_{task_name}"] = ray.tune.uniform(min_bound, max_bound)
            else:
                # https://docs.ray.io/en/latest/tune/api/search_space.html
                hyperparams_space[hyp_name] = ray.tune.uniform(min_bound, max_bound)

        return hyperparams_space


class LoggerCallback(RayLoggerCallback):
    def set_file_logger(self, file_logger: FileLogger):
        self._file_logger = file_logger

    def set_checkpoint_logger(self, checkpoint_logger: CheckpointLogger):
        self._checkpoint_logger = checkpoint_logger

    def set_mlflow_update_func(self, mlflow_update_func: Optional[Callable] = None):
        self._mlflow_update_func = mlflow_update_func

    def on_trial_result(self, iteration, trials, trial, result, **info):

        # save new mutation to json file and update the best yaml file with hyp
        logger.info(f"Done {len(trials)-1}-th mutation")
        new_hyps = reformat_config(deepcopy(result["config"]), result["task_names"])
        self._file_logger.append_mutation_to_file(
            new_hyps, result["results_per_task"], result["train_epochs"], len(trials) - 1
        )

        # save best checkpoint and remove unnecessary last
        if self._file_logger.is_last_mutation_best():
            logger.info(f"New best mutation at step {len(trials)-1}")
            self._checkpoint_logger.update_best_model()
        else:
            self._checkpoint_logger.remove_last_model()

        if self._mlflow_update_func is not None:
            self._mlflow_update_func(result["results_per_task"])


class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


def reformat_config(config: Dict[str, float], task_names: List[str]) -> Dict[str, Any]:
    """
    Reformat config from {"shear_clothes": 0.0, "shear_shoes": 0.2, "scale": 0.3} to {"shear": [0.0, 0.2], "scale": 0.3}
    """

    if "point" in config:
        # remove the key that appeared after the dragonfly algo
        del config["point"]

    def params_grouper(item: str) -> str:
        found_tasks = [task_name for task_name in task_names if task_name in item]
        if len(found_tasks) == 0:
            return item
        return item.split(f"_{found_tasks[0]}")[0]

    new_config: Dict[str, Any] = {}

    hyp_names = list(config.keys())
    hyp_names = sorted(hyp_names, key=params_grouper)

    for hyp_name, group_items in groupby(hyp_names, key=params_grouper):
        sub_keys = list(group_items)
        if len(sub_keys) == 1:
            new_config[hyp_name] = config[sub_keys[0]]
        else:
            new_config[hyp_name] = [config[f"{hyp_name}_{task_name}"] for task_name in task_names]

    return new_config
