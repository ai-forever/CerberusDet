import logging
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.utils.checks import check_file
from cerberusdet.utils.ckpt_utils import dict_to_cerber, intersect_dicts
from cerberusdet.utils.general import check_dataset, colorstr, labels_to_class_weights, strip_optimizer
from cerberusdet.utils.mlflow_logging import MLFlowLogger
from cerberusdet.utils.plots import plot_images
from cerberusdet.utils.torch_utils import (
    ModelEMA,
    de_parallel,
    get_hyperparameter,
    is_parallel,
    model_info,
    set_hyperparameter,
    torch_distributed_zero_first,
)
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """
    Model loading, saving. Logs writing and images plotting during training.
    """

    def __init__(self, hyp, opt, rank, local_rank):
        # Directories
        save_dir = opt.save_dir
        self.save_dir = Path(save_dir)
        wdir = self.save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last = wdir / "last.pt"
        self.best = wdir / "best.pt"
        self.results_file = self.save_dir / "results.txt"

        self.opt = opt
        self.rank = rank

        # Save run settings
        if not self.opt.evolve:
            with open(self.save_dir / "opt.yaml", "w") as f:
                yaml.safe_dump(vars(opt), f, sort_keys=False)

        # Hyperparameters
        hyp = self.get_hyp(hyp)

        with open(opt.data) as f:
            data_dict = yaml.safe_load(f)  # data dict

        with torch_distributed_zero_first(local_rank):
            check_dataset(data_dict)  # check

        if isinstance(data_dict["train"], list):
            self.num_tasks = len(data_dict["train"])
            self.task_ids = data_dict["task_ids"]
        else:
            self.num_tasks = 1
            data_dict["train"] = [data_dict["train"]]
            assert not isinstance(data_dict["val"], list) or len(data_dict["val"]) == 1
            data_dict["val"] = [data_dict["val"]]
            if data_dict.get("task_ids") is None or len(data_dict["task_ids"]) != 1:
                data_dict["task_ids"] = ["detection"]

            self.task_ids = data_dict["task_ids"]

        assert len(np.unique(self.task_ids)) == self.num_tasks

        # Loggers
        if rank in [-1, 0]:
            self.loggers = self.get_loggers(hyp)

        weights, epochs = opt.weights, opt.epochs

        for i in range(self.num_tasks):
            task_nc = int(data_dict["nc"]) if not isinstance(data_dict["nc"], list) else int(data_dict["nc"][i])
            task_nc = 1 if self.opt.single_cls else task_nc  # number of classes

            task_names = data_dict["names"] if not isinstance(data_dict["nc"], list) else data_dict["names"][i]
            task_names = ["item"] if self.opt.single_cls and len(task_names) != 1 else task_names

            if not isinstance(data_dict["nc"], list):
                # one task training
                data_dict["nc"] = [task_nc]
                data_dict["names"] = [task_names]
            else:
                data_dict["nc"][i] = task_nc
                data_dict["names"][i] = task_names

        self.data_dict = data_dict
        self.ckpt = None
        self.weights = weights
        self.hyp = hyp
        self.epochs = epochs

        LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in self.hyp.items()))

    def get_hyp(self, hyp, dump_name="hyp.yaml"):

        if isinstance(hyp, str):
            hyp = check_file(hyp)

        # Save run settings
        if not self.opt.evolve:
            with open(self.save_dir / dump_name, "w") as f:
                yaml.safe_dump(hyp, f, sort_keys=False)

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp) as f:
                hyp = yaml.safe_load(f)  # load hyps dict

        return hyp

    def fill_tasks_parameters(self, nl, imgsz, model, datasets, device):
        model.names = dict()
        model.class_weights = dict()

        for task_i, (task, dataset) in enumerate(zip(self.task_ids, datasets)):
            nc = self.data_dict["nc"][task_i]

            box_w = get_hyperparameter(self.hyp, "box", task_i, task)
            cls_w = get_hyperparameter(self.hyp, "cls", task_i, task)

            box_w *= 3.0 / nl  # scale to layers
            cls_w *= (imgsz / 640) ** 2 * 3.0 / nl  # scale to image size and layers

            set_hyperparameter(self.hyp, "box", box_w, task_i, task)
            set_hyperparameter(self.hyp, "cls", cls_w, task_i, task)

            if not isinstance(dataset, torch.utils.data.Subset):
                model.class_weights[task] = (
                    labels_to_class_weights(dataset.labels, nc).to(device) * nc
                )  # attach class weights
            model.names[task] = self.data_dict["names"][task_i]

        model.nc = dict()
        for task, nc in zip(self.task_ids, self.data_dict["nc"]):
            model.nc[task] = nc  # attach number of classes to model
        model.hyp = self.hyp  # attach hyperparameters to model

        if is_parallel(model):
            model.yaml = de_parallel(model).yaml
            model.stride = de_parallel(model).stride
            de_parallel(model).nc = model.nc
            de_parallel(model).hyp = self.hyp  # attach hyperparameters to model

    def from_ckpt(self, ckpt, model, exclude=[]):

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"].float().state_dict()  # to FP32
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            state_dict = ckpt.state_dict()
        loaded = False

        if "blocks." in list(model.state_dict().keys())[0] and "blocks." not in list(state_dict.keys())[0]:
            # if loading weights from yolov5 to cerbernet
            state_dict = dict_to_cerber(state_dict, model)  # intersect
            loaded = True

            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            LOGGER.info("Transferred %g/%g items" % (len(state_dict), len(model.state_dict())))  # report

        return state_dict, loaded

    def load_model(self, cfg, device, verbose=True):
        pretrained = self.weights.endswith(".pt")

        loaded = False
        if self.rank in [-1, 0]:
            print(self.hyp)

        if pretrained:
            LOGGER.info(f"Trying to restore weights from {self.weights} ...")

            self.ckpt = torch.load(self.weights, map_location=device)  # load checkpoint
            exclude = ["anchor"] if (cfg or self.hyp.get("anchors")) and not self.opt.resume else []  # exclude keys

            model = CerberusDet(
                task_ids=self.task_ids,
                nc=self.data_dict["nc"],
                cfg=cfg or self.ckpt["model"].yaml,
                ch=3,
                verbose=verbose,
            ).to(device)

            state_dict, loaded = self.from_ckpt(self.ckpt, model, exclude)
        else:
            model = CerberusDet(
                task_ids=self.task_ids,
                nc=self.data_dict["nc"],
                cfg=cfg,
                ch=3,
                verbose=verbose,
            ).to(device)

        # Do not move: its important to try to load pretrained model before splittig model
        if model.yaml.get("cerber") and len(model.yaml["cerber"]):
            cerber_schedule = model.yaml["cerber"]
            # cerber_schedule = [[0, [[17], [15, 16]]], [4, [[15], [16]]]]
            if self.rank in [-1, 0] and self.loggers["mlflow"]:
                self.loggers["mlflow"].log_params({"cerber": cerber_schedule})
            model.sequential_split(deepcopy(cerber_schedule), device)
            if verbose and self.rank in [-1, 0]:
                print(model.info())

        if verbose and self.rank in [-1, 0]:
            model_info(model)

        if pretrained and not loaded:
            # if loading weights from pretrained cerbernet
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            LOGGER.info(
                "Transferred %g/%g items from %s" % (len(state_dict), len(model.state_dict()), self.weights)
            )  # report

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        freeze_model(model, freeze)
        ema = ModelEMA(model) if self.rank in [-1, 0] else None

        if self.ckpt:

            # EMA
            if ema and self.ckpt.get("ema"):
                LOGGER.info("Loading ema from ckpt..")
                ema.ema.load_state_dict(self.ckpt["ema"].float().state_dict())
                ema.updates = self.ckpt["updates"]

            # Results
            if self.ckpt.get("training_results") is not None:
                self.results_file.write_text(self.ckpt["training_results"])  # write results.txt

            # Epochs
            start_epoch = self.ckpt.get("epoch", -1) + 1
            if self.opt.resume:
                assert start_epoch > 0, "%s training to %g epochs is finished, nothing to resume." % (
                    self.weights,
                    self.epochs,
                )

            if self.epochs < start_epoch:
                LOGGER.info(
                    "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                    % (self.weights, self.ckpt["epoch"], self.epochs)
                )
                self.epochs += self.ckpt["epoch"]  # finetune additional epochs

        return model, ema

    def save_model(
        self,
        epoch,
        best_fitness_per_task,
        best_fitness,
        model,
        ema,
        optimizer_state_dict,
        is_best=False,
    ):
        if ema:
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        ckpt = self._get_ckpt_to_save(epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict)

        torch.save(ckpt, self.last)
        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_model(str(self.last))

        if is_best:
            torch.save(ckpt, self.best)
            if self.loggers["mlflow"]:
                self.loggers["mlflow"].log_model(str(self.best))

    def save_best_task_model(
        self, task_name, epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict
    ):

        ckpt = self._get_ckpt_to_save(epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict)

        best_path = self.save_dir / "weights" / f"{task_name}_best.pt"
        torch.save(ckpt, best_path)
        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_model(str(best_path))

    def _get_ckpt_to_save(self, epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict):
        training_results = self.results_file.read_text() if self.results_file.exists() else None
        ckpt = {
            "epoch": epoch,
            "best_fitness_per_task": best_fitness_per_task,
            "best_fitness": best_fitness,
            "training_results": training_results,
            "model": deepcopy(de_parallel(model)).half(),
            "ema": deepcopy(ema.ema).half(),
            "updates": ema.updates,
            "optimizer": optimizer_state_dict,
        }
        return ckpt

    def strip_optimizer(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers

    def log_models(self):
        if self.loggers["mlflow"] and not self.opt.evolve:
            if self.best.exists():
                self.loggers["mlflow"].log_model(str(self.best))
            if self.last.exists():
                self.loggers["mlflow"].log_model(str(self.last))

    def train_log(
        self,
        task,
        lr,
        mloss,
        epoch,
        last_stat,
        scale_t=None,
        tag_prefix="",
        tags=("box_loss", "obj_loss", "cls_loss", "lr0", "lr1", "lr2"),
    ):
        """Log train info into tb, mlflow, local txt file"""

        mlflow_metrics = {}
        cnt = 0

        # log loss and lr params
        for full_prefix, param_group in zip(
            [f"{tag_prefix}train/{task}/", f"{tag_prefix}x/{task}/"], [list(mloss[:-1]), lr]
        ):
            n_group_params = len(param_group)
            group_tags = tags[cnt : (cnt + n_group_params)]
            for x, tag in zip(param_group, group_tags):
                full_tag = f"{full_prefix}{tag}"
                if self.loggers["tb"]:
                    self.loggers["tb"].add_scalar(full_tag, x, epoch)  # TensorBoard
                if self.loggers["mlflow"]:  # MLFlow
                    mlflow_metrics[full_tag.replace("/", "_")] = (
                        float(x.cpu().numpy()) if isinstance(x, torch.Tensor) else x
                    )
            cnt += n_group_params

        if scale_t:
            tag = f"{tag_prefix}{task}/scale"
            if self.loggers["tb"]:
                self.loggers["tb"].add_scalar(tag, scale_t, epoch)  # TensorBoard
            if self.loggers["mlflow"]:  # MLFlow
                mlflow_metrics[tag.replace("/", "_")] = (
                    float(scale_t.cpu().numpy()) if isinstance(scale_t, torch.Tensor) else scale_t
                )

        with open(self.results_file, "a") as f:
            # 'Task', 'epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'
            f.write(f"Train {task}: " + last_stat + "\n")  # append train metrics

        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_metrics(mlflow_metrics, step=epoch)

    def val_log(self, task, results, epoch, is_best):

        with open(self.results_file, "a") as f:
            f.write(f"Val {task}: " + "%10.4g" * 7 % results + "\n")  # append val metrics, val_loss

        mlflow_metrics = {}

        # Log
        tags = [
            f"metrics/{task}/precision",
            f"metrics/{task}/recall",
            f"metrics/{task}/mAP_0.5",
            f"metrics/{task}/mAP_0.5:0.95",
            f"val/{task}/box_loss",
            f"val/{task}/obj_loss",
            f"val/{task}/cls_loss",  # val loss
        ]

        for x, tag in zip(list(results), tags):
            if self.loggers["tb"]:
                self.loggers["tb"].add_scalar(tag, x, epoch)  # TensorBoard
            if self.loggers["mlflow"]:  # MLFlow
                mlflow_metrics[tag.replace("/", "_").replace(":", "_")] = (
                    float(x.cpu().numpy()) if isinstance(x, torch.Tensor) else x
                )

        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_metrics(mlflow_metrics, step=epoch)

    def plot_train_images(self, ni, task, batch, model):
        imgs = batch["img"]
        if ni < 3:
            plot_images(
                images=batch["img"],
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=batch["bboxes"],
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch{ni}_{task}.jpg",
                mlflow_logger=self.loggers["mlflow"],
            )

            if self.loggers["tb"] and ni == 0:  # TensorBoard
                if not self.opt.sync_bn:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # suppress jit trace warning
                        if hasattr(model, "set_task"):
                            model.set_task(task)
                        self.loggers["tb"].add_graph(torch.jit.trace(model, imgs[0:1], strict=False), [])

    def get_loggers(self, hyp, include=("tb",)):
        loggers = {"tb": None, "mlflow": None}  # loggers dict

        # TensorBoard
        if not self.opt.evolve and "tb" in include:
            prefix = colorstr("tensorboard: ")
            LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {self.opt.project}', view at http://localhost:6006/")
            loggers["tb"] = SummaryWriter(str(self.save_dir))

        # MLFlow
        if self.opt.mlflow_url:
            loggers["mlflow"] = MLFlowLogger(self.opt, hyp)
        else:
            LOGGER.info("MLFlow logger will not be used")
            loggers["mlflow"] = None

        self.opt.hyp = hyp  # add hyperparameters

        return loggers


def freeze_model(model, freeze):
    for k, v in model.named_parameters():
        if v.requires_grad:
            v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False
