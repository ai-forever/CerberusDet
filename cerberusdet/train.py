import argparse
import logging
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import yaml
from cerberusdet import val
from cerberusdet.data.dataloaders import _create_dataloader
from cerberusdet.data.datasets import LoadImagesAndLabels
from cerberusdet.evolvers.predefined_evolvers import EVOLVER_TYPES
from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.models.experimental import attempt_load
from cerberusdet.trainers.averaging import Averaging
from cerberusdet.utils.checks import check_file, check_git_status, check_requirements
from cerberusdet.utils.general import check_img_size, colorstr, get_latest_run, increment_path, init_seeds, set_logging
from cerberusdet.utils.models_manager import ModelManager
from cerberusdet.utils.plots import plot_labels
from cerberusdet.utils.torch_utils import select_device
from cerberusdet.utils.train_utils import create_data_loaders, get_init_metrics_per_task
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

torch.backends.cudnn.enabled = False
warnings.filterwarnings(
    "ignore", message="torch.distributed._all_gather_base is a private function and will be deprecated"
)

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
ROOT = Path(__file__).absolute().parents[1]


def train(
    hyp,  # path/to/hyp.yaml or hyp dictionary
    opt,
    device,
    train_dataset: Optional[List[LoadImagesAndLabels]] = None,
    val_dataset: Optional[List[LoadImagesAndLabels]] = None,
):
    """
    Returns
        results_per_task: Dict[str, Tuple[float]] - calculated metrics per task in the order
        [P, R, mAP@0.5, mAP@0.5:0.95, val_box, val_cls, val_dfl]

        e.g. {
            "clothes": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            "shoes":   (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        }
    """

    evolve, cfg = opt.evolve, opt.cfg

    # Configure
    plots = not evolve  # create plots
    cuda = device.type != "cpu"
    init_seeds(1 + RANK)

    model_manager = ModelManager(hyp, opt, RANK, LOCAL_RANK)
    save_dir = model_manager.save_dir
    hyp = model_manager.hyp

    # Model, EMA
    verbose_model = not evolve
    model, ema = model_manager.load_model(cfg, device, verbose_model)

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    if isinstance(model, CerberusDet):
        nl = None
        for task_name in model.heads:
            head = model.get_head(task_name)
            if nl is None:
                nl = head.nl
            else:
                assert nl == head.nl
            assert hasattr(head, "stride")

    else:
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs)  # verify imgsz is gs-multiple

    # Trainloader
    # dataset: train dataset
    if train_dataset is None:
        train_loader, val_loader, dataset, _ = create_data_loaders(
            model_manager.data_dict, RANK, WORLD_SIZE, opt, hyp, gs, imgsz, balanced_sampler=True
        )
    else:
        # update dataset hyperparams
        train_loader = []
        dataset = train_dataset
        for train_task_dataset in dataset:
            train_task_dataset.update_hyp(hyp)
            train_task_dataloader = _create_dataloader(
                train_task_dataset,
                workers=opt.workers,
                batch_size=opt.batch_size,
                rank=RANK,
                use_balanced_sampler=True,
            )
            train_loader.append(train_task_dataloader)

        val_loader = []
        for val_task_dataset in val_dataset:
            val_task_dataloader = _create_dataloader(
                val_task_dataset,
                workers=opt.workers,
                batch_size=opt.batch_size,
                rank=-1,
                use_balanced_sampler=False,
            )
            val_loader.append(val_task_dataloader)

    assert len(dataset) == model_manager.num_tasks

    # Optimizer
    trainer = Averaging(device, model, model_manager, train_loader, val_loader, dataset, imgsz, gs)

    # Resume
    start_epoch = trainer.resume(model_manager.ckpt)

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logger.warning(
            "DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        if RANK in [-1, 0]:
            logger.info("Using SyncBatchNorm()")

    if RANK in [-1, 0] and not model_manager.opt.resume:
        for ii, (task_dataset, task_name) in enumerate(zip(dataset, model_manager.task_ids)):

            task_nc = model_manager.data_dict["nc"][ii]  # number of classes
            names = model_manager.data_dict["names"][ii]
            assert len(names) == task_nc, "%g names found for nc=%g dataset in %s" % (
                len(names),
                task_nc,
                opt.data,
            )  # check

            labels = np.concatenate(task_dataset.labels, 0)
            if plots:
                plot_labels(labels, names, save_dir, model_manager.loggers, name=task_name)

        model.half().float()  # pre-reduce anchor precision

    if RANK in [-1, 0] and model_manager.loggers["mlflow"] and not evolve:
        model_manager.loggers["mlflow"].save_artifacts(save_dir)

    for ii, (task_dataset, task_name) in enumerate(zip(dataset, model_manager.task_ids)):
        labels = np.concatenate(task_dataset.labels, 0)

        if labels.shape[1] == 6:
            mlc = labels[:, 0].max()  # max label class
        elif labels.shape[1] == 7:
            mlc = labels[:, 1].max()

        task_nc = model_manager.data_dict["nc"][ii]
        assert mlc < task_nc, "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (
            mlc,
            task_nc,
            opt.data,
            task_nc - 1,
        )

    # DDP mode
    if cuda and RANK != -1:
        logger.info(f"Using DDP on gpu {LOCAL_RANK}")
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)

    # Model parameters
    model_manager.fill_tasks_parameters(nl, imgsz, model, trainer.dataset, device)

    # Start training
    t0 = time.time()

    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results_per_task = get_init_metrics_per_task(model_manager)

    # do not move
    trainer.set_loss(model)
    if isinstance(trainer.scheduler, dict):
        for _, scheduler in trainer.scheduler.items():
            scheduler.last_epoch = start_epoch - 1
    else:
        trainer.scheduler.last_epoch = start_epoch - 1

    if RANK in [-1, 0]:
        logger.info(
            f"Image sizes {imgsz} train, {imgsz} val\n"
            f"Using {train_loader[0].num_workers} dataloader workers\n"
            f"Logging results to {model_manager.save_dir}\n"
            f"Starting training for {model_manager.epochs} epochs..."
        )

    # Train and val
    for epoch in range(start_epoch, model_manager.epochs):

        prev_best_fitness = trainer.best_fitness
        trainer.train_epoch(model, ema, epoch, RANK, WORLD_SIZE)

        # DDP process 0 or single-GPU
        if RANK in [-1, 0]:
            results_per_task = trainer.val_epoch(model, ema, epoch, WORLD_SIZE)

            if trainer.best_fitness > prev_best_fitness and not evolve and not opt.nosave:
                logger.info("Best model updated")

            stop = trainer.stopper(epoch=epoch, fitness=trainer.last_fitness)
            if stop:
                break

    # end training
    if RANK in [-1, 0]:
        logger.info(f"{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n")

        if not evolve:
            for task_i, (task, val_loader) in enumerate(zip(trainer.task_ids, trainer.val_loaders)):

                ckpts = (
                    [model_manager.last, model_manager.best] if model_manager.best.exists() else [model_manager.last]
                )
                for m in ckpts:  # speed, mAP tests
                    results_per_task[task] = val.run(
                        model_manager.data_dict,
                        batch_size=trainer.batch_size,
                        imgsz=imgsz,
                        model=attempt_load(m, device).half(),
                        single_cls=model_manager.opt.single_cls,
                        dataloader=val_loader,
                        save_dir=model_manager.save_dir,
                        plots=True,
                        verbose=True,
                        task_id=task,
                        task_ind=task_i,
                        labels_from_xml=opt.labels_from_xml,
                        use_multi_labels=opt.use_multi_labels,
                        use_soft_labels=opt.use_soft_labels,
                    )[0]

                    if model_manager.loggers["mlflow"] and m == model_manager.best:
                        model_manager.val_log(task, results_per_task[task], epoch, is_best=True)

        # Strip optimizers
        model_manager.strip_optimizer()
        model_manager.log_models()

        if model_manager.loggers["mlflow"]:
            if not evolve:
                model_manager.loggers["mlflow"].save_artifacts(model_manager.save_dir)
                if model_manager.best.exists():
                    half_best_model = attempt_load(model_manager.best, device).half()
                    model_manager.loggers["mlflow"].log_model_signature(half_best_model, imgsz, device, "best_model")
                    model_manager.loggers["mlflow"].log_best_model_md5(str(model_manager.best), "best_model")
            else:
                # Save nothing here, artifacts will be saved later in the evolver object
                pass
            model_manager.loggers["mlflow"].finish_run()

    torch.cuda.empty_cache()
    return results_per_task, epoch


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="pretrained/yolov8x_state_dict.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="cerberusdet/models/yolov8x.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="data/voc_obj365.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="data/hyps/hyp.cerber-voc_obj365.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for one GPU")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolver", type=str, default="yolov5", help="Evolve algo to use", choices=EVOLVER_TYPES + ["yolov5"]
    )
    parser.add_argument(
        "--params_to_evolve",
        type=str,
        default=None,
        help="Parameters to find separated by comma",
    )
    parser.add_argument(
        "--evolve_per_task", action="store_true", help="whether to evolve params specified per task differently"
    )
    parser.add_argument("--cache-images", action="store_true", help="cache images for faster training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=16, help="maximum number of dataloader workers")
    parser.add_argument("--project", default=str(ROOT / "runs/train"), help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument("--experiment_name", type=str, default="cerberus_exp", help="MlFlow experiment name")
    parser.add_argument("--patience", type=int, default=30, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument(
        "--mlflow-url",
        type=str,
        default=None,
        help="Param for mlflow.set_tracking_uri(), may be 'local'",
    )
    parser.add_argument("--local-rank", type=int, default=-1, help="DDP parameter, do not modify")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument(
        "--use-multi-labels", action="store_true", help="Loading multiple labels for boxes, if available"
    )
    parser.add_argument("--use-soft-labels", action="store_true", help="Class probability based on annotation votes")
    parser.add_argument("--labels-from-xml", action="store_true", help="Load labels from xml files")
    parser.add_argument(
        "--freeze-shared-till-epoch",
        type=int,
        default=0,
        help="Freeze shared between all tasks params for first N epochs",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr("train: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
        check_git_status()
        check_requirements(exclude=["thop"])

    # Resume
    if opt.resume and not opt.evolve:  # resume an interrupted run
        ckpt = (
            opt.resume if isinstance(opt.resume, str) else get_latest_run(opt.project)
        )  # specified or most recent path
        logger.info(f"Resume from {ckpt}")
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(Path(ckpt).parent.parent / "opt.yaml") as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = "", ckpt, True  # reinstate

        LOGGER.info(f"Resuming training from {ckpt}")
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.name = f"{opt.evolver}_{opt.name}"
        opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if RANK in [-1, 0]:
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True))
        else:
            opt.save_dir = ""

    # DDP mode
    if LOCAL_RANK != -1:
        from datetime import timedelta

        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10000)
        )
    else:
        device = select_device(opt.device, batch_size=opt.batch_size)

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            print("Destroying process group... ")
            dist.destroy_process_group()
            print("Done.")

    # Evolve hyperparameters (optional)
    else:
        from evolvers import RayEvolver, Yolov5Evolver

        if opt.evolver == "yolov5":
            evolver = Yolov5Evolver(opt, device)
        else:
            evolver = RayEvolver(opt, device)
        evolver.run_evolution(train)


def run(**kwargs):
    # Usage: from cerberusdet import train; train.run(imgsz=640, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
