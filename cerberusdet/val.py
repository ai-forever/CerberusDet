import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from cerberusdet.data.dataloaders import create_dataloader
from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.models.experimental import attempt_load
from cerberusdet.utils.checks import check_file, check_requirements
from cerberusdet.utils.general import (
    box_iou,
    check_dataset,
    check_img_size,
    colorstr,
    increment_path,
    non_max_suppression,
    scale_boxes,
    set_logging,
    strip_optimizer,
    xywh2xyxy,
)
from cerberusdet.utils.metrics import ConfusionMatrix, DetMetrics
from cerberusdet.utils.mlflow_logging import MLFlowLogger
from cerberusdet.utils.plots import output_to_target, plot_images
from cerberusdet.utils.torch_utils import Profile, model_info, select_device
from tqdm import tqdm

# torch.backends.cudnn.enabled = False


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=detections.device)


def preprocess(batch, half, device):
    batch["img"] = batch["img"].to(device, non_blocking=True)
    batch["img"] = (batch["img"].half() if half else batch["img"].float()) / 255
    for k in ["batch_idx", "cls", "prob", "bboxes"]:
        batch[k] = batch[k].to(device)

    return batch


def plot_val_samples(batch, ni, task_id, save_dir, names):
    plot_images(
        batch["img"],
        batch["batch_idx"],
        batch["cls"].squeeze(-1),
        batch["bboxes"],
        paths=batch["im_file"],
        fname=save_dir / f"val_batch{ni}_labels_{task_id}.jpg",
        names=names,
    )


def plot_predictions(batch, preds, ni, task_id, save_dir, names):
    plot_images(
        batch["img"],
        *output_to_target(preds, max_det=15),
        paths=batch["im_file"],
        fname=save_dir / f"val_batch{ni}_pred_{task_id}.jpg",
        names=names,
    )  # pred


def get_stats(stats, metrics: DetMetrics, nc: int):
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        metrics.process(*stats)
    nt_per_class = np.bincount(stats[-1].astype(int), minlength=nc)  # number of targets per class
    return metrics.results_dict, nt_per_class


def print_results(task: str, stats: list, metrics: DetMetrics, nt_per_class, nc, names, seen, verbose, training):
    pf = "%22s" * 2 + "%11i" * 2 + "%11.3g" * len(metrics.keys)  # print format
    print(pf % (task, "all", seen, nt_per_class.sum(), *metrics.mean_results()))
    if nt_per_class.sum() == 0:
        print("WARNING ⚠️ no labels found in dataset, can not compute metrics without labels")

    # Print results per class
    if (verbose or not training) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            print(pf % (task, names[c], seen, nt_per_class[c], *metrics.class_result(i)))


def update_mlflow_metrics(mlflow_log_metrics, task_id, metrics, names, verbose=False):
    mp, mr, map50, map = metrics.mean_results()
    mlflow_log_metrics.update(
        {
            f"metrics_{task_id}_precision": mp,
            f"metrics_{task_id}_recall": mr,
            f"metrics_{task_id}_mAP_0.5": map50,
            f"metrics_{task_id}_mAP_0.5_0.95": map,
        }
    )

    if verbose:
        for i, c in enumerate(metrics.ap_class_index):
            mlflow_log_metrics[f"metrics_{task_id}_{names[c]}_AP_0.5"] = metrics.class_result(i)[2]


def unique_with_index(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index


@torch.no_grad()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    single_cls=False,  # treat as single-class dataset
    verbose=False,  # verbose output
    project="runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    compute_loss=None,
    task_id=None,
    task_ind=None,
    mlflow_url=None,
    labels_from_xml=False,
    use_multi_labels=False,
    use_soft_labels=False,
    experiment_name=None,
):
    # save_json=True
    # Initialize/load model and set device
    training = model is not None
    mlflow_logger = None
    mlflow_log_metrics = {}
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
        task_ids = data["task_ids"]

    else:  # called directly

        if mlflow_url:
            assert experiment_name is not None
            mlflow_logger = MLFlowLogger(parse_opt())

        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True)  # increment run

        # Load model
        model = attempt_load(weights, map_location=device, mlflow_url=mlflow_url)  # load FP32 model
        print(f"Loaded model {type(model)} from {weights}:")
        print(model.info())
        model_info(model)

        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Data
        with open(data) as f:
            data = yaml.safe_load(f)

        if isinstance(data["train"], list):
            num_tasks = len(data["train"])
            task_ids = data["task_ids"]
            data["nc"] = [1 if single_cls else int(data["nc"][i]) for i in range(num_tasks)]
        else:
            task_ids = ["detection"]
            data["train"] = [data["train"]]
            data["val"] = [data["val"]]
            data["task_ids"] = task_ids
            nc = 1 if single_cls else int(data["nc"])  # number of classes
            data["nc"] = [nc]

        check_dataset(data)  # check

    # Half
    half &= device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    inference_all = task == "speed" and task_id is None

    # Dataloader
    if not training:

        if device.type != "cpu":
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())), task_ids[0])  # run once
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images

        assert isinstance(data[task], list)
        assert len(data[task]) == len(task_ids)
        dataloaders = []
        for _task_ind, (_task_id, data_path) in enumerate(zip(task_ids, data[task])):
            dataloader, dataset = create_dataloader(
                data_path,
                imgsz,
                batch_size,
                gs,
                single_cls,
                pad=0.5,
                rect=True,
                prefix=colorstr(f"{task}: "),
                task_ind=_task_ind,
                classnames=data["names"][_task_ind],
                labels_from_xml=labels_from_xml,
                as_multi_label=use_multi_labels,
                as_soft_label=use_soft_labels,
            )  # , skip_prefix="products")
            print(f"Dataset {_task_id} length: ", len(dataset))
            dataloaders.append(dataloader)
        tasks_nc = data["nc"]
        if task_id is not None:
            _task_ind = task_ids.index(task_id)
            if task_ind is not None:
                assert _task_ind == task_ind
            tasks_nc = [tasks_nc[_task_ind]]
            task_ids = [task_ids[_task_ind]]
            dataloaders = [dataloaders[_task_ind]]
    else:
        assert task_id is not None and task_ind is not None
        dataloaders = [dataloader]
        tasks_nc = [data["nc"][task_ind]]
        task_ids = [data["task_ids"][task_ind]]

    total_cnt = 0
    mean_inference_time = Profile()
    for task_i, (task_id, nc, dataloader) in enumerate(zip(task_ids, tasks_nc, dataloaders)):
        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc)
        names = model.names if hasattr(model, "names") else model.module.names

        if isinstance(names, list):
            names = {task_id: names}
        names = {k: v for k, v in enumerate(names[task_id])}

        metrics = DetMetrics(save_dir=save_dir)
        metrics.names = names
        metrics.plot = plots

        s = ("%20s" * 2 + "%11s" * 6) % ("Task", "Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95")
        loss = torch.zeros(3, device=device)
        stats = []

        dt = Profile(), Profile(), Profile(), Profile()
        for batch_i, batch in enumerate(tqdm(dataloader, desc=s)):

            if single_cls and use_multi_labels:
                # In order not to count the same box several times in the metric
                _, _, _, index = unique_with_index(batch["bboxes"], dim=0)
                batch["cls"] = batch["cls"][index]
                batch["prob"] = batch["prob"][index]
                batch["bboxes"] = batch["bboxes"][index]
                batch["batch_idx"] = batch["batch_idx"][index]

            # pre-process
            with dt[0]:
                batch = preprocess(batch, half, device)

            # inference
            with dt[1]:
                if inference_all and isinstance(model, CerberusDet):
                    # inference all tasks
                    all_out = model(batch["img"])
                    assert all(t in all_out for t in task_ids), f"Invalid task_ids: {task_ids}"
                    out, train_out = all_out[task_id]
                else:
                    out, train_out = model(batch["img"], task_id)  # inference and training outputs

            if not (task_i == 0 and batch_i < 2):
                mean_inference_time.t += dt[1].dt

            # Compute loss
            with dt[2]:
                if compute_loss:
                    task_loss = compute_loss([x.float() for x in train_out], batch, task_id)[1][:3]
                    loss += task_loss  # box, obj, cls

            # Run NMS
            with dt[3]:
                out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, agnostic=single_cls)

            # Statistics per image
            for si, pred in enumerate(out):
                idx = batch["batch_idx"] == si
                cls, bbox = batch["cls"][idx], batch["bboxes"][idx]
                nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
                shape = batch["ori_shape"][si]
                correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                seen += 1
                if not (task_i == 0 and batch_i < 2):
                    total_cnt += 1

                if npr == 0:
                    if nl:
                        stats.append((correct_bboxes, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                        if plots:
                            confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                    continue

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_boxes(
                    batch["img"][si].shape[1:], predn[:, :4], shape, ratio_pad=batch["ratio_pad"][si]
                )  # native-space pred

                # Evaluate
                if nl:
                    height, width = batch["img"].shape[2:]
                    tbox = xywh2xyxy(bbox) * torch.tensor((width, height, width, height), device=device)  # target boxes
                    scale_boxes(
                        batch["img"][si].shape[1:], tbox, shape, ratio_pad=batch["ratio_pad"][si]
                    )  # native-space labels
                    labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                    correct_bboxes = process_batch(predn, labelsn, iouv)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            if plots and batch_i < 3:
                plot_val_samples(batch, batch_i, task_id, save_dir, names)
                plot_predictions(batch, out, batch_i, task_id, save_dir, names)

        # Compute statistics
        stats_dict, nt_per_class = get_stats(stats, metrics, nc)

        # Print results
        print_results(task_id, stats, metrics, nt_per_class, nc, names, seen, verbose, training)
        speed = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image

        # Print speeds
        if not training:
            shape = (batch_size, 3, imgsz, imgsz)
            print(
                f"Speed: %.1fms pre-process, %.1fms inference, %.1fms loss, %.1fms NMS per image at shape "
                f"{shape} and batch {batch_size} on {seen} runs" % speed
            )

        # Plots
        if plots:
            confusion_matrix.plot(
                save_dir=save_dir, names=list(names.values()), f_name=f"{task_id}_confusion_matrix.png"
            )

        if mlflow_logger and not training:
            update_mlflow_metrics(mlflow_log_metrics, task_id, metrics, names, verbose=nc > 1 and len(stats))

    if not training:
        print(f"Results saved to {save_dir}")

        if inference_all:
            mean_time = mean_inference_time.t / total_cnt * 1e3
            print(
                f"Mean inference time for all tasks: %.1fms per image. "
                f"Batch {batch_size}; {total_cnt} runs" % (mean_time,)
            )
        if mlflow_logger:
            mlflow_logger.save_artifacts(save_dir)

            w = str(Path(weights[0] if isinstance(weights, list) else weights))

            if not w.startswith("models:/"):
                strip_optimizer(w)

                mlflow_logger.save_artifact(w)
                # double check
                half_best_model = attempt_load(w, device, mlflow_url=mlflow_url).half().eval()
                mlflow_logger.log_model_signature(half_best_model, imgsz, device, "best_model")
                mlflow_logger.log_best_model_md5(str(w), "best_model")

            mlflow_logger.log_params(
                dict(
                    preprocess_time_in_ms=speed[0],
                    inference_time_in_ms=speed[1],
                    nms_time_in_ms=speed[2],
                )
            )

            mlflow_logger.log_metrics(mlflow_log_metrics, step=-1)

        return None

    # Return results for the last evaluated task
    # During training its always the only one
    if training:
        model.float()

        mp, mr, map50, map = metrics.mean_results()
        maps = np.zeros(nc) + map

        for i, c in enumerate(metrics.ap_class_index):
            maps[c] = metrics.class_result(i)[2]  # ap

        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, speed


def parse_opt():
    parser = argparse.ArgumentParser(prog="val.py")
    parser.add_argument("--data", type=str, default="data/voc_obj365.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", type=str, default="", help="model.pt path")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--task", default="val", help="train, val, test or speed")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", default="runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument(
        "--mlflow-url",
        type=str,
        default=None,
        help="Param for mlflow.set_tracking_uri(), may be 'local'",
    )
    parser.add_argument("--experiment_name", type=str, default="cerberus_exp", help="MlFlow experiment name")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument(
        "--use-multi-labels", action="store_true", help="Loading multiple labels for boxes, if available"
    )
    parser.add_argument("--use-soft-labels", action="store_true", help="Class probability based on annotation votes")
    parser.add_argument("--labels-from-xml", action="store_true", help="Load labels from xml files")

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    if not torch.backends.cudnn.enabled:
        print("Warning: Use cudnn to speed up inference")

    set_logging()
    print(colorstr("val: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))

    if opt.task in ("train", "val", "test"):  # run normally
        run(**vars(opt))

    elif opt.task == "speed":  # speed benchmarks
        run(
            opt.data,
            weights=opt.weights,
            batch_size=opt.batch_size,
            imgsz=opt.imgsz,
            plots=False,
            task="speed",
            use_multi_labels=opt.use_multi_labels,
            use_soft_labels=opt.use_soft_labels,
        )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
