import argparse
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from cerberusdet.data.dataset_images import LoadImages
from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.models.experimental import attempt_load
from cerberusdet.utils.checks import check_requirements
from cerberusdet.utils.general import (
    check_img_size,
    colorstr,
    increment_path,
    nms_between_tasks,
    non_max_suppression,
    save_one_box,
    scale_boxes,
    set_logging,
)
from cerberusdet.utils.metrics import box_iou
from cerberusdet.utils.plots import colors, create_label, plot_one_box
from cerberusdet.utils.torch_utils import Profile, select_device


def get_unique_bbox_indices(bboxes: torch.Tensor, iou_threshold: float = 1.0) -> List[List[int]]:
    # xyxy conf cls

    unique_bboxes: List[List[int]] = list()  # example [[0, 3], [1, 2]]

    for i, (*xyxy, _, cls) in enumerate(bboxes):
        if not unique_bboxes:
            unique_bboxes.append([0])

        else:
            added = False
            for j, unique_bbox in enumerate(unique_bboxes):
                bbox_index: int = unique_bbox[0]
                compared_xyxy: torch.Tensor = bboxes[bbox_index, :4]
                iou: float = box_iou(torch.vstack(xyxy).permute(1, 0), compared_xyxy[None])[0].item()

                if iou >= iou_threshold:
                    unique_bboxes[j].append(i)
                    added = True
                    break

            if not added:
                unique_bboxes.append([i])

    return unique_bboxes


def get_categories_map(class_names: Dict[str, List[str]]):
    categories_inds_map: Dict[str, Dict[int, int]] = {}
    # stack all categories
    all_class_names: List[str] = []

    tmp_categories_ids: List[List[int]] = []
    for task_name, task_categories in class_names.items():

        last_ind = tmp_categories_ids[-1][-1] + 1 if len(tmp_categories_ids) != 0 else 0
        cur_categories_ids = list(range(len(task_categories)))
        tmp_categories_ids.append([ind + last_ind for ind in cur_categories_ids])
        categories_inds_map[task_name] = {
            prev_id: new_id for prev_id, new_id in zip(cur_categories_ids, tmp_categories_ids[-1])
        }
        all_class_names.extend(task_categories)

    return categories_inds_map, all_class_names


def inference_image(
    model: CerberusDet,
    img: np.ndarray,
    device: torch.device,
    half: bool,
    conf_thres: float,
    iou_thres: float,
    agnostic_nms: bool,
    max_det: int,
) -> Dict[str, torch.Tensor]:

    results_per_task: Dict[str, torch.Tensor] = {}

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    # Inference
    dt = Profile()
    with dt:
        all_out = model(img)
        for task in all_out.keys():
            task_pred, _ = all_out[task]
            # Apply NMS
            pred = non_max_suppression(task_pred, conf_thres, iou_thres, agnostic=agnostic_nms, max_det=max_det)
            results_per_task[task] = pred[0]

    # Print time (inference + NMS)
    print(f"Inference + MNS done for 1 image. ({dt.t * 1e3:.3f}ms)")
    return results_per_task


def combine_output(output_per_task: Dict[str, torch.Tensor], categories_inds_map) -> torch.Tensor:
    output = torch.zeros((0, 6))
    # map category_ids and stack results per each image
    for task, bboxes in output_per_task.items():
        bboxes = bboxes.cpu()
        bboxes[:, 5].apply_(lambda cat: categories_inds_map[task][int(cat)])
        output = torch.cat((output, bboxes), 0)
    return output


@torch.no_grad()
def run(
    weights="yolov5s.pt",  # model.pt path(s)
    source="data/images",  # file/dir/URL/glob
    imgsz=640,  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    agnostic_nms=False,  # class-agnostic NMS
    project="runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    hide_task=False,  # hide task
    half=False,  # use FP16 half-precision inference,
    mlflow_url=None,
    iou_thres_between_tasks=0.8,
):
    save_img = not nosave and not source.endswith(".txt")  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True)  # increment run

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model: CerberusDet = attempt_load(weights, map_location=device, mlflow_url=mlflow_url)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    if half:
        model.half()  # to FP16

    # Configure
    model.eval()

    classnames: Dict[str, List[str]] = model.names if hasattr(model, "names") else model.module.names
    categories_inds_map, all_class_names = get_categories_map(classnames)
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    for path, img, im0s in dataset:

        results_per_task: Dict[str, torch.Tensor] = inference_image(
            model, img, device, half, conf_thres, iou_thres, agnostic_nms, max_det
        )

        # Process detections
        p, s, im0 = path, "", im0s.copy()
        save_path = str(save_dir / p.name)  # img.jpg
        s += "%gx%g " % img.shape[1:]

        # apply nms between objects of different tasks
        det: torch.Tensor = combine_output(results_per_task, categories_inds_map)
        det = nms_between_tasks(det, categories_inds_map, iou_thres=iou_thres_between_tasks)
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(img.shape[1:], det[:, :4], im0.shape).round()

        imc = im0.copy() if save_crop else im0  # for save_crop
        if len(det) == 0:
            continue

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {all_class_names[int(c)]}{'s' * (n > 1)}, "  # add to string

        unique_bboxes = get_unique_bbox_indices(det, iou_threshold=0.99)

        # Write results
        for box_indices in reversed(unique_bboxes):
            xyxys, probas, labels = det[box_indices].cpu().split((4, 1, 1), dim=1)
            labels = [int(x.item()) for x in labels]
            probas = [float(x.item()) for x in probas]
            tasks = []
            for lb in labels:
                for task in categories_inds_map:
                    task_cat_ids = categories_inds_map[task].values()
                    if lb in task_cat_ids:
                        tasks.append(task)
            xyxy = xyxys[0]

            if save_img or save_crop or view_img:  # Add bbox to image
                label = create_label(all_class_names, labels, probas, tasks, hide_labels, hide_conf, hide_task)
                plot_one_box(xyxy, im0, label=label, color=colors(labels[0], True), line_thickness=line_thickness)
                print(label)

                if save_crop:
                    save_one_box(
                        xyxy, imc, file=save_dir / "crops" / all_class_names[labels[0]] / f"{p.stem}.jpg", BGR=True
                    )

        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)
            print(f"Image was saved to {save_path}")

    if save_img:
        print(f"Results saved to {save_dir}")

    print(f"Done. ({time.time() - t0:.3f}s)")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="voc_obj365_v8x_best.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="data/images", help="file/dir/URL/glob")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image per task")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=0, type=int, help="bounding box thickness (pixels). 0 => auto")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--hide-task", default=False, action="store_true", help="hide task")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")

    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr("detect: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
