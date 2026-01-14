import torch
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np
import random

from cerberusdet.models.experimental import attempt_load
from cerberusdet.utils.general import (
    nms_between_tasks,
    non_max_suppression,
    scale_boxes,
    check_img_size,
)
from cerberusdet.utils.torch_utils import select_device
from cerberusdet.models.cerberus import CerberusDet


class CerberusDetInference:
    def __init__(
        self,
        weights: str,
        device: str = "",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        iou_thres_between_tasks: float = 0.8,
        half: bool = False,
        img_size: int = 640,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.iou_thres_between_tasks = iou_thres_between_tasks

        self.device = select_device(device)
        self.half = half & (self.device.type != "cpu")

        # Load model
        self.model: CerberusDet = attempt_load(weights, map_location=self.device)

        if self.half:
            self.model.half()
        self.model.eval()

        # Get model parameters needed for the outside world (e.g., for the preprocessor)
        self.stride = int(self.model.stride.max())
        self.names: Dict[str, List[str]] = self.model.names if hasattr(self.model, "names") else self.model.module.names

        # Create category mapping
        self.categories_inds_map, self.all_class_names = self._get_categories_map(self.names)

        # Warmup
        dummy_size = check_img_size(img_size, s=self.stride)
        dtype = next(self.model.parameters())
        dummy_img = torch.zeros(1, 3, dummy_size, dummy_size).to(self.device).type_as(dtype)
        self.model(dummy_img)

    def _get_categories_map(self, class_names: Dict[str, List[str]]):
        categories_inds_map: Dict[str, Dict[int, int]] = {}
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

    def _combine_output(self, output_per_task: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combines task results for a SINGLE image.
        """
        output = torch.zeros((0, 6))
        for task, bboxes in output_per_task.items():
            bboxes = bboxes.cpu()
            # Mapping local class IDs to global ones
            if bboxes.shape[0] > 0:
                bboxes[:, 5].apply_(lambda cat: self.categories_inds_map[task][int(cat)])
                output = torch.cat((output, bboxes), 0)
        return output

    @torch.no_grad()
    def predict(
        self,
        tensor: torch.Tensor,
        original_shape: Union[Tuple[int, int], List[Tuple[int, int]], None] = None,
        max_det: int = 300,
        agnostic_nms: bool = False,
        conf_thres: float = None,
        iou_thres: float = None,
        iou_thres_between_tasks: float = None,
    ) -> List[List[Dict]]:
        """
        Args:
            tensor: Prepared tensor [Batch, 3, H, W].
            original_shape:
                - Tuple (H, W) if all images in the batch have the same original size.
                - List[(H, W), ...] if each image has its own original size.
                - None if coordinate scaling is not required.
            max_det: maximum detections per image per task
            agnostic_nms: class-agnostic NMS

        Returns:
            List[List[Dict]]: The outer list corresponds to the batch.
                              The inner list contains detections for a specific image.
        """

        conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        iou_thres = iou_thres if iou_thres is not None else self.iou_thres
        iou_thres_between_tasks = iou_thres_between_tasks if iou_thres_between_tasks is not None else self.iou_thres_between_tasks

        # 1. Forward
        # model() returns dict: {task_name: (predictions, ...)}
        all_out = self.model(tensor)

        batch_size = tensor.shape[0]

        # Dictionary to store "raw" NMS results for each task across the entire batch
        # task_name -> List[Tensor] (list length = batch_size)
        nms_results_per_task_batch: Dict[str, List[torch.Tensor]] = {}

        for task in all_out.keys():
            task_pred, _ = all_out[task]
            # non_max_suppression returns a list of tensors of length batch_size
            preds_batch = non_max_suppression(
                task_pred,
                conf_thres,
                iou_thres,
                agnostic=agnostic_nms,
                max_det=max_det
            )
            nms_results_per_task_batch[task] = preds_batch

        final_batch_results = []

        # 2. Process each image in the batch individually
        for i in range(batch_size):
            # Collect detections from all tasks for the current image i
            # task_name -> Tensor (detections for img[i])
            current_img_task_results = {
                task: preds_list[i]
                for task, preds_list in nms_results_per_task_batch.items()
            }

            # 3. Combine & Cross-task NMS
            det = self._combine_output(current_img_task_results)
            det = nms_between_tasks(det, self.categories_inds_map, iou_thres=iou_thres_between_tasks)

            # 4. Scale coords to original image
            if len(det) > 0 and original_shape is not None:
                # Determine shape for the current image
                if isinstance(original_shape, list):
                    curr_shape = original_shape[i]
                else:
                    curr_shape = original_shape

                # Scale: tensor.shape[2:] is (H_net, W_net) of the current tensor
                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], curr_shape).round()

            # 5. Format output for current image
            image_results = []
            if len(det) > 0:
                for *xyxy, conf, cls in det:
                    c = int(cls)

                    # Find task name
                    detected_task = "unknown"
                    for task_name, mapping in self.categories_inds_map.items():
                        if c in mapping.values():
                            detected_task = task_name
                            break

                    image_results.append({
                        "box": [int(x.item()) for x in xyxy],
                        "score": float(conf.item()),
                        "label": c,
                        "label_name": self.all_class_names[c],
                        "task": detected_task
                    })

            final_batch_results.append(image_results)

        return final_batch_results


class CerberusVisualizer:
    def __init__(self, line_thickness: int = 3, text_scale: float = 0.6):
        """
        Class for rendering detection results.

        Args:
            line_thickness: Thickness of bounding box lines.
            text_scale: Font size of labels.
        """
        self.line_thickness = line_thickness
        self.text_scale = text_scale
        self.colors = {}  # Color cache for classes

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Generates a stable color for each class_id (BGR format)."""
        if class_id not in self.colors:
            random.seed(class_id)
            # Generate bright colors
            b = random.randint(50, 255)
            g = random.randint(50, 255)
            r = random.randint(50, 255)
            self.colors[class_id] = (b, g, r)
        return self.colors[class_id]

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        hide_labels: bool = False,
        hide_conf: bool = False,
        hide_task: bool = False,
    ) -> np.ndarray:
        """
        Draws boxes and labels on the image.

        Args:
            image: Original image (BGR, numpy array). It will be copied.
            detections: List of results from the predict() method.
            hide_labels: Hide class name.
            hide_conf: Hide confidence (score).
            hide_task: Hide task name (task).

        Returns:
            annotated_image: Image with drawn results.
        """
        im_result = image.copy()

        for det in detections:
            box = det['box']  # [x1, y1, x2, y2]
            score = det['score']  # float
            label_idx = det['label']  # int
            label_name = det['label_name']  # str
            task_name = det['task']  # str

            color = self._get_color(label_idx)

            # 1. Draw rectangle (Bounding Box)
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(im_result, p1, p2, color, thickness=self.line_thickness, lineType=cv2.LINE_AA)

            if not hide_labels:
                # 2. Form label text
                parts = []
                parts.append(label_name)

                if not hide_conf:
                    parts.append(f"{score:.2f}")

                if not hide_task:
                    parts.append(f"({task_name})")

                label_text = " ".join(parts)

                # 3. Draw text background and text itself
                self._draw_label(im_result, label_text, p1, color)

        return im_result

    def _draw_label(self, image, text, pos, bg_color):
        """Draws text with a background above the top-left corner of the box."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(self.line_thickness - 1, 1)

        # 1. Calculate text size
        # (w, h) - text width and height, baseline - bottom text offset (tails of g, j, y)
        (text_w, text_h), baseline = cv2.getTextSize(text, font, self.text_scale, thickness)

        x, y = pos  # Top-left corner of bounding box
        padding = 3  # Text padding from background edges

        # 2. Determine coordinates
        # Check if text fits ABOVE the box (y - text height > 0)
        outside = y - text_h - baseline - padding * 2 >= 0

        if outside:
            # Option 1: Background ABOVE the box
            # p1_rect (top-left corner of background)
            p1_rect = (x, y - text_h - baseline - padding * 2)
            # p2_rect (bottom-right corner of background — touches top of box)
            p2_rect = (x + text_w, y)

            # Text position (baseline): move up from box line
            text_origin = (x, y - padding - baseline)
        else:
            # Option 2: Background INSIDE the box (if object is at the very top of image)
            # p1_rect (top-left corner of background — touches top of box)
            p1_rect = (x, y)
            # p2_rect (bottom-right corner of background)
            p2_rect = (x + text_w, y + text_h + baseline + padding * 2)

            # Text position: move down from box line
            text_origin = (x, y + text_h + padding)

        # 3. Draw background (Filled rectangle)
        cv2.rectangle(image, p1_rect, p2_rect, bg_color, -1, cv2.LINE_AA)

        # 4. Draw text (in white color)
        cv2.putText(image, text, text_origin, font, self.text_scale, (255, 255, 255), thickness, cv2.LINE_AA)
