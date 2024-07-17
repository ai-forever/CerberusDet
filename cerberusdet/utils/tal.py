# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from cerberusdet.utils.checks import check_version
from cerberusdet.utils.metrics import bbox_iou

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    target_gt_idx = (torch.arange(end=n_max_boxes, device=mask_pos.device)[None, :, None] * mask_pos).long()

    # If the maximum iou is the same for multiple gt_bboxes, then it is a multilabel bbox and we keep all labels for
    # that maximum. If iou is unique, then the label from the box with the largest intersection is taken as usual
    max_mask = overlaps == torch.max(overlaps, dim=1, keepdim=True).values
    target_gt_idx *= max_mask
    unique_bbox_idx = target_gt_idx.argmax(1)

    fg_mask = mask_pos.sum(-2)

    return target_gt_idx, fg_mask, mask_pos, unique_bbox_idx


class TaskAlignedAssigner(nn.Module):
    def __init__(
        self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, label_smoothing=0.0, use_soft_labels=False
    ):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_soft_labels = use_soft_labels
        self.label_smoothing = label_smoothing

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_probs, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_probes (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos, unique_bbox_idx = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        # assigned target
        target_bboxes, target_hard_scores, target_soft_scores = self.get_targets(
            gt_labels, gt_probs, gt_bboxes, target_gt_idx, fg_mask, unique_bbox_idx
        )

        # normalize
        pos_align_metrics = (align_metric * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (
            (align_metric * pos_overlaps * mask_pos / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        )

        target_scores = target_hard_scores * norm_align_metric

        if self.use_soft_labels:
            assert target_soft_scores is not None
            target_scores = target_scores * target_soft_scores

        if self.label_smoothing > 0.0:
            target_scores[target_scores != 0] = target_scores[target_scores != 0] * (1.0 - 0.5 * self.label_smoothing)
            target_scores[target_scores == 0] = 0.5 * self.label_smoothing

        assert (
            0 <= target_scores.min() and target_scores.max() <= 1
        ), f"min={target_scores.min()}, max={target_scores.max()}"

        return target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask, (b, max_num_obj, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool()
        )
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w

        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics: (b, max_num_obj, h*w).
            topk_mask: (b, max_num_obj, topk) or None
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def _get_bbox_targets(self, gt_bboxes, target_gt_idx, unique_bbox_idx):
        bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        bboxes = torch.gather(bboxes, 1, unique_bbox_idx[:, None, :, None].expand(-1, 1, -1, 4)).squeeze(1)

        return bboxes

    def _get_hard_targets(self, target_labels, fg_mask):
        hard_scores = F.one_hot(target_labels, self.num_classes).sum(-3)
        # Get clean hard vectors
        fg_columns = hard_scores.sum(1).argmax(1)
        diff = (self.n_max_boxes - fg_mask).long()
        # Create an index mask for fg_columns and perform subtraction without loops
        # A simple alternative with a loop to understand
        # for idx, column in enumerate(fg_columns):
        #     hard_scores[idx, :, column] -= diff[idx, :]
        mask = torch.zeros_like(hard_scores)
        mask[torch.arange(hard_scores.size(0)), :, fg_columns] = 1

        non_fg_count = (hard_scores * ~mask.to(torch.bool)).sum(-1)
        hard_scores = hard_scores - (diff.unsqueeze(-1) * mask)

        denominator = (fg_mask - non_fg_count).unsqueeze(-1) * mask
        non_zero_mask = denominator != 0
        hard_scores[non_zero_mask] = (hard_scores[non_zero_mask] / denominator[non_zero_mask]).long()

        assert hard_scores.max() <= self.n_max_boxes
        assert not torch.isnan(hard_scores).any().item(), "Hard scores is nan"
        assert not torch.isinf(hard_scores).any().item(), "Hard scores is inf"

        return hard_scores, diff, mask

    def _get_soft_targets(
        self, target_labels, target_probs, fg_mask, gt_probs, hard_scores, diff, mask, multilabel_mask, eps=1e-5
    ):
        indices = target_labels.unsqueeze(-1).expand(-1, -1, -1, self.num_classes)
        soft_scores = torch.zeros(indices.shape, dtype=target_probs.dtype, device=target_probs.device)
        soft_scores.scatter_(-1, indices, target_probs.unsqueeze(-1).expand(-1, -1, -1, self.num_classes))
        soft_scores = soft_scores.sum(-3)

        # Get clean hard vectors
        fg_values = gt_probs.flatten()[:: self.n_max_boxes]
        fg_part = diff * fg_values.unsqueeze(-1)

        non_fg_count = (soft_scores * ~mask.to(torch.bool)).sum(-1)
        soft_scores = soft_scores - (fg_part.unsqueeze(-1) * mask)

        denominator = (fg_mask - non_fg_count).unsqueeze(-1) * mask
        non_zero_mask = denominator != 0
        soft_scores[non_zero_mask] = soft_scores[non_zero_mask] / denominator[non_zero_mask]

        soft_scores[multilabel_mask] = soft_scores[multilabel_mask] / hard_scores[multilabel_mask]

        soft_scores[torch.logical_and(soft_scores >= 0, soft_scores <= eps)] = 0
        soft_scores[torch.logical_and(soft_scores < 0, soft_scores >= -eps)] = 0
        soft_scores[torch.logical_and(soft_scores >= 1, soft_scores <= 1 + eps)] = 1
        soft_scores[torch.logical_and(soft_scores < 1, soft_scores >= 1 - eps)] = 1

        assert not torch.isnan(soft_scores).any().item(), "Soft scores is nan"
        assert not torch.isinf(soft_scores).any().item(), "Soft scores is inf"

        return soft_scores

    def get_targets(self, gt_labels, gt_probs, gt_bboxes, target_gt_idx, fg_mask, unique_bbox_idx):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_probs: (b, max_num_obj, 4)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        """
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        batch_ind = batch_ind.repeat(1, self.n_max_boxes)
        target_gt_idx = target_gt_idx + (batch_ind * self.n_max_boxes)[..., None]

        # ASSIGNED TARGET LABELS AND PROBS, (b, 1)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, max_num_obj, h*w)
        target_labels.clamp(0)

        target_probs = gt_probs.flatten()[target_gt_idx]
        target_probs.clamp(0)

        # ASSIGNED TARGET BBOXES, (b, max_num_obj, 4) -> (b, h*w)
        bboxes = self._get_bbox_targets(gt_bboxes, target_gt_idx, unique_bbox_idx)

        # ASSIGNED TARGET HARD SCORES
        hard_scores, diff, mask = self._get_hard_targets(target_labels, fg_mask)

        # ASSIGNED TARGET SOFT SCORES
        multilabel_mask = (
            hard_scores > 1
        )  # Anchor point can correspond to >=2 close different boxes with the same class

        soft_scores = None

        if self.use_soft_labels:
            soft_scores = self._get_soft_targets(
                target_labels, target_probs, fg_mask, gt_probs, hard_scores, diff, mask, multilabel_mask
            )

        hard_scores[multilabel_mask] = 1

        return bboxes, hard_scores, soft_scores


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)
