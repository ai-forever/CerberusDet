# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F
from cerberusdet.utils.general import xywh2xyxy
from cerberusdet.utils.metrics import bbox_iou
from cerberusdet.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors
from cerberusdet.utils.torch_utils import get_hyperparameter


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


# Criterion class for computing training losses
class Loss:
    def __init__(self, model, task_ids):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        self.device = device

        self.stride, self.reg_max = None, None
        self.bce = {}
        self.nc, self.no = {}, {}
        self.assigner = {}
        self.loss_weights = {}

        for task_idx, task in enumerate(task_ids):
            self.bce[task] = nn.BCEWithLogitsLoss(reduction="none")

            box_w = get_hyperparameter(h, "box", task_idx, task)
            dfl_w = get_hyperparameter(h, "dfl", task_idx, task)
            cls_w = get_hyperparameter(h, "cls", task_idx, task)
            self.loss_weights[task] = dict(box=box_w, dfl=dfl_w, cls=cls_w)

        if hasattr(model, "heads"):  # CerberNet
            for task_name, _ in model.heads.items():
                head = model.get_head(task_name)
                self.nc[task_name] = head.nc
                self.no[task_name] = head.no

                if self.reg_max is None:
                    self.reg_max = head.reg_max
                else:
                    assert self.reg_max == head.reg_max

                if self.stride is None:
                    self.stride = head.stride
                else:
                    assert torch.equal(self.stride, head.stride)

                self.assigner[task_name] = TaskAlignedAssigner(
                    topk=10,
                    num_classes=head.nc,
                    alpha=0.5,
                    beta=6.0
                )

        else:
            m = model.model[-1]  # Detect() module
            task = task_ids[0]
            self.stride = m.stride  # model strides
            self.nc[task] = m.nc  # number of classes
            self.no[task] = m.no
            self.reg_max = m.reg_max
            self.assigner[task] = TaskAlignedAssigner(
                topk=10,
                num_classes=m.nc,
                alpha=0.5,
                beta=6.0,
            )

        self.use_dfl = self.reg_max > 1
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 2:6] = xywh2xyxy(out[..., 2:6].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))

        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, task):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, total
        feats = preds[1] if isinstance(preds, tuple) else preds
        dtype = feats[0].dtype
        batch_size = feats[0].shape[0]

        pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.no[task], -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc[task]), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["prob"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_probs, gt_bboxes = targets.split((1, 1, 4), 2)  # cls, prob, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner[task](
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce[task](pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.loss_weights[task]["box"]  # box gain
        loss[1] *= self.loss_weights[task]["cls"]  # cls gain
        loss[2] *= self.loss_weights[task]["dfl"]  # dfl gain
        loss[3] = loss.sum()

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, total)
