# Plotting utils

import contextlib
import json
import math
from copy import copy
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import yaml
from cerberusdet.utils.checks import check_font, check_requirements, is_ascii
from cerberusdet.utils.general import get_user_config_dir, xywh2xyxy, xyxy2xywh
from cerberusdet.utils.metrics import overall_fitness
from cerberusdet.utils.torch_utils import threaded
from PIL import Image, ImageDraw, ImageFont

# Settings
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only
USER_CONFIG_DIR = get_user_config_dir()  # settings dir


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def create_label(names, classes, probas, tasks, hide_labels, hide_conf, hide_task):
    if hide_labels:
        label = None
    else:
        if hide_conf and hide_task:
            label = " ".join([names[x] for x in classes])
        elif hide_task:
            label = " ".join([f"{names[xx]} {yy:.2f}" for xx, yy in zip(classes, probas)])
        elif hide_conf:
            label = " ".join([f"{names[xx]} {task}" for xx, task in zip(classes, tasks)])
        else:
            label = " ".join([f"{names[xx]} {yy:.2f} {task}" for xx, yy, task in zip(classes, probas, tasks)])

    return label


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image."
    tl = line_thickness or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        if c1[1] - t_size[1] - 3 > 0:
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            text_y = c1[1] - 2
        else:
            c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
            text_y = c1[1] + t_size[1]
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], text_y), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def output_to_target(output, max_det=300):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:]


class Annotator:
    # YOLOv8 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images."
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(
                font="Arial.Unicode.ttf" if non_ascii else font,
                size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12),
            )
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height (WARNING: deprecated) in 9.2.0
                # _, _, w, h = self.font.getbbox(label)  # text width, height (New)
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (
                        box[0],
                        box[1] - h if outside else box[1],
                        box[0] + w + 1,
                        box[1] + 1 if outside else box[1] + h + 1,
                    ),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top"):
        # Add text to image (PIL-only)
        if anchor == "bottom":  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def check_pil_font(font, size):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (USER_CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:  # download if missing
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements("Pillow>=8.4.0")  # known issue https://github.com/ultralytics/yolov5/issues/5374
        except URLError:  # not online
            return ImageFont.load_default()


@threaded
def plot_images(
    images,
    batch_idx,
    cls,
    bboxes,
    masks=np.zeros(0, dtype=np.uint8),
    paths=None,
    fname="images.jpg",
    names=None,
    mlflow_logger=None,
):

    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y : y + h, x : x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i

            boxes = xywh2xyxy(bboxes[idx, :4]).T
            classes = cls[idx].astype("int")
            labels = bboxes.shape[1] == 4  # labels if no conf column
            conf = None if labels else bboxes[idx, 4]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                c = classes[j]
                color = colors(c)
                c = names[c] if names else c
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                    annotator.box_label(box, label, color=color)

            # Plot masks
            if len(masks):
                if masks.max() > 1.0:  # mean that masks are overlap
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape(nl, 1, 1) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)
                else:
                    image_masks = masks[idx]

                im = np.asarray(annotator.im).copy()
                for j, box in enumerate(boxes.T.tolist()):
                    if labels or conf[j] > 0.25:  # 0.25 conf thresh
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                annotator.fromarray(im)

    annotator.im.save(fname)  # save

    if mlflow_logger:
        mlflow_logger.save_artifact(str(fname))


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=""):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])
    plt.plot(y, ".-", label="LR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / "LR.png", dpi=200)
    plt.close()


def plot_labels(labels, names=(), save_dir=Path(""), loggers=None, name="labels"):
    # plot dataset labels
    print("Plotting labels... ")
    if labels.shape[1] == 6:
        _, b = labels[:, 0], labels[:, 2:].transpose()  # classes, boxes
    elif labels.shape[1] == 7:
        _, b = labels[:, 1], labels[:, 3:].transpose()  # classes, boxes

    # nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use("svg")  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    # y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    # [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # update colors bug #3195
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    sn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)

    if labels.shape[1] == 6:
        labels[:, 2:4] = 0.5
        labels[:, 2:] = xywh2xyxy(labels[:, 2:]) * 2000
        for cls, prob, *box in labels[:1000]:
            ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    elif labels.shape[1] == 7:
        labels[:, 3:5] = 0.5
        labels[:, 3:] = xywh2xyxy(labels[:, 3:]) * 2000
        for task_id, cls, prob, *box in labels[:1000]:
            ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot

    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / f"{name}.jpg", dpi=200)
    matplotlib.use("Agg")
    plt.close()


def plot_evolution(
    yaml_file="data/hyp.finetune.yaml", evolve_results_file="evolve.json", evolved_params: Optional[List[str]] = None
):
    """
    yaml_file: file with saved best hyperparameters
    evolve_results_file: file with all hypersearch generations
    evolved_params: list of names of evolved params. If None => assumed that all params were evolved
    """
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as file:
        hyp = yaml.safe_load(file)

    with open(evolve_results_file) as file:
        mutations_list = json.load(file)

    fit = np.array([overall_fitness(x["results_per_task"]) for x in mutations_list])
    j = np.argmax(fit)  # max fitness index
    print(f"Best results from {j}-th element of {evolve_results_file}:")

    task_names = mutations_list[0]["results_per_task"].keys()
    # weights = (f - f.min()) ** 2  # for weighted results
    if evolved_params is None:
        evolved_params = hyp.keys()

    for task_ind, task_name in enumerate(task_names):
        plt.figure(figsize=(10, 12), tight_layout=True)
        matplotlib.rc("font", **{"size": 8})
        for i, k in enumerate(evolved_params):
            # all hyperparam values for the task
            y = [x["hyps"][k] for x in mutations_list]
            y = np.array([v[task_ind] if isinstance(v, list) else v for v in y])
            # best single result
            best_mu = mutations_list[j]["hyps"][k]
            mu = best_mu[task_ind] if isinstance(best_mu, list) else best_mu

            plt.subplot(6, 5, i + 1)
            plt.scatter(y, fit, c=hist2d(y, fit, 20), cmap="viridis", alpha=0.8, edgecolors="none")
            plt.plot(mu, fit.max(), "k+", markersize=15)
            plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
            if i % 5 != 0:
                plt.yticks([])
            print(f"{k:>15}: {mu:.3g}")

        out_img = evolve_results_file.replace(".json", f"_{task_name}.png")
        plt.savefig(out_img, dpi=200)
        plt.close()
        print(f"\nPlot for task {task_name} saved to {out_img}")


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if "Detect" not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            print(f"Saving {save_dir / f}... ({n}/{channels})")
            plt.savefig(save_dir / f, dpi=300, bbox_inches="tight")
