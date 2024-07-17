import hashlib
import json
import os
import random
import xml.etree.ElementTree as ET
from functools import partial
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from cerberusdet.data.augmentations import Albumentations, augment_hsv, letterbox, mixup, random_perspective
from cerberusdet.utils.general import xywhn2xyxy, xyxy2xywhn
from loguru import logger
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

# import stackprinter
# logger.add(sys.stdout, format="{extra} {message}")
# stackprinter.set_excepthook(style='darkbg2')


# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]  # acceptable image suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    def get_orientation_tag():
        # Get orientation exif tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                return orientation
        return None

    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        orientation = get_orientation_tag()
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def img2label_paths(img_paths, use_xml: bool):
    # Define label paths as a function of image paths
    label_paths: List[str] = list()
    sa, sb = os.sep + "images" + os.sep, os.sep + "labels" + os.sep  # /images/, /labels/ substrings

    for image_path in img_paths:
        label_path = Path(image_path).with_suffix(".xml" if use_xml else ".txt")

        if use_xml:
            assert label_path.exists()

        label_paths.append(sb.join(str(label_path).rsplit(sa, 1)))

    return label_paths


def get_task_hyperparams(hyp: Dict[str, Any], task_ind: int, task_name: Optional[str] = None) -> Dict[str, float]:
    """
    Get dict with hyperparams for particular task from passed hyp
        Input hyp can be with items like
            {"perspective": 0.0, "flipud": [0.02143, 0.00983], "shear_clothes": 0.0, "shear_shoes": 0.2}
    """
    if hyp is None:
        return None

    new_hyp = {}
    for k, v in hyp.items():
        if isinstance(v, list):
            assert task_ind is not None
            assert task_ind < len(v)
            new_hyp[k] = v[task_ind]
        elif task_name is not None and f"{task_name}_" in k or f"_{task_name}" in k:
            new_k = k.replace(f"{task_name}_", "").replace(f"_{task_name}", "")
            new_hyp[new_k] = v
        else:
            new_hyp[k] = v

    return new_hyp


def load_images_files(path, prefix, skip_prefix):
    f = []  # image files
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            # f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            f += [os.path.join(p, x) for x in os.listdir(p)]
            # f = list(p.rglob('**/*.*'))  # pathlib
        elif p.is_file() and p.suffix == ".json":
            logger.info("Read images from .json file..")
            with open(p.as_posix(), "r") as json_file:
                data = json.load(json_file)
            images = data["images"]
            root_dir = p.absolute().parents[0].as_posix()
            for im_info in images:
                file_name = im_info["file_name"]
                full_path = os.path.join(root_dir, file_name)
                f.append(full_path)
        elif p.is_file():  # file
            with open(p, "r") as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f"{prefix}{p} does not exist")

    if skip_prefix is not None:
        img_files = sorted(
            [
                x.replace("/", os.sep)
                for x in f
                if x.split(".")[-1].lower() in IMG_FORMATS and skip_prefix not in x.split(".")[0]
            ]
        )
    else:
        img_files = sorted([x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS])
    # img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
    return img_files, p


class LoadImagesAndLabels(Dataset):  # for training/testing
    cache_version = 0.4

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        skip_prefix=None,
        task_ind=None,
        task_names=None,
        labels_from_xml=False,
        classnames=None,
        as_multi_label=False,
        as_soft_label=False,
    ):

        self.img_size = img_size
        self.augment = augment
        self.task_ind = task_ind
        self.hyp = get_task_hyperparams(hyp, self.task_ind, task_names[task_ind] if task_names is not None else None)
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        self.use_cache, self.update_cache = True, False
        self.task_names = task_names
        self.labels_from_xml = labels_from_xml
        self.classnames = classnames

        if self.labels_from_xml:
            # Class names are needed to convert names to indexes
            assert classnames is not None

        self.as_multi_label = as_multi_label
        self.as_soft_label = as_soft_label

        try:
            self.img_files, p = load_images_files(path, prefix, skip_prefix)
            assert self.img_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}")

        self.label_files = img2label_paths(self.img_files, use_xml=self.labels_from_xml)  # labels
        # Check cache
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            assert self.use_cache and not self.update_cache
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version and cache["hash"] == get_hash(
                self.label_files + self.img_files
            )
        except Exception:
            # load and cache labels
            cache, exists = self.cache_labels(cache_path, prefix), False  # create cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            # if cache["msgs"]:
            #     logger.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels in {cache_path}. Can not train without labels."

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)  # image shapes
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys(), use_xml=self.labels_from_xml)  # update

        assert self.labels[0].shape[1] == 6

        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f"{prefix}Caching images ({gb / 1E9:.1f}GB)"
            pbar.close()

    def update_hyp(self, hyp: Dict[str, Any]) -> None:
        assert self.task_names is not None
        self.hyp = get_task_hyperparams(
            hyp, self.task_ind, self.task_names[self.task_ind] if self.task_names is not None else None
        )

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        worker = partial(
            verify_image_label,
            prefix=prefix,
            use_xml=self.labels_from_xml,
            classnames=self.classnames,
            as_multi_label=self.as_multi_label,
            as_soft_label=self.as_soft_label,
        )

        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap_unordered(worker, zip(self.img_files, self.label_files)), desc=desc, total=len(self.img_files)
            )
            for im_file, l, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        # if msgs:
        #     logger.info("\n".join(msgs))
        if nf == 0:
            logger.info(f"{prefix}WARNING: No labels found in {path}")
        x["hash"] = get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, len(self.img_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            logger.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            logger.info(f"{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}")  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()

            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 2:] = xywhn2xyxy(labels[:, 2:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    scaleup=hyp["scaleup"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels

        if nl:
            assert labels.shape[1] == 6
            labels[:, 2:6] = xyxy2xywhn(labels[:, 2:6], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 3] = 1 - labels[:, 3]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 7))

        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        # YOLOv8 collate function, outputs dict
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()

        cat_label = torch.cat(label, 0)
        batch_idx, cls, prob, bboxes = cat_label.split((1, 1, 1, 4), dim=1)

        return {
            "ori_shape": tuple((x[0] if x else None) for x in shapes),
            "ratio_pad": tuple((x[1] if x else None) for x in shapes),
            "im_file": path,
            "img": torch.stack(im, 0),
            "cls": cls,
            "prob": prob,
            "bboxes": bboxes,
            "batch_idx": batch_idx.view(-1),
        }


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(
                img,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR,
            )
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_mosaic(self, index, indexes_to_select=None):
    # loads images in a 4-mosaic

    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    if indexes_to_select is None:
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    else:
        indices = [index] + random.choices(indexes_to_select, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 2:] = xywhn2xyxy(labels[:, 2:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        labels4.append(labels)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    bbox_ind = 2

    for x in (labels4[:, bbox_ind:],):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    img4, labels4 = random_perspective(
        img4,
        labels4,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        scaleup=self.hyp["scaleup"],
        shear=self.hyp["shear"],
        perspective=self.hyp["perspective"],
        border=self.mosaic_border,
    )  # border to remove

    return img4, labels4


def xml_jsonify(tree: ET.ElementTree):
    extracted: Dict[str, Any] = dict()

    root = tree.getroot()

    extracted["folder"] = root.find("folder").text
    extracted["filename"] = root.find("filename").text
    extracted["path"] = root.find("path").text
    extracted["split"] = root.find("split").text if root.find("split") else None  # old format

    extracted["width"] = int(root.find("size").find("width").text)
    extracted["height"] = int(root.find("size").find("height").text)

    # support old format
    info = root.find("info")
    if info:
        extracted["source"] = info.find("source").text if info.find("source") else None
        extracted["annotation"] = info.find("annotation").text if info.find("annotation") else None
        extracted["database"] = info.find("database").text if info.find("database") else None
        extracted["validated"] = info.find("validated").text if info.find("validated") else None

    extracted["bounding_boxes"] = list()

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        minors = obj.find("minors")

        extracted["bounding_boxes"].append(
            {
                "class": obj.find("name").text,
                "x_min": int(float(bbox.find("xmin").text)),
                "y_min": int(float(bbox.find("ymin").text)),
                "x_max": int(float(bbox.find("xmax").text)),
                "y_max": int(float(bbox.find("ymax").text)),
                "minors": {x.find("name").text: int(x.find("votes").text) for x in minors} if minors else None,
            }
        )

    return extracted


def convert_to_lb(annotation, classnames, as_multi_label: bool, as_soft_label: bool):
    lb = list()

    for bbox in annotation["bounding_boxes"]:
        cx = (bbox["x_max"] + bbox["x_min"]) / 2 / annotation["width"]
        cy = (bbox["y_max"] + bbox["y_min"]) / 2 / annotation["height"]

        w = (bbox["x_max"] - bbox["x_min"]) / annotation["width"]
        h = (bbox["y_max"] - bbox["y_min"]) / annotation["height"]

        classes_map = bbox["minors"].copy() if bbox["minors"] else dict()

        # Add main class if it wasn't in minors
        # Some boxes have the number of votes for the main class and some don't (need to be fixed).
        # Here we bring it all together
        if bbox["class"] not in classes_map.keys():
            classes_map[bbox["class"]] = sum(classes_map.values()) + 1

        if as_soft_label:
            # Replace votes count by probas
            votes_count = sum(classes_map.values())
            classes_map = {k: v / votes_count for k, v in classes_map.items()}
        else:
            classes_map = {k: 1 for k, v in classes_map.items()}

        # Remove minors if multilabel is not needed
        if not as_multi_label:
            classes_map = {k: v for k, v in classes_map.items() if k == bbox["class"]}

        for cls, prob in classes_map.items():
            lb.append([classnames.index(cls), prob, cx, cy, w, h])

    return np.array(lb, dtype=np.float32)


def verify_image_label(args, prefix, use_xml, classnames, as_multi_label, as_soft_label):
    # Verify one image-label pair
    im_file, lb_file = args
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt

    label_size = 6  # cat_id proba x y w h

    try:
        msg = ""
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} < 10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING: {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found

            if use_xml:
                tree = ET.parse(lb_file)
                annotation = xml_jsonify(tree)
                l = convert_to_lb(annotation, classnames, as_multi_label, as_soft_label)

            else:
                with open(lb_file, "r") as f:
                    l = [x.split() for x in f.read().strip().splitlines() if len(x)]

                    # Add probabilty in hardlabel style if it is not in the annotation
                    if any([len(x) == 5 for x in l]):  # cat_id x y w h -> add proba
                        l = [[x[0], "1.0", *x[1:]] for x in l]
                    elif any([len(x) > 5 for x in l]):
                        raise ValueError("Invalid annotation file")

                l = np.array(l, dtype=np.float32)

            nl = len(l)
            if nl:
                assert l.shape[1] == label_size, f"labels require {label_size} columns each"

                if label_size == 6:  # default specific row
                    assert (l >= 0).all(), "negative labels"
                    assert (l[:, 2:] <= 1).all(), "non-normalized or out of bounds coordinate labels"
                else:  # task specific row
                    assert (l >= 0).all(), "negative labels"
                    assert (l[:, 3:] <= 1).all(), "non-normalized or out of bounds coordinate labels"
                # assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'

                _, i = np.unique(l, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    l = l[i]  # remove duplicates
                    msg = f"{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                l = np.zeros((0, label_size), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, label_size), dtype=np.float32)
        return im_file, l, shape, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}"
        return [None, None, None, nm, nf, ne, nc, msg]
