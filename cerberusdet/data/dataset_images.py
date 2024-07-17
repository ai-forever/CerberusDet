import glob
import os
from pathlib import Path

import cv2
import numpy as np
from cerberusdet.data.augmentations import letterbox
from cerberusdet.data.datasets import IMG_FORMATS
from loguru import logger


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = len(images)  # number of files
        assert self.nf > 0, f"No images found in {p}. " f"Supported formats are:\nimages: {IMG_FORMATS}"

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, "Image Not Found " + path
        logger.info(f"image {self.count}/{self.nf} {path}: ", end="")

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return Path(path), img, img0

    def __len__(self):
        return self.nf  # number of files
