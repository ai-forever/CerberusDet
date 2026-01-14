import torch
import numpy as np
from typing import Union, List

from cerberusdet.data.augmentations import letterbox
from cerberusdet.utils.general import (
    check_img_size,
)
from cerberusdet.utils.torch_utils import select_device


class CerberusPreprocessor:
    def __init__(
        self,
        img_size: int = 640,
        stride: int = 32,
        device: Union[str, torch.device] = "cpu",
        half: bool = False,
        auto: bool = False,
    ):
        """
        Class for preparing images for inference.

        Args:
            img_size: Desired image size.
            stride: Model stride (obtained from CerberusDetInference).
            device: Device.
            half: Whether to use FP16.
            auto: If True, creates a minimal rectangle (as in validation).
                  If False (recommended for streams), forces a square size (e.g. 640x640) with gray padding.
        """
        self.device = select_device(device)
        self.stride = stride
        self.half = half & (self.device.type != "cpu")
        self.auto = auto

        # Check if the size is a multiple of the stride. If not, update it.
        self.img_size = check_img_size(img_size, s=self.stride)
        if self.img_size != img_size:
            print(
                f"Warning: --img-size {img_size} must be multiple of max stride {self.stride}, updating to {self.img_size}"
            )

    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Args:
            images: List of source images in BGR format (H, W, 3).

        Returns:
            img_tensor: Tensor [B, 3, H_new, W_new], normalized 0-1.
        """
        processed_images = []

        for img0 in images:
            # 1. Padded resize (letterbox)
            # Use function from cerberusdet.data.augmentations
            img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

            # 2. Convert HWC -> CHW, BGR -> RGB
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)

            processed_images.append(img)

        # 3. Stack into a single numpy array (Batch dimension)
        # Result shape: [Batch_Size, 3, H, W]
        batched_img = np.stack(processed_images, axis=0)

        # 4. To Tensor
        img_tensor = torch.from_numpy(batched_img).to(self.device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()

        # 5. Normalize 0 - 255 to 0.0 - 1.0
        img_tensor /= 255.0

        return img_tensor
