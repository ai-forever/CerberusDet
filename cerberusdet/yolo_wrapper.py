import os
from typing import Union, Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image
import torch.nn as nn
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import (
    DetectionModel,
    BaseModel,
    yaml_model_load,
    LOGGER,
    parse_model,
    deepcopy,
    Detect,
    Segment,
    Pose,
    OBB,
    initialize_weights,
)
from ultralytics.engine.results import Results

os.unsetenv("CUBLAS_WORKSPACE_CONFIG")


class YoloV8Config:
    model_type = 'yolov8'

    def __init__(
        self,
        model_config: str = "yolov8x.yaml",
        task: str = 'detect',
        num_classes: int = 2,
        num_channels: int = 3,
        input_size: int = 640,
        names: Dict = {"0": "person", "1": "face"},
        stride: List[int] = [8, 16, 32],
        verbose: bool = False,
        use_return_dict: bool = True,
        **kwargs: Any
    ):
        self.input_size = input_size
        self.num_channels = num_channels
        self.task = task
        self.model_config = model_config
        self.num_classes = num_classes
        self.stride = stride
        self.verbose = bool(verbose)
        self.names = {int(key): value for key, value in names.items()}
        self.use_return_dict = use_return_dict

        super().__init__(**kwargs)


class YOLOV8DetectionModel(BaseModel):
    _predict_augment = DetectionModel._predict_augment
    _descale_pred = DetectionModel._descale_pred
    _clip_augmented = DetectionModel._clip_augmented
    init_criterion = DetectionModel.init_criterion

    # model, input channels, number of classes
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True, stride: List[int]=[8, 16, 32]):
        """
        Initializes the YOLOv8 detection model with the given configuration and parameters.

        This constructor parses the model configuration (YAML), sets up the input channels and number of classes,
        builds the model architecture, and initializes the strides and weights.

        Args:
            cfg (str | dict): Path to the YAML configuration file or the configuration dictionary itself. Defaults to "yolov8n.yaml".
            ch (int): Number of input channels. Defaults to 3.
            nc (int, optional): Number of classes. If provided, overrides the value in the YAML config. Defaults to None.
            verbose (bool): Whether to print model details during initialization. Defaults to True.
            stride (List[int]): A list of stride values for the detection layer. Defaults to [8, 16, 32].
        """
        super().__init__()

        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose, OBB)):
            m.inplace = self.inplace
            m.stride = torch.tensor(stride, dtype=torch.float32)  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")


class YOLOWrapper(YOLO):

    def __init__(self, model: torch.nn.Module, task=None) -> None:
        """
        Initializes the YOLO wrapper around a specific PyTorch model.

        This allows a standard PyTorch module to be used within the Ultralytics YOLO ecosystem
        by overriding the default initialization to accept an existing model object.

        Args:
            model (torch.nn.Module): The PyTorch model instance to wrap.
            task (str, optional): The specific task type for the YOLO model (e.g., 'detect'). Defaults to None.
        """
        super().__init__(model="", task=task)
        self.model = model


class YOLOV8ForObjectDetection(nn.Module):

    def __init__(self, config: YoloV8Config):
        """
        Initializes the YOLOv8 object detection model based on the provided configuration.

        Args:
            config (YoloV8Config): The configuration object containing model parameters, channels, classes, and strides.
        """
        super().__init__()
        self.config = config

        # initialize a model
        self.model = YOLOV8DetectionModel(
            cfg=self.config.model_config,
            ch=self.config.num_channels,
            nc=self.config.num_classes,
            verbose=self.config.verbose,
            stride=self.config.stride,
        )
        self.model.names = self.config.names
        self.yolo: YOLOWrapper = None
        self.half = False

    def from_pretrained(self, pretrained_model_path: str, **kwargs):  # type: ignore
        """Loads a pretrained YOLOv8 model from a local path or the Hugging Face Hub and initializes the wrapper.

        This class method loads the model weights, creates the `YOLOWrapper` instance, and configures
        task-specific overrides to enable inference immediately after loading.

        Args:
            pretrained_model_path (str): The path of the pretrained model.
            kwargs: Additional keyword arguments

        Returns:
            YOLOV8ForObjectDetection: The initialized model with loaded weights and active YOLO wrapper.
        """
        dtype = torch.float32
        if "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
        elif "torch_dtype" in kwargs:
            dtype = kwargs.pop("torch_dtype")

        if "device" not in kwargs or not kwargs["device"]:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = kwargs.pop("device")

        fuse = True
        inplace = True

        # set model weights
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device=device, dtype=torch.float32)

        # fuse model
        for module in self.model.modules():
            module.requires_grad_(False)

        self.model = (
            self.model.fuse().eval()
            if fuse and hasattr(self.model, "fuse")
            else self.model.eval()
        )

        # module updates
        for m in self.model.modules():
            t = type(m)
            if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment, Pose, OBB):
                m.inplace = inplace
            elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        self.model.fp16 = True if dtype is torch.float16 else False
        self.half = True if dtype is torch.float16 else False

        # initialize a wrapper
        yolo = YOLOWrapper(model=self.model, task=self.config.task)
        yolo.overrides["model"] = pretrained_model_path
        yolo.overrides["task"] = self.config.task
        yolo.overrides["half"] = True if dtype is torch.float16 else False
        self.yolo = yolo
        self.yolo.ckpt = pretrained_model_path
        if dtype is torch.float16:
            self.yolo.model = self.yolo.model.half()

        return self

    def predict(
        self,
        model_input: Optional[Union[Image.Image, np.ndarray, str]] = None,
        return_dict: Optional[bool] = None,
        conf: float = 0.4,
        iou: float = 0.7,
        max_det: int = 300,
        verbose: bool = False,
        **inference_kwargs,
    ) -> Union[Tuple, Results]:
        """
        Performs a forward pass (inference) on the input data using the wrapped YOLO model.

        This method handles image preprocessing, inference, and post-processing (NMS) based on the provided arguments.
        It requires `from_pretrained` to have been called effectively to populate the internal YOLO wrapper.

        Args:
            model_input (Image.Image | np.ndarray | str, optional): The input image(s). Accepts file paths, PIL Images, or NumPy arrays.
            return_dict (bool, optional): Whether to return a dictionary (or Results object) instead of a tuple. Defaults to model config.
            conf (float): Confidence threshold for Non-Maximum Suppression (NMS). Defaults to 0.4.
            iou (float): IoU threshold for NMS. Defaults to 0.7.
            max_det (int): Maximum number of detections allowed per image. Defaults to 100.
            verbose (bool): Whether to print verbose output during inference. Defaults to False.
            **inference_kwargs: Additional arguments supported by the Ultralytics predictor (e.g., `imgsz`, `device`).
                                See all available arguments at https://docs.ultralytics.com/usage/cfg.
        Returns:
            Union[Tuple, Results]: A tuple containing the `Results` object if `return_dict` is False, otherwise the `Results` object directly.

        Raises:
            RuntimeError: If the internal YOLO wrapper is not initialized (e.g., model not loaded via `.from_pretrained()`).
        """

        if self.yolo is None:
            raise RuntimeError("Call .from_pretrained(...) before forward().")

        # accepted image url, PIL.Image or np.ndarray image
        assert isinstance(model_input, (Image.Image, np.ndarray, str))
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        detector_kwargs = {"conf": conf, "iou": iou, "verbose": verbose, "max_det": max_det}
        detector_kwargs.update(inference_kwargs)
        results: Results = self.yolo.predict(model_input, **detector_kwargs)

        if not return_dict:
            return (results,)

        return results
