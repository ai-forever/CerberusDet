import logging
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from cerberusdet.models.common import (
    C2,
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    Concat,
    Contract,
    Conv,
    DWConv,
    Expand,
    Focus,
)
from cerberusdet.models.experimental import CrossConv, GhostBottleneck, GhostConv, MixConv2d
from cerberusdet.utils.general import make_divisible
from cerberusdet.utils.plots import feature_visualization
from cerberusdet.utils.tal import dist2bbox, make_anchors
from cerberusdet.utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, time_sync
from loguru import logger

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", -1))

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add cerberusdet/ to path


class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


# heads
class Detect(nn.Module):
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Model(nn.Module):
    def __init__(self, cfg="yolov8x.yaml", ch=3, nc=None, without_head=False, _=None, verbose=True):
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels

        if isinstance(nc, list) and self.yaml.get("nc", None) is not None:
            self.yaml["nc"] = nc  # override yaml value
        elif nc and (self.yaml.get("nc") is None or nc != self.yaml["nc"]):
            if verbose and LOCAL_RANK in [0, -1]:
                LOGGER.info(f"Overriding model.yaml nc={self.yaml.get('nc')} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value

        self.model, self.save, self.saved_ch = parse_model(
            deepcopy(self.yaml), ch=[ch], without_head=without_head, verbose=verbose
        )  # model, savelist

        self.without_head = without_head
        self.inplace = self.yaml.get("inplace", True)

        if not without_head:
            if isinstance(self.yaml["nc"], list):
                assert len(self.yaml["nc"]) == 1
                self.yaml["nc"] = self.yaml["nc"][0]

            self.names = [str(i) for i in range(self.yaml["nc"])]  # default names

            # Build strides, anchors
            m = self.model[-1]  # Detect()
            if isinstance(m, Detect):
                s = 256  # 2x min stride
                m.inplace = self.inplace
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
                self.stride = m.stride
                m.bias_init()  # only run once

                # Init weights, biases
                initialize_weights(self)

        if verbose and LOCAL_RANK in [0, -1]:
            self.info()
            LOGGER.info("")

    def forward(self, x, profile=False, visualize=False):
        return self.forward_once(x, profile, visualize)  # single-scale inference, train

    def forward_once(self, x, profile=False, visualize=False):

        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
                t = time_sync()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0] and LOCAL_RANK in [0, -1]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                if LOCAL_RANK in [0, -1]:
                    LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if profile and LOCAL_RANK in [0, -1]:
            LOGGER.info("%.1fms total" % sum(dt))

        if hasattr(self, "without_head") and self.without_head:
            # pass all saved outputs to cerbernet blocks
            return y

        return x

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        if not isinstance(m, Detect):
            return
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ("%6g Conv2d.bias:" + "%10.3g" * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        if LOCAL_RANK in [0, -1]:
            LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        prefix = "Model Summary:"
        if hasattr(self, "without_head") and self.without_head:
            # CerberusDet case
            prefix = "Backbone summary:"
        model_info(self, verbose, img_size, prefix=prefix)


def parse_model(yaml_config, ch, without_head=False, verbose=True):  # model_dict, input_channels(3)
    if verbose and LOCAL_RANK in [0, -1]:
        LOGGER.info("\n%3s%18s%3s%10s  %-40s%-30s" % ("", "from", "n", "params", "module", "arguments"))
    gd, gw = yaml_config["depth_multiple"], yaml_config["width_multiple"]
    max_channels = yaml_config.get("max_channels", 1024)

    nc = yaml_config["nc"]

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    if without_head:
        module_args = yaml_config["backbone"]
    else:
        module_args = yaml_config["backbone"]
        if yaml_config.get("neck"):
            module_args += yaml_config["neck"]
        module_args += yaml_config["head"]

    for i, (f, n, m, args) in enumerate(module_args):  # from, number, module, args

        m = eval(m) if isinstance(m, str) else m  # eval strings
        args, _, n, c2, m_ = get_next_layer_from_cfg(gd, ch, gw, nc, m, n, f, args, max_channels)
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params

        if verbose and LOCAL_RANK in [0, -1]:
            LOGGER.info("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))  # print

        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist

        if verbose and LOCAL_RANK in [0, -1]:
            logger.info("model build info: ", c2, ch)

    if without_head:
        i = len(layers)
        for h_layer in yaml_config["neck"] + yaml_config["head"]:  # from, number, module, args
            f = h_layer[0]
            f = [f] if isinstance(f, int) else f
            save.extend(x % i for x in f if x != -1 and x < len(layers))  # append to savelist
            i += 1

    return nn.Sequential(*layers), sorted(save), ch


def get_next_layer_from_cfg(gd, ch, gw, nc, m, n, f, args, max_channels):
    for j, a in enumerate(args):
        try:
            args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        except (ValueError, SyntaxError, NameError, TypeError):
            pass

    c2 = None
    n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
    if m in [
        Conv,
        GhostConv,
        Bottleneck,
        GhostBottleneck,
        SPP,
        SPPF,
        DWConv,
        MixConv2d,
        Focus,
        CrossConv,
        BottleneckCSP,
        C3,
        C3TR,
        C3SPP,
        C2f,
        C2,
    ]:
        c1, c2 = ch[f], args[0]
        if all([c2 != nc_ for nc_ in nc]):  # if not output
            c2 = make_divisible(min(c2, max_channels) * gw, 8)

        args = [c1, c2, *args[1:]]
        if m in [BottleneckCSP, C3, C3TR, C2f, C2]:
            args.insert(2, n)  # number of repeats
            n = 1
    elif m is nn.BatchNorm2d:
        args = [ch[f]]
    elif m is Concat:
        c2 = sum([ch[x] for x in f])
    elif m in [Detect]:
        if len(args) == 0:
            nc_ = nc.pop(0)
            args.append(nc_)
        elif isinstance(args[0], list):
            args[0] = args[0][0]

        args.append([ch[x] for x in f])
        c2 = None
    elif m is Contract:
        c2 = ch[f] * args[0] ** 2
    elif m is Expand:
        c2 = ch[f] // args[0] ** 2
    else:
        c2 = ch[f]

    module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
    return args, nc, n_, c2, module
