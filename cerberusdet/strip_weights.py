import argparse
import os
from pathlib import Path

from cerberusdet.utils.general import strip_optimizer


def strip_weights(opt):
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"
    last = wdir / "last.pt"
    best = wdir / "best.pt"

    # Strip optimizers
    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="", help="")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main():
    opt = parse_opt(True)
    assert os.path.exists(opt.save_dir)
    strip_weights(opt)


if __name__ == "__main__":
    """
    Usage: python3 cerberusdet/strip_weights.py --save-dir runs/train/exp
    """
    main()
