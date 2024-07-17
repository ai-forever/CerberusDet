import os
import shutil
from pathlib import Path

from loguru import logger


class CheckpointLogger:
    def __init__(self, save_dir: str):
        self.last = Path(save_dir) / "weights" / "last.pt"  # ckpt path from a train iter
        self.best = Path(save_dir) / "weights" / "best.pt"  # ckpt path for the best hyp

    def update_best_model(self) -> None:

        if not self.last.exists():
            logger.error(f"Model {str(self.last)} does not exists")
            return

        shutil.move(str(self.last), str(self.best))
        logger.info(f"Best model updated to {str(self.best)}")

    def remove_last_model(self) -> None:
        if self.last.exists():
            os.remove(str(self.last))
            logger.info(f"Model {str(self.last)} was removed")
