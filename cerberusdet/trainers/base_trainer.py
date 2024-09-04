import logging
from collections import defaultdict

import numpy as np
import torch
from cerberusdet import val
from cerberusdet.utils.loss import Loss
from cerberusdet.utils.metrics import fitness
from cerberusdet.utils.models_manager import ModelManager
from cerberusdet.utils.torch_utils import EarlyStopping, de_parallel, get_hyperparameter

LOGGER = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        device: torch.device,
        train_loaders: list,
        val_loaders: list,
        model_manager: ModelManager,
        imgsz: int,
        gs: int,
        dataset: list,
    ):

        self.model_manager = model_manager
        self.hyp = model_manager.hyp
        self.opt = model_manager.opt
        self.val_loaders = val_loaders
        self.imgsz = imgsz
        self.gs = gs
        self.train_loaders = train_loaders
        self.dataset = dataset
        self.task_ids = model_manager.task_ids  # from data.yaml
        self.best_fitness_per_task = dict()
        self.maps_per_task = dict()
        for i, task in enumerate(self.task_ids):
            self.best_fitness_per_task[task] = 0.0
            nc = self.model_manager.data_dict["nc"][i]
            self.maps_per_task[task] = np.zeros(nc)
        self.best_fitness = 0.0
        self.last_fitness = 0.0
        self.device = device

        self.stopper = EarlyStopping(patience=self.opt.patience)

    def set_loss(self, model):
        self.compute_loss = Loss(de_parallel(model), self.task_ids)

    def train_epoch(self, model, ema, epoch, local_rank, world_size):
        """Trains `self.model` for one epoch over test `self.test_loaders`"""
        raise NotImplementedError

    def get_optimizer_dict(self):
        raise NotImplementedError

    def resume(self, ckpt):
        raise NotImplementedError

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def reset_print_info(self, local_rank, loss_size=4):
        assert loss_size >= 4, "Invalid log parameter"

        self.mloss_per_task = [torch.zeros(loss_size, device=self.device) for _ in self.task_ids]
        self.stat = defaultdict(str)
        self.task_cnt = defaultdict(int)

        if local_rank in [-1, 0]:
            log_headers = "\n"
            log_headers += ("%10s" * 3) % ("task", "epoch", "gpu_mem")
            log_headers += ("%10s" * 4) % ("box", "cls", "dfl", "total")
            if loss_size > 4:
                log_headers += ("%10s" * (loss_size - 4)) % ("loss_item",) * (loss_size - 4)
            log_headers += ("%10s" * 2) % ("labels", "img_size")
            LOGGER.info(log_headers)

    def _log_info(self, epoch, task_id, local_rank, loss_items, nb, imsz):

        if local_rank in [-1, 0]:
            task_i = self.task_ids.index(task_id)
            n_add_losses = self.mloss_per_task[task_i].shape[0] - 4
            self.mloss_per_task[task_i] = (self.mloss_per_task[task_i] * self.task_cnt[task_id] + loss_items) / (
                self.task_cnt[task_id] + 1
            )  # update mean losses
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
            self.stat[task_id] = ("%10s" * 3 + "%10.4g" * (6 + n_add_losses)) % (
                f"{task_id}",
                f"{epoch}/{self.model_manager.epochs - 1}",
                mem,
                *self.mloss_per_task[task_i],
                nb,
                imsz,
            )
            self.task_cnt[task_id] += 1

    @staticmethod
    def warmup_lr(ni, epoch, optimizer, nw, hyp, lf):

        xi = [0, nw]  # x interp
        # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            warmup_bias_lr = get_hyperparameter(hyp, "warmup_bias_lr")
            x["lr"] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
            if "momentum" in x:
                warmup_momentum = get_hyperparameter(hyp, "warmup_momentum")
                momentum = get_hyperparameter(hyp, "momentum")
                x["momentum"] = np.interp(ni, xi, [warmup_momentum, momentum])

    def val_epoch(self, model, ema, epoch, world_size):

        if not self.opt.evolve:
            final_epoch = (epoch + 1 == self.model_manager.epochs) or self.stopper.possible_stop
        else:
            final_epoch = epoch + 1 == self.model_manager.epochs

        if not (not self.opt.noval or final_epoch):
            return {}

        plots = not self.opt.evolve

        ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        fitness_per_task = dict()
        results_per_task = dict()

        for task_i, (task, val_loader) in enumerate(zip(self.task_ids, self.val_loaders)):

            nc = self.model_manager.data_dict["nc"][task_i]

            # mAP
            results, maps, _ = val.run(
                self.model_manager.data_dict,
                batch_size=self.batch_size // world_size * 2,
                imgsz=self.imgsz,
                model=ema.ema,
                single_cls=self.opt.single_cls,
                dataloader=val_loader,
                save_dir=self.model_manager.save_dir,
                verbose=nc < 50 and final_epoch and not self.opt.evolve,
                plots=plots,
                compute_loss=self.compute_loss,
                task_id=task,
                task_ind=task_i,
            )
            results_per_task[task] = results
            self.maps_per_task[task] = maps

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fitness_per_task[task] = fi
            if fi > self.best_fitness_per_task[task]:
                self.best_fitness_per_task[task] = fi

                # Save best task model
                if (not self.opt.nosave) or final_epoch:  # if save
                    # NOTE: just to track models, but not for training resuming later
                    self.model_manager.save_best_task_model(
                        task,
                        epoch,
                        self.best_fitness_per_task,  # does not relate to cur ckpt (other tasks values)
                        self.best_fitness,  # does not relate to cur ckpt
                        model,
                        ema,
                        self.get_optimizer_dict(),
                    )

            # Write
            self.model_manager.val_log(task, results, epoch, is_best=self.best_fitness_per_task[task] == fi)

        # Mean overall fitness
        self.last_fitness = np.mean(list(fitness_per_task.values()))
        print("Cur fitness:", fitness_per_task, self.last_fitness)
        if self.last_fitness > self.best_fitness:
            self.best_fitness = self.last_fitness

        # Save model
        if (not self.opt.nosave) or final_epoch:  # if save
            # save only last model during evolve
            is_best = self.best_fitness == self.last_fitness and not self.opt.evolve
            self.model_manager.save_model(
                epoch,
                self.best_fitness_per_task,
                self.best_fitness,
                model,
                ema,
                self.get_optimizer_dict(),
                is_best,
            )

        return results_per_task
