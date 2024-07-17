import os

import torch
from cerberusdet.data.datasets import LoadImagesAndLabels
from cerberusdet.data.samplers import BalancedBatchSampler, DistributedSamplerWrapper, RepeatSampler
from cerberusdet.utils.torch_utils import torch_distributed_zero_first
from loguru import logger


def _create_dataloader(dataset, workers, batch_size, image_weights=False, rank=-1, use_balanced_sampler=True):
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader

    if use_balanced_sampler and rank == -1:
        logger.info(f"RANK {rank}: Balanced sampler is used")
        sampler = BalancedBatchSampler(dataset)
        loader = torch.utils.data.DataLoader
    elif use_balanced_sampler and rank != -1:
        logger.info(f"RANK {rank}: Distributed balanced sampler is used")
        sampler = BalancedBatchSampler(dataset)
        sampler = DistributedSamplerWrapper(sampler)
        loader = torch.utils.data.DataLoader

    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return dataloader


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    prefix="",
    skip_prefix=None,
    task_ind=None,
    balanced_sampler=False,
    task_names=None,
    classnames=None,
    labels_from_xml=False,
    as_multi_label=False,
    as_soft_label=False,
):
    """
    Create dataset and dataloader
    """

    logger.info(f"{prefix} Loading {path} dataset")
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            skip_prefix=skip_prefix,
            task_ind=task_ind,
            task_names=task_names,
            classnames=classnames,
            labels_from_xml=labels_from_xml,
            as_multi_label=as_multi_label,
            as_soft_label=as_soft_label,
        )

    dataloader = _create_dataloader(dataset, workers, batch_size, image_weights, rank, balanced_sampler)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
