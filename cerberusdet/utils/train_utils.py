from cerberusdet.data.dataloaders import create_dataloader
from cerberusdet.utils.general import colorstr


def create_data_loaders(
    data_dict, rank, world_size, opt, hyp, gs, imgsz, val_rect=True, val_pad=0.5, skip_train_load=False, balanced_sampler=True,
):
    workers, noval, batch_size, single_cls = opt.workers, opt.noval, opt.batch_size, opt.single_cls

    task_ids = data_dict["task_ids"]
    train_paths = data_dict["train"]
    val_paths = data_dict["val"]

    train_loaders, val_loaders, train_datasets, valid_datasets = [], [], [], []

    assert len(train_paths) == len(val_paths)
    # Datasets loader
    for task_ind, (task_id, train_data_path, val_data_path) in enumerate(zip(task_ids, train_paths, val_paths)):

        train_loader, dataset = None, None
        if not skip_train_load:
            train_loader, dataset = create_dataloader(
                train_data_path,
                imgsz,
                opt.batch_size[task_ind] if isinstance(opt.batch_size, list) else opt.batch_size,
                gs,
                single_cls,
                hyp=hyp,
                augment=True,
                cache=opt.cache_images,
                rank=rank,
                workers=workers,
                prefix=colorstr("train: "),
                task_ind=task_ind,
                balanced_sampler=balanced_sampler,
                task_names=task_ids,
                classnames=data_dict["names"][task_ind],
                labels_from_xml=opt.labels_from_xml,
                as_multi_label=opt.use_multi_labels,
                as_soft_label=opt.use_soft_labels,
            )

        val_loader, val_dataset = None, None
        if rank in [-1, 0]:
            val_loader, val_dataset = create_dataloader(
                val_data_path,
                imgsz,
                max(opt.batch_size) if isinstance(opt.batch_size, list) else opt.batch_size,
                gs,
                single_cls,
                hyp=hyp,
                cache=opt.cache_images and not noval,
                rect=val_rect,
                rank=-1,
                workers=workers,
                pad=val_pad,
                prefix=colorstr("val: "),
                task_ind=task_ind,
                classnames=data_dict["names"][task_ind],
                labels_from_xml=opt.labels_from_xml,
                as_multi_label=opt.use_multi_labels,
                as_soft_label=opt.use_soft_labels,
            )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        train_datasets.append(dataset)
        valid_datasets.append(val_dataset)

    return train_loaders, val_loaders, train_datasets, valid_datasets


def get_init_metrics_per_task(model_manager):

    results = {}
    for task_i, task in enumerate(model_manager.task_ids):
        results[task] = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    return results
