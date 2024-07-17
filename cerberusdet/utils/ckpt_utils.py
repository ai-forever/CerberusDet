from cerberusdet.models.cerberus import CerberusDet
from loguru import logger


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values

    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def dict_to_cerber(loaded_dict: dict, model: CerberusDet):
    """Map yolo state_dict to cerberusdet state_dict

    :param loaded_dict: state_dict of original yolo model
    :param model: an instance of cerberusdet model
    :return: state_dict to initialize cerberusdet with weights from loaded_dict
    """

    cerber_state_dict = model.state_dict()
    old_head_n = None

    heads_nums = list(model.heads.values())

    for k, v in loaded_dict.items():
        if ".dfl" in k:
            old_head_n = k.split(".")[1]

    blocks = model.blocks
    key_prefix = "blocks"

    yolo_to_cerber_inds = {}
    # map old indexes to new blocks
    for ind, block in enumerate(blocks):
        if ind == 0:
            next_block_example = blocks[1]
            # backbone
            for old_i in range(next_block_example.i):
                yolo_to_cerber_inds[old_i] = 0
            continue

        cur_block = block
        yolo_to_cerber_inds[cur_block.i] = ind

    # map backbone, neck and head keys
    new_dict = {}
    for k, v in loaded_dict.items():
        if old_head_n is not None and f"model.{old_head_n}." in k:
            # heads
            for i in heads_nums:
                new_cerber_key = f"{key_prefix}.{i}." + ".".join(k.split(".")[2:])
                new_dict[new_cerber_key] = v
        else:
            yolov8_i = int(k.split(".")[1])  # model.24.m.0.bias -> 24

            new_cerber_key = None
            if yolov8_i in yolo_to_cerber_inds and yolo_to_cerber_inds[yolov8_i] == 0:
                # backbone
                new_cerber_key = f"{key_prefix}.0.{k}"
            elif yolov8_i in yolo_to_cerber_inds:
                # neck
                cerber_block_i = yolo_to_cerber_inds[yolov8_i]
                if isinstance(model, CerberusDet):
                    new_cerber_key = f"{key_prefix}.{cerber_block_i}." + ".".join(k.split(".")[2:])
                else:
                    new_cerber_key = [
                        f"{key_prefix}.{cerber_block_i}.path.{task_ind}." + ".".join(k.split(".")[2:])
                        for task_ind in range(len(model.tasks))
                    ]  # type: ignore

            if new_cerber_key is None:
                logger.warning(f"Yolo key has not been mapped: {k}")
                continue

            new_cerber_key = new_cerber_key if isinstance(new_cerber_key, list) else [new_cerber_key]  # type: ignore

            for cerber_key in new_cerber_key:
                if cerber_key not in cerber_state_dict:
                    logger.warning(f"\tKey {cerber_key} has not been found in the current cerberus model dict")
                    continue

                if cerber_state_dict[cerber_key].shape != v.shape:
                    logger.warning(
                        f"\tMismatched shapes for {cerber_key}: "
                        f"old {v.shape} and new {cerber_state_dict[cerber_key].shape}"
                    )
                    continue

                new_dict[cerber_key] = v

    return new_dict
