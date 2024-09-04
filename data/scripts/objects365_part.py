import os
import shutil
from loguru import logger

from cerberusdet.utils.general import Path, download, np, xyxy2xywhn

# check_requirements('pycocotools>=2.0')
from pycocotools.coco import COCO
from tqdm import tqdm


def download_archive(urls, dir):
    for url in urls:
        archive_name = os.path.basename(url)
        download([url], dir=dir, curl=True, delete=True, threads=1)
        archive_path = os.path.join(dir, archive_name)
        if os.path.exists(archive_path):
            logger.warning(f"Downloading archive again: {archive_name}")
            os.remove(archive_path)
            download_archive([url], dir)


if __name__ == '__main__':
    yaml = {"path": "/data/Objects365_part"}

    # ['Monkey', 'Rabbit', 'Yak', 'Antelope', 'Pig',  'Bear', 'Deer', 'Giraffe', 'Zebra', 'Elephant',
    # 'Lion', 'Donkey', 'Camel', 'Jellyfish', 'Other Fish', 'Dolphin', 'Crab', 'Seal', 'Goldfish']
    animals_categories_ids = [341, 342, 344, 318, 300, 295, 240, 180, 178, 144, 324, 323, 307, 330, 103, 326, 311, 320,
                              273]

    subsets = {
        "animals": animals_categories_ids,
        # "all": None,
    }

    out_images_dir_names = [f"images/{subset_name}" for subset_name in subsets.keys()]
    out_labels_dir_names = [f"labels/{subset_name}" for subset_name in subsets.keys()]

    # Make Directories
    dir = Path(yaml["path"])  # dataset root dir
    for p in ["tmp_images"] + out_images_dir_names + out_labels_dir_names:
        (dir / p).mkdir(parents=True, exist_ok=True)
        for q in "train", "val":
            (dir / p / q).mkdir(parents=True, exist_ok=True)

    # Train, Val Splits
    for split, patches in [("val", 43 + 1), ("train", 50 + 1)]:
        print(f"Processing {split} in {patches} patches ...")
        tmp_images = dir / "tmp_images" / split
        # Download
        url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"
        if split == "train":
            download([f"{url}zhiyuan_objv2_{split}.tar.gz"], dir=dir, delete=True)  # annotations json
            download_archive([f"{url}patch{i}.tar.gz" for i in range(patches)], tmp_images)
        elif split == "val":
            download([f"{url}zhiyuan_objv2_{split}.json"], dir=dir, delete=False)  # annotations json
            download_archive([f"{url}images/v1/patch{i}.tar.gz" for i in range(15 + 1)], tmp_images)
            download_archive([f"{url}images/v2/patch{i}.tar.gz" for i in range(16, patches)], tmp_images)

        # Move
        for f in tqdm(tmp_images.rglob("*.jpg"), desc=f"Moving {split} images"):
            f.rename(tmp_images / f.name)  # move to /tmp_images/{split}

        # Labels
        coco = COCO(dir / f"zhiyuan_objv2_{split}.json")
        names = [x["name"] for x in coco.loadCats(coco.getCatIds())]

        images_to_save = set()
        for subset_name, categories_ids in subsets.items():
            for cid, cat in enumerate(names):
                if categories_ids is not None and cid not in categories_ids:
                    continue
                catIds = coco.getCatIds(catNms=[cat])
                imgIds = coco.getImgIds(catIds=catIds)
                for im in tqdm(coco.loadImgs(imgIds), desc=f"Scan {cid + 1}/{len(names)} {cat}"):
                    path = tmp_images / Path(im["file_name"]).name  # image filename
                    images_to_save.add(str(path))
                    # print(path)

        for f in tqdm(tmp_images.rglob("*.jpg")):
            if str(f) not in images_to_save:
                os.remove(str(f))
                print(f"Remove {f}")

        for subset_name, categories_ids in subsets.items():
            missed = 0
            n_images = 0
            images = dir / f"images/{subset_name}" / split
            labels = dir / f"labels/{subset_name}" / split
            for cid, cat in enumerate(names):
                if categories_ids is not None:
                    if cid not in categories_ids:
                        continue
                    new_cat_id = categories_ids.index(cid)
                else:
                    new_cat_id = cid
                catIds = coco.getCatIds(catNms=[cat])
                imgIds = coco.getImgIds(catIds=catIds)
                for im in tqdm(coco.loadImgs(imgIds), desc=f"Class {cid + 1}/{len(names)} {cat}"):
                    width, height = im["width"], im["height"]
                    path = tmp_images / Path(im["file_name"]).name  # image filename
                    try:
                        annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=False)
                        annots = coco.loadAnns(annIds)
                        if len(annots) == 0:
                            continue
                        if not path.exists():
                            missed += 1
                            continue
                        with open(labels / path.with_suffix(".txt").name, "a") as file:
                            for a in annots:
                                x, y, w, h = a["bbox"]  # bounding box in xywh (xy top-left corner)
                                xyxy = np.array([x, y, x + w, y + h])[None]  # pixels(1,4)
                                x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # normalized and clipped
                                file.write(f"{new_cat_id} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
                    except Exception as e:
                        print(e)

                    if not (images / path.name).exists():
                        print(f"rename {path} to {images / path.name}")
                        shutil.copyfile(str(path), str(images / path.name))
                        n_images += 1

            print(f"{subset_name}{split} Missed images: {missed} Get images: {n_images}")

        for f in tqdm(tmp_images.rglob("*.jpg"), desc=f"Removing tmp {split} images"):
            os.remove(str(f))
            print(f"Remove {f}")
