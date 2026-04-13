import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from pycocotools import mask as mask_utils

BALLOON_URL = "https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip"


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Balloon and convert it to COCO with RLE encoding."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Root directory where the dataset will be stored.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=BALLOON_URL,
        help="Dataset zip URL.",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="balloon_dataset.zip",
        help="Local name for the downloaded zip file.",
    )
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="Delete the zip file after extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload / reextract even if files already exist.",
    )
    return parser.parse_args()


def dataset_ready(root: Path) -> bool:
    train_ann = list(root.rglob("train/via_region_data.json"))
    val_ann = list(root.rglob("val/via_region_data.json"))
    return len(train_ann) > 0 and len(val_ann) > 0


def download_file(url: str, dst: Path, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        log(f"Zip already exists: {dst}")
        return

    log(f"Downloading {url}")
    last_percent = {"value": -1}

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            if percent != last_percent["value"]:
                last_percent["value"] = percent
                done_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(
                    f"\r[INFO] Downloading... {percent:3d}% "
                    f"({done_mb:.1f}/{total_mb:.1f} MB)",
                    end="",
                    flush=True,
                )
        else:
            done_mb = downloaded / (1024 * 1024)
            print(
                f"\r[INFO] Downloading... {done_mb:.1f} MB",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(url, dst, reporthook=reporthook)
    print()
    log(f"Downloaded to {dst}")


def safe_extract_zip(zip_path: Path, out_dir: Path, overwrite: bool = False) -> None:
    log(f"Extracting {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)

        for i, member in enumerate(members, start=1):
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")

            if target.exists() and not overwrite:
                continue

            zf.extract(member, out_dir)

            if i == 1 or i == total or i % 10 == 0:
                print(f"\r[INFO] Extracting... {i}/{total}", end="", flush=True)

    print()
    log("Extraction done")


def find_split_dir(root: Path, split: str) -> Path:
    matches = sorted(root.rglob(f"{split}/via_region_data.json"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {split}/via_region_data.json under {root}"
        )
    return matches[0].parent


def polygon_to_rle(all_points_x, all_points_y, height: int, width: int):
    poly = []
    for x, y in zip(all_points_x, all_points_y):
        poly.extend([float(x) + 0.5, float(y) + 0.5])

    rles = mask_utils.frPyObjects([poly], height, width)
    rle = mask_utils.merge(rles)

    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def convert_balloon_split_to_coco(
    ann_file: Path, out_file: Path, image_dir: Path
) -> None:
    log(f"Converting {ann_file}")

    with open(ann_file, "r", encoding="utf-8") as f:
        data_infos = json.load(f)

    images = []
    annotations = []
    ann_id = 1
    values = list(data_infos.values())
    total = len(values)

    for img_id, item in enumerate(values, start=1):
        filename = item["filename"]
        img_path = image_dir / filename

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as im:
            width, height = im.size

        images.append(
            {
                "id": img_id,
                "file_name": filename,
                "height": height,
                "width": width,
            }
        )

        regions = item.get("regions", [])
        if isinstance(regions, dict):
            regions = regions.values()

        for region in regions:
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue

            px = shape["all_points_x"]
            py = shape["all_points_y"]

            rle = polygon_to_rle(px, py, height, width)
            bbox = mask_utils.toBbox(rle).tolist()
            area = float(mask_utils.area(rle))

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(v) for v in bbox],
                    "area": area,
                    "segmentation": rle,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        if img_id == 1 or img_id == total or img_id % 5 == 0:
            print(
                f"\r[INFO] Converting images... {img_id}/{total} | "
                f"annotations: {len(annotations)}",
                end="",
                flush=True,
            )

    print()

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "balloon",
            }
        ],
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    log(f"Saved {out_file} " f"({len(images)} images, {len(annotations)} annotations)")


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / args.zip_name

    if not dataset_ready(root) or args.overwrite:
        download_file(args.url, zip_path, overwrite=args.overwrite)
        safe_extract_zip(zip_path, root, overwrite=args.overwrite)
        if args.delete_zip and zip_path.exists():
            zip_path.unlink()
            log(f"Deleted {zip_path}")
    else:
        log("Dataset already extracted, skipping download/extraction.")

    train_dir = find_split_dir(root, "train")
    val_dir = find_split_dir(root, "val")

    convert_balloon_split_to_coco(
        ann_file=train_dir / "via_region_data.json",
        out_file=train_dir / "annotation_coco_rle.json",
        image_dir=train_dir,
    )
    convert_balloon_split_to_coco(
        ann_file=val_dir / "via_region_data.json",
        out_file=val_dir / "annotation_coco_rle.json",
        image_dir=val_dir,
    )

    log("All done.")
    log(f"Train JSON: {train_dir / 'annotation_coco_rle.json'}")
    log(f"Val JSON:   {val_dir / 'annotation_coco_rle.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
