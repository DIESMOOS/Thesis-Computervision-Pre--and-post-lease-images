import random
import shutil
from pathlib import Path


# =========================================
# PATH CONFIG
# =========================================
ROOT_DIR = Path(__file__).resolve().parents[2]

SOURCE_DATA_DIR = ROOT_DIR / "data" / "Original data folders"
DATASET_DIR = ROOT_DIR / "data" / "inspection_dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


# =========================================
# FINAL YOLO CLASSES
# =========================================
TARGET_CLASSES = {
    "damage": 0,
    "crack": 1,
    "mold": 2,
    "wear": 3,
    "asbestos": 4,
}


# =========================================
# ORIGINAL DATASET LABEL MAPPING
# =========================================
CLASS_MAPPING = {
    "crack": {
        0: "crack",
    },

    "paint": {
        0: "wear",
    },

    "mold": {
        0: "crack",
        1: "mold",
        2: "wear",
        3: "crack",
        4: "wear",
    },

    "mold2": {
        0: "mold",
    },

    "house": {
        0: "damage",
        1: "damage",
        2: None,       # NoDamage ignored for YOLO object detection
        3: "damage",
    },

    "surface damage": {
        0: "damage",
    },

    "asbestos": {
        0: "asbestos",
        1: "asbestos",
        2: "asbestos",
        3: "asbestos",
    },
}


# =========================================
# FOLDER SETUP
# =========================================
def make_dirs():
    for split in ["all", "train", "val", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Folder structure checked/created.")


def clear_folder(folder: Path):
    if not folder.exists():
        return

    for file in folder.glob("*"):
        if file.is_file():
            file.unlink()


def clear_output_folders():
    for split in ["all", "train", "val", "test"]:
        clear_folder(DATASET_DIR / "images" / split)
        clear_folder(DATASET_DIR / "labels" / split)

    print("Old generated inspection_dataset files cleared.")


def create_data_yaml():
    yaml_path = DATASET_DIR / "data.yaml"

    content = """path: data/inspection_dataset

train: images/train
val: images/val
test: images/test

names:
  0: damage
  1: crack
  2: mold
  3: wear
  4: asbestos
"""

    yaml_path.write_text(content, encoding="utf-8")
    print(f"data.yaml created at: {yaml_path}")


# =========================================
# DATASET DISCOVERY
# =========================================
def find_images(dataset_dir: Path) -> list[Path]:
    return [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def find_label_for_image(image_path: Path) -> Path | None:
    label_name = image_path.stem + ".txt"

    candidates = [
        image_path.parent.parent / "labels" / label_name,
        image_path.parent / "labels" / label_name,
        image_path.parent / label_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    possible = list(image_path.parent.parent.rglob(label_name))

    if possible:
        return possible[0]

    return None


# =========================================
# LABEL CONVERSION
# =========================================
def segmentation_to_bbox(values: list[str]):
    coords = list(map(float, values))

    xs = coords[0::2]
    ys = coords[1::2]

    xs = [min(max(x, 0.0), 1.0) for x in xs]
    ys = [min(max(y, 0.0), 1.0) for y in ys]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def remap_label_file(source_label: Path, target_label: Path, dataset_name: str) -> bool:
    mapping = CLASS_MAPPING[dataset_name]
    output_lines = []

    lines = source_label.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        original_class_id = int(float(parts[0]))

        if original_class_id not in mapping:
            continue

        target_class = mapping[original_class_id]

        # None = NoDamage / ignored class
        if target_class is None:
            continue

        target_class_id = TARGET_CLASSES[target_class]

        # YOLO bbox format: class x_center y_center width height
        if len(parts) == 5:
            x, y, w, h = map(float, parts[1:5])

        # YOLO segmentation format: class x1 y1 x2 y2 ...
        else:
            x, y, w, h = segmentation_to_bbox(parts[1:])

        if w <= 0 or h <= 0:
            continue

        output_lines.append(
            f"{target_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    # Empty label files are valid. They mean no visible target object.
    target_label.write_text("\n".join(output_lines), encoding="utf-8")
    return True


# =========================================
# MERGE DATASETS
# =========================================
def merge_datasets():
    images_all = DATASET_DIR / "images" / "all"
    labels_all = DATASET_DIR / "labels" / "all"

    total_copied = 0
    total_skipped = 0

    for dataset_name in CLASS_MAPPING:
        source_dir = SOURCE_DATA_DIR / dataset_name

        if not source_dir.exists():
            print(f"\nSkipping missing dataset: {dataset_name}")
            continue

        images = find_images(source_dir)

        print(f"\nDataset: {dataset_name}")
        print(f"Found images: {len(images)}")

        copied = 0
        skipped = 0

        for image_path in images:
            label_path = find_label_for_image(image_path)

            if label_path is None:
                if skipped < 10:
                    print(f"Missing label for: {image_path}")
                skipped += 1
                continue

            new_stem = f"{dataset_name}_{image_path.stem}"
            target_image = images_all / f"{new_stem}{image_path.suffix.lower()}"
            target_label = labels_all / f"{new_stem}.txt"

            shutil.copy2(image_path, target_image)
            remap_label_file(label_path, target_label, dataset_name)

            copied += 1

        print(f"Copied: {copied}")
        print(f"Skipped: {skipped}")

        total_copied += copied
        total_skipped += skipped

    print("\nMerge completed.")
    print(f"Total copied to images/all: {total_copied}")
    print(f"Total skipped: {total_skipped}")

    if total_copied == 0:
        raise ValueError("No images copied. Check SOURCE_DATA_DIR and dataset folders.")


# =========================================
# TRAIN / VAL / TEST SPLIT
# =========================================
def split_dataset():
    images_all = DATASET_DIR / "images" / "all"
    labels_all = DATASET_DIR / "labels" / "all"

    images = [
        p for p in images_all.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not images:
        raise ValueError(f"No images found in {images_all}")

    random.seed(SEED)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    print("\nSplitting dataset...")

    for split, split_images in split_map.items():
        for image_path in split_images:
            label_path = labels_all / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            shutil.copy2(
                image_path,
                DATASET_DIR / "images" / split / image_path.name,
            )

            shutil.copy2(
                label_path,
                DATASET_DIR / "labels" / split / label_path.name,
            )

        print(f"{split}: {len(split_images)} images")

    print("Split completed.")


def main():
    make_dirs()
    clear_output_folders()
    create_data_yaml()
    merge_datasets()
    split_dataset()


if __name__ == "__main__":
    main()