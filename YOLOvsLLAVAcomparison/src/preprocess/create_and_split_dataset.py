import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


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

TARGET_CLASSES = {
    "damage": 0,
    "crack": 1,
    "mold": 2,
    "wear": 3,
    "asbestos": 4,
}

CLASS_NAMES = {v: k for k, v in TARGET_CLASSES.items()}

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
        2: None,
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

        if target_class is None:
            continue

        target_class_id = TARGET_CLASSES[target_class]

        if len(parts) == 5:
            x, y, w, h = map(float, parts[1:5])
        else:
            x, y, w, h = segmentation_to_bbox(parts[1:])

        if w <= 0 or h <= 0:
            continue

        output_lines.append(
            f"{target_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    target_label.write_text("\n".join(output_lines), encoding="utf-8")
    return True


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

            safe_dataset_name = dataset_name.replace(" ", "_")
            new_stem = f"{safe_dataset_name}_{image_path.stem}"

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


def get_label_classes(label_path: Path) -> set[int]:
    classes = set()

    if not label_path.exists():
        return classes

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            classes.add(int(float(parts[0])))

    return classes


def count_labels_in_split(split: str):
    labels_dir = DATASET_DIR / "labels" / split
    counter = Counter()

    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                counter[int(float(parts[0]))] += 1

    return counter


def copy_split_files(split_map: dict[str, list[Path]]):
    for split, split_images in split_map.items():
        for image_path in split_images:
            label_path = DATASET_DIR / "labels" / "all" / f"{image_path.stem}.txt"

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

    image_classes = {}
    class_image_counts = Counter()
    background_images = []

    for image_path in images:
        label_path = labels_all / f"{image_path.stem}.txt"
        classes = get_label_classes(label_path)
        image_classes[image_path] = classes

        if not classes:
            background_images.append(image_path)

        for cls in classes:
            class_image_counts[cls] += 1

    grouped_images = defaultdict(list)

    for image_path, classes in image_classes.items():
        if not classes:
            continue

        rarest_class = min(classes, key=lambda cls: class_image_counts[cls])
        grouped_images[rarest_class].append(image_path)

    split_map = {
        "train": [],
        "val": [],
        "test": [],
    }

    print("\nClass-aware splitting dataset...")

    for cls_id, class_images in grouped_images.items():
        random.shuffle(class_images)

        total = len(class_images)

        if total < 3:
            split_map["train"].extend(class_images)
            print(f"Warning: class {cls_id} {CLASS_NAMES.get(cls_id)} has only {total} images, placed in train.")
            continue

        train_end = max(1, int(total * SPLIT_RATIOS["train"]))
        val_count = max(1, int(total * SPLIT_RATIOS["val"]))
        val_end = min(total - 1, train_end + val_count)

        split_map["train"].extend(class_images[:train_end])
        split_map["val"].extend(class_images[train_end:val_end])
        split_map["test"].extend(class_images[val_end:])

    random.shuffle(background_images)

    total_bg = len(background_images)
    train_end = int(total_bg * SPLIT_RATIOS["train"])
    val_end = train_end + int(total_bg * SPLIT_RATIOS["val"])

    split_map["train"].extend(background_images[:train_end])
    split_map["val"].extend(background_images[train_end:val_end])
    split_map["test"].extend(background_images[val_end:])

    for split in split_map:
        split_map[split] = list(dict.fromkeys(split_map[split]))

    copy_split_files(split_map)

    print("\nClass counts after split:")
    for split in ["train", "val", "test"]:
        counter = count_labels_in_split(split)
        print(f"\n{split.upper()}")
        for cls_id, cls_name in CLASS_NAMES.items():
            print(f"{cls_id} {cls_name}: {counter[cls_id]}")

    print("\nClass-aware split completed.")


def main():
    make_dirs()
    clear_output_folders()
    create_data_yaml()
    merge_datasets()
    split_dataset()


if __name__ == "__main__":
    main()