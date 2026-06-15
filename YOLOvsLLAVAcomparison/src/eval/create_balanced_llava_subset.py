from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[2]

IMG_DIRS = [
    ROOT / "data" / "inspection_dataset" / "images" / "train",
    ROOT / "data" / "inspection_dataset" / "images" / "val",
    ROOT / "data" / "inspection_dataset" / "images" / "test",
]

LBL_DIRS = [
    ROOT / "data" / "inspection_dataset" / "labels" / "train",
    ROOT / "data" / "inspection_dataset" / "labels" / "val",
    ROOT / "data" / "inspection_dataset" / "labels" / "test",
]

OUT_IMG = ROOT / "data" / "llava_balanced_test" / "images"
OUT_LBL = ROOT / "data" / "llava_balanced_test" / "labels"

CLASS_MAP = {
    0: "damage",
    1: "crack",
    2: "mold",
    3: "wear",
    4: "asbestos",
}

LABELS = ["damage", "crack", "mold", "wear", "asbestos", "no_damage"]

N_PER_CLASS = 10
SEED = 42


def get_label(label_path: Path) -> str:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "no_damage"

    found = []

    for line in label_path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        cls_id = int(line.split()[0])
        found.append(CLASS_MAP.get(cls_id, "damage"))

    for label in ["mold", "asbestos", "crack", "damage", "wear"]:
        if label in found:
            return label

    return "no_damage"


def main() -> None:
    random.seed(SEED)

    if OUT_IMG.exists():
        shutil.rmtree(OUT_IMG)

    if OUT_LBL.exists():
        shutil.rmtree(OUT_LBL)

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    buckets = {label: [] for label in LABELS}

    for img_dir, lbl_dir in zip(IMG_DIRS, LBL_DIRS):
        for image_path in sorted(img_dir.glob("*")):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_path = lbl_dir / f"{image_path.stem}.txt"
            label = get_label(label_path)

            buckets[label].append((image_path, label_path))

    for label, items in buckets.items():
        selected = random.sample(items, min(N_PER_CLASS, len(items)))

        print(f"{label}: {len(selected)} of {len(items)}")

        for image_path, label_path in selected:
            shutil.copy2(image_path, OUT_IMG / image_path.name)

            output_label_path = OUT_LBL / f"{image_path.stem}.txt"

            if label_path.exists():
                shutil.copy2(label_path, output_label_path)
            else:
                output_label_path.write_text("")


if __name__ == "__main__":
    main()