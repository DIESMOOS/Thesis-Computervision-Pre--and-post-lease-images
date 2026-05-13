from pathlib import Path
import random
import shutil
import json

random.seed(42)

ROOT = Path(".")
RAW = ROOT / "data" / "Original data folders"
INSPECTION_DATASET = ROOT / "data" / "inspection_dataset"
PROPERTIES_ROOT = ROOT / "data" / "properties"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 10 pre + 10 post per property
PROFILES = {
    "001": {
        "pre": {"no_damage": 6, "wear": 4},
        "post": {"damage": 6, "wear": 4},
    },
    "002": {
        "pre": {"no_damage": 7, "wear": 3},
        "post": {"no_damage": 4, "wear": 3, "damage": 3},
    },
    "003": {
        "pre": {"no_damage": 5, "damage": 5},
        "post": {"no_damage": 2, "damage": 8},
    },
    "004": {
        "pre": {"no_damage": 4, "wear": 3, "damage": 3},
        "post": {"no_damage": 8, "wear": 2},
    },
    "005": {
        "pre": {"no_damage": 8, "wear": 2},
        "post": {"no_damage": 7, "wear": 3},
    },
    "006": {
        "pre": {"no_damage": 6, "wear": 4},
        "post": {"no_damage": 5, "wear": 3, "damage": 2},
    },
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and is_image(p)])


def corresponding_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)

    if "images" not in parts:
        raise ValueError(f"No 'images' folder found in path: {img_path}")

    idx = parts.index("images")
    parts[idx] = "labels"

    return Path(*parts[:-1], img_path.stem + ".txt")


def find_all_images_dirs(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted([p for p in dataset_root.rglob("images") if p.is_dir()])


def collect_dataset_images(dataset_root: Path) -> list[Path]:
    pool = []

    for images_dir in find_all_images_dirs(dataset_root):
        pool.extend(collect_images(images_dir))

    return sorted(set(pool))


def read_house_pools(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    """
    House classes:
    0: Amber
    1: Green
    2: NoDamage
    3: Red

    Rule:
    - only class 2 -> no_damage
    - any other class present -> damage
    """
    damage = []
    no_damage = []

    for images_dir in find_all_images_dirs(dataset_root):
        for img in collect_images(images_dir):
            label_path = corresponding_label_path(img)

            if not label_path.exists():
                continue

            text = label_path.read_text(encoding="utf-8").strip()

            if not text:
                continue

            classes = {line.split()[0] for line in text.splitlines() if line.strip()}

            if classes == {"2"}:
                no_damage.append(img)
            else:
                damage.append(img)

    return sorted(set(damage)), sorted(set(no_damage))


def sample_without_reuse(pool: list[Path], n: int, used: set[Path]) -> list[Path]:
    available = [p for p in pool if p not in used]

    if len(available) < n:
        raise ValueError(
            f"Not enough images left. Needed {n}, available {len(available)}"
        )

    chosen = random.sample(available, n)
    used.update(chosen)

    return chosen


def copy_images(images: list[Path], target_dir: Path, category: str):
    target_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(images, start=1):
        dst = target_dir / f"{category}_{i:02d}_{src.name}"
        shutil.copy2(src, dst)


def write_old_report(property_id: str, pre_counts: dict):
    damage = pre_counts.get("damage", 0)
    wear = pre_counts.get("wear", 0)
    no_damage = pre_counts.get("no_damage", 0)

    if damage > 0 and wear > 0:
        summary = "Previous inspection showed visible damage and surface wear."
    elif damage > 0:
        summary = "Previous inspection showed visible damage."
    elif wear > 0:
        summary = "Previous inspection showed surface wear."
    else:
        summary = "Previous inspection showed no visible inspection-relevant issues."

    report = {
        "property_id": property_id,
        "category_counts": {
            "damage": damage,
            "wear": wear,
            "alteration": 0,
            "no_damage": no_damage,
        },
        "summary": summary,
        "inspection_recommended": damage > 0,
    }

    out_path = PROPERTIES_ROOT / property_id / "old_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def clear_old_properties():
    for property_id in PROFILES:
        prop_dir = PROPERTIES_ROOT / property_id

        if not prop_dir.exists():
            continue

        for subfolder in ["pre_lease", "post_lease"]:
            subdir = prop_dir / subfolder

            if subdir.exists():
                for file_path in subdir.glob("*"):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                    except PermissionError:
                        print(f"Could not delete file: {file_path}")

        old_report = prop_dir / "old_report.json"

        if old_report.exists():
            try:
                old_report.unlink()
            except PermissionError:
                print(f"Could not delete file: {old_report}")


def print_dataset_debug():
    print("RAW PATH:", RAW.resolve())
    print("RAW EXISTS:", RAW.exists())
    print("INSPECTION DATASET:", INSPECTION_DATASET.resolve())
    print("INSPECTION DATASET EXISTS:", INSPECTION_DATASET.exists())
    print()

    for name in [
        "paint",
        "crack",
        "house",
        "surface damage",
        "asbestos",
        "mold",
        "mold2",
    ]:
        dataset_root = RAW / name
        image_dirs = find_all_images_dirs(dataset_root)
        image_count = len(collect_dataset_images(dataset_root))

        print(f"{name}:")
        print(f"  exists: {dataset_root.exists()}")
        print(f"  image dirs: {len(image_dirs)}")
        print(f"  images: {image_count}")

    inspection_images = collect_images(INSPECTION_DATASET / "images")
    print()
    print(f"inspection_dataset images: {len(inspection_images)}")
    print()


def main():
    print_dataset_debug()

    crack_pool = collect_dataset_images(RAW / "crack")
    surface_damage_pool = collect_dataset_images(RAW / "surface damage")
    paint_pool = collect_dataset_images(RAW / "paint")

    house_damage_pool, house_nodamage_pool = read_house_pools(RAW / "house")

    damage_pool = sorted(set(crack_pool + surface_damage_pool + house_damage_pool))
    wear_pool = sorted(set(paint_pool))

    # Primary source: house images labeled only as NoDamage.
    no_damage_pool = sorted(set(house_nodamage_pool))

    # Fallback: if house has too few no_damage images, use inspection_dataset images.
    # This is acceptable for fake properties because these are synthetic property folders.
    needed_no_damage = sum(
        phase_counts.get("no_damage", 0)
        for property_config in PROFILES.values()
        for phase_counts in property_config.values()
    )

    if len(no_damage_pool) < needed_no_damage:
        fallback_no_damage = collect_images(INSPECTION_DATASET / "images")
        no_damage_pool = sorted(set(no_damage_pool + fallback_no_damage))

    print(f"damage pool: {len(damage_pool)}")
    print(f"wear pool: {len(wear_pool)}")
    print(f"no_damage pool: {len(no_damage_pool)}")
    print()

    needed = {
        "damage": sum(
            phase_counts.get("damage", 0)
            for property_config in PROFILES.values()
            for phase_counts in property_config.values()
        ),
        "wear": sum(
            phase_counts.get("wear", 0)
            for property_config in PROFILES.values()
            for phase_counts in property_config.values()
        ),
        "no_damage": needed_no_damage,
    }

    print("Needed images:")
    print(needed)
    print()

    if len(wear_pool) < needed["wear"]:
        raise ValueError(
            f"Not enough wear images. Needed {needed['wear']}, found {len(wear_pool)}."
        )

    if len(no_damage_pool) < needed["no_damage"]:
        raise ValueError(
            f"Not enough no_damage images. Needed {needed['no_damage']}, found {len(no_damage_pool)}."
        )

    if len(damage_pool) < needed["damage"]:
        raise ValueError(
            f"Not enough damage images. Needed {needed['damage']}, found {len(damage_pool)}."
        )

    pools = {
        "damage": damage_pool,
        "wear": wear_pool,
        "no_damage": no_damage_pool,
    }

    clear_old_properties()

    used = set()

    for property_id, config in PROFILES.items():
        prop_dir = PROPERTIES_ROOT / property_id
        pre_dir = prop_dir / "pre_lease"
        post_dir = prop_dir / "post_lease"

        pre_counts = {}

        for phase, target_dir in [("pre", pre_dir), ("post", post_dir)]:
            for category, count in config[phase].items():
                chosen = sample_without_reuse(pools[category], count, used)
                copy_images(chosen, target_dir, category)

                if phase == "pre":
                    pre_counts[category] = count

        write_old_report(property_id, pre_counts)

    print("Finished building 6 fake properties with 20 images each.")


if __name__ == "__main__":
    main()