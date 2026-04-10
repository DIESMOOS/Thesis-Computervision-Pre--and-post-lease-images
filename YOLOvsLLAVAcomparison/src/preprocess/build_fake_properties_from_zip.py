from pathlib import Path
import random
import shutil
import json

random.seed(42)

ROOT = Path(".")
RAW = ROOT / "data"
PROPERTIES_ROOT = ROOT / "data" / "properties"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

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
    "004": {  # improvement case
        "pre": {"no_damage": 4, "wear": 3, "damage": 3},
        "post": {"no_damage": 8, "wear": 2},
    },
    "005": {  # mostly stable
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
    """
    Converts:
    .../images/abc.jpg -> .../labels/abc.txt
    """
    parts = list(img_path.parts)
    if "images" not in parts:
        raise ValueError(f"No 'images' folder found in path: {img_path}")
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts[:-1], img_path.stem + ".txt")


def find_all_images_dirs(dataset_root: Path) -> list[Path]:
    """
    Finds all directories named 'images' anywhere below dataset_root.
    This avoids assumptions about train/valid/test naming.
    """
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
    """
    Safer on Windows/OneDrive:
    do not remove folders, only delete files inside them.
    """
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


def main():
    print("RAW PATH:", RAW.resolve())
    print("RAW EXISTS:", RAW.exists())
    print("paint images dirs found:", find_all_images_dirs(RAW / "paint"))

    crack_pool = collect_dataset_images(RAW / "crack")
    paint_pool = collect_dataset_images(RAW / "paint")
    house_damage_pool, house_nodamage_pool = read_house_pools(RAW / "house")

    damage_pool = sorted(set(crack_pool + house_damage_pool))
    wear_pool = sorted(set(paint_pool))
    no_damage_pool = sorted(set(house_nodamage_pool))

    print(f"damage pool: {len(damage_pool)}")
    print(f"wear pool: {len(wear_pool)}")
    print(f"no_damage pool: {len(no_damage_pool)}")

    if len(wear_pool) < 17:
        raise ValueError("Not enough wear images for the current profiles.")
    if len(no_damage_pool) < 42:
        raise ValueError("Not enough no_damage images for the current profiles.")
    if len(damage_pool) < 24:
        raise ValueError("Not enough damage images for the current profiles.")

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