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

# More images per property.
# Each property has a different inspection story:
# - some improve
# - some get worse
# - some stay similar
PROFILES = {
    "001": {
        "story": "deterioration",
        "pre":  {"no_damage": 8, "wear": 4, "damage": 2, "crack": 1},
        "post": {"no_damage": 3, "wear": 4, "damage": 5, "crack": 4, "mold": 2, "asbestos": 2},
    },
    "002": {
        "story": "slight_deterioration",
        "pre":  {"no_damage": 10, "wear": 3, "damage": 1, "crack": 1},
        "post": {"no_damage": 7, "wear": 4, "damage": 3, "crack": 2, "mold": 2, "asbestos": 2},
    },
    "003": {
        "story": "stable_bad",
        "pre":  {"no_damage": 3, "wear": 4, "damage": 4, "crack": 3, "mold": 2, "asbestos": 1},
        "post": {"no_damage": 3, "wear": 4, "damage": 4, "crack": 3, "mold": 2, "asbestos": 1},
    },
    "004": {
        "story": "improvement",
        "pre":  {"no_damage": 3, "wear": 4, "damage": 5, "crack": 3, "mold": 2, "asbestos": 1},
        "post": {"no_damage": 10, "wear": 3, "damage": 1, "crack": 1},
    },
    "005": {
        "story": "stable_good",
        "pre":  {"no_damage": 12, "wear": 3},
        "post": {"no_damage": 12, "wear": 3},
    },
    "006": {
        "story": "mold_problem",
        "pre":  {"no_damage": 8, "wear": 4, "damage": 2},
        "post": {"no_damage": 5, "wear": 3, "damage": 2, "mold": 7, "crack": 2, "asbestos": 1},
    },
    "007": {
        "story": "asbestos_problem",
        "pre":  {"no_damage": 9, "wear": 3, "damage": 2, "crack": 1},
        "post": {"no_damage": 5, "wear": 3, "damage": 2, "crack": 2, "mold": 1, "asbestos": 7},
    },
    "008": {
        "story": "mixed_damage",
        "pre":  {"no_damage": 6, "wear": 4, "damage": 3, "crack": 2},
        "post": {"no_damage": 4, "wear": 4, "damage": 4, "crack": 3, "mold": 2, "asbestos": 3},
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
    - only class 2 = no_damage
    - any other class = damage
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


def sample_without_reuse(pool: list[Path], n: int, used: set[Path], category: str) -> list[Path]:
    if n == 0:
        return []

    available = [p for p in pool if p not in used]

    # If there are not enough unique images, allow reuse.
    # This prevents the script from crashing when a category is small.
    if len(available) < n:
        print(f"Warning: reusing images for category '{category}'. Needed {n}, available unique {len(available)}.")
        chosen = random.choices(pool, k=n)
    else:
        chosen = random.sample(available, n)
        used.update(chosen)

    return chosen


def copy_images(images: list[Path], target_dir: Path, category: str):
    target_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(images, start=1):
        dst = target_dir / f"{category}_{i:02d}_{src.name}"
        shutil.copy2(src, dst)


def write_report(property_id: str, pre_counts: dict, post_counts: dict, story: str):
    report = {
        "property_id": property_id,
        "story": story,
        "category_counts": {
            "damage": pre_counts.get("damage", 0),
            "crack": pre_counts.get("crack", 0),
            "mold": pre_counts.get("mold", 0),
            "wear": pre_counts.get("wear", 0),
            "asbestos": pre_counts.get("asbestos", 0),
            "no_damage": pre_counts.get("no_damage", 0),
        },
        "expected_post_counts": {
            "damage": post_counts.get("damage", 0),
            "crack": post_counts.get("crack", 0),
            "mold": post_counts.get("mold", 0),
            "wear": post_counts.get("wear", 0),
            "asbestos": post_counts.get("asbestos", 0),
            "no_damage": post_counts.get("no_damage", 0),
        },
        "summary": make_summary(pre_counts),
        "inspection_recommended": any(
            pre_counts.get(cat, 0) > 0
            for cat in ["damage", "crack", "mold", "asbestos"]
        ),
    }

    out_path = PROPERTIES_ROOT / property_id / "old_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def make_summary(counts: dict) -> str:
    issues = []

    for cat in ["damage", "crack", "mold", "wear", "asbestos"]:
        if counts.get(cat, 0) > 0:
            issues.append(cat)

    if not issues:
        return "Previous inspection showed no visible inspection-relevant issues."

    return "Previous inspection showed: " + ", ".join(issues) + "."


def clear_old_properties():
    for property_id in PROFILES:
        prop_dir = PROPERTIES_ROOT / property_id

        if not prop_dir.exists():
            continue

        for subfolder in ["pre_lease", "post_lease"]:
            subdir = prop_dir / subfolder

            if subdir.exists():
                for file_path in subdir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()

        old_report = prop_dir / "old_report.json"

        if old_report.exists():
            old_report.unlink()


def print_dataset_debug():
    print("RAW PATH:", RAW.resolve())
    print("RAW EXISTS:", RAW.exists())
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

    print()


def main():
    print_dataset_debug()

    crack_pool = collect_dataset_images(RAW / "crack")
    surface_damage_pool = collect_dataset_images(RAW / "surface damage")
    paint_pool = collect_dataset_images(RAW / "paint")
    mold_pool = collect_dataset_images(RAW / "mold") + collect_dataset_images(RAW / "mold2")
    asbestos_pool = collect_dataset_images(RAW / "asbestos")

    house_damage_pool, house_nodamage_pool = read_house_pools(RAW / "house")

    damage_pool = sorted(set(surface_damage_pool + house_damage_pool))
    crack_pool = sorted(set(crack_pool))
    wear_pool = sorted(set(paint_pool))
    mold_pool = sorted(set(mold_pool))
    asbestos_pool = sorted(set(asbestos_pool))
    no_damage_pool = sorted(set(house_nodamage_pool))

    fallback_no_damage = collect_images(INSPECTION_DATASET / "images")
    no_damage_pool = sorted(set(no_damage_pool + fallback_no_damage))

    pools = {
        "damage": damage_pool,
        "crack": crack_pool,
        "mold": mold_pool,
        "wear": wear_pool,
        "asbestos": asbestos_pool,
        "no_damage": no_damage_pool,
    }

    print("Pools:")
    for category, pool in pools.items():
        print(f"  {category}: {len(pool)}")

    print()

    for category, pool in pools.items():
        if len(pool) == 0:
            raise ValueError(f"No images found for category: {category}")

    clear_old_properties()

    used = set()

    for property_id, config in PROFILES.items():
        prop_dir = PROPERTIES_ROOT / property_id
        pre_dir = prop_dir / "pre_lease"
        post_dir = prop_dir / "post_lease"

        pre_counts = config["pre"]
        post_counts = config["post"]

        for phase, target_dir in [("pre", pre_dir), ("post", post_dir)]:
            for category, count in config[phase].items():
                chosen = sample_without_reuse(
                    pools[category],
                    count,
                    used,
                    category
                )
                copy_images(chosen, target_dir, category)

        write_report(
            property_id=property_id,
            pre_counts=pre_counts,
            post_counts=post_counts,
            story=config["story"]
        )

        print(f"Built property {property_id}: {config['story']}")

    print()
    print(f"Finished building {len(PROFILES)} fake properties.")


if __name__ == "__main__":
    main()