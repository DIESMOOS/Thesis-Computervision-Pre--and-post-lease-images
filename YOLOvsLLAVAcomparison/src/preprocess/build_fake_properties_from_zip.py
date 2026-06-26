from pathlib import Path
import random
import shutil
import json

random.seed(42)

ROOT = Path(".")
RAW = ROOT / "data" / "Original data folders"
PROPERTIES_ROOT = ROOT / "data" / "properties"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORIES = ["damage", "crack", "mold", "wear", "asbestos", "no_damage"]
DEFECT_CATEGORIES = ["damage", "crack", "mold", "wear", "asbestos"]

PROFILES = {
    "001": {
        "pre":  {"damage": 3, "crack": 2, "mold": 0, "wear": 3, "asbestos": 0, "no_damage": 12},
        "post": {"damage": 5, "crack": 3, "mold": 0, "wear": 4, "asbestos": 0, "no_damage": 8},
    },

    "002": {
        "pre":  {"damage": 3, "crack": 1, "mold": 0, "wear": 4, "asbestos": 0, "no_damage": 12},
        "post": {"damage": 4, "crack": 2, "mold": 0, "wear": 4, "asbestos": 0, "no_damage": 10},
    },

    "003": {
        "pre":  {"damage": 6, "crack": 4, "mold": 2, "wear": 4, "asbestos": 1, "no_damage": 3},
        "post": {"damage": 6, "crack": 4, "mold": 2, "wear": 4, "asbestos": 1, "no_damage": 3},
    },

    "004": {
        "pre":  {"damage": 7, "crack": 4, "mold": 2, "wear": 3, "asbestos": 1, "no_damage": 3},
        "post": {"damage": 3, "crack": 2, "mold": 1, "wear": 3, "asbestos": 0, "no_damage": 11},
    },

    "005": {
        "pre":  {"damage": 1, "crack": 1, "mold": 0, "wear": 4, "asbestos": 0, "no_damage": 14},
        "post": {"damage": 0, "crack": 0, "mold": 0, "wear": 4, "asbestos": 0, "no_damage": 16},
    },

    "006": {
        "pre":  {"damage": 2, "crack": 0, "mold": 1, "wear": 3, "asbestos": 0, "no_damage": 14},
        "post": {"damage": 4, "crack": 0, "mold": 3, "wear": 4, "asbestos": 0, "no_damage": 9},
    },

    "007": {
        "pre":  {"damage": 3, "crack": 2, "mold": 0, "wear": 3, "asbestos": 0, "no_damage": 12},
        "post": {"damage": 4, "crack": 3, "mold": 0, "wear": 4, "asbestos": 0, "no_damage": 9},
    },

    "008": {
        "pre":  {"damage": 6, "crack": 3, "mold": 1, "wear": 4, "asbestos": 0, "no_damage": 6},
        "post": {"damage": 5, "crack": 3, "mold": 0, "wear": 4, "asbestos": 0, "no_damage": 8},
    },
}

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and is_image(p))


def corresponding_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    if "images" not in parts:
        return Path("__missing__")
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts[:-1], img_path.stem + ".txt")


def find_all_images_dirs(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted(p for p in dataset_root.rglob("images") if p.is_dir())


def collect_dataset_images(dataset_root: Path) -> list[Path]:
    pool = []
    for images_dir in find_all_images_dirs(dataset_root):
        pool.extend(collect_images(images_dir))
    return sorted(set(pool))


def read_house_pools(dataset_root: Path) -> tuple[list[Path], list[Path]]:
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
        raise ValueError(f"Not enough images left. Needed {n}, available {len(available)}")

    chosen = random.sample(available, n)
    used.update(chosen)
    return chosen


def copy_images(images: list[Path], target_dir: Path, category: str):
    target_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(images, start=1):
        dst = target_dir / f"{category}_{i:02d}_{src.name}"
        shutil.copy2(src, dst)


def build_summary(counts: dict) -> str:
    parts = []
    for cat in DEFECT_CATEGORIES:
        if counts.get(cat, 0) > 0:
            parts.append(f"{counts[cat]} {cat}")

    if not parts:
        return "Previous inspection showed no visible inspection relevant issues."

    return "Previous inspection showed " + ", ".join(parts) + "."


def write_old_report(property_id: str, pre_counts: dict):
    report_counts = {cat: int(pre_counts.get(cat, 0)) for cat in CATEGORIES}

    report = {
        "property_id": property_id,
        "category_counts": report_counts,
        "summary": build_summary(report_counts),
        "inspection_recommended": any(report_counts.get(cat, 0) > 0 for cat in DEFECT_CATEGORIES),
    }

    out_path = PROPERTIES_ROOT / property_id / "old_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def clear_properties():
    if PROPERTIES_ROOT.exists():
        shutil.rmtree(PROPERTIES_ROOT)
    PROPERTIES_ROOT.mkdir(parents=True, exist_ok=True)


def print_debug(pools: dict):
    print("RAW:", RAW.resolve())
    print("RAW exists:", RAW.exists())
    print()
    print("Available pools:")
    for cat in CATEGORIES:
        print(f"{cat}: {len(pools[cat])}")

    needed = {cat: 0 for cat in CATEGORIES}

    for property_config in PROFILES.values():
        for phase in ["pre", "post"]:
            for cat, n in property_config[phase].items():
                needed[cat] += n

    print()
    print("Needed images:")
    for cat in CATEGORIES:
        print(f"{cat}: {needed[cat]}")
    print()


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Original data folder not found: {RAW}")

    crack_pool = collect_dataset_images(RAW / "crack")
    mold_pool = sorted(set(collect_dataset_images(RAW / "mold") + collect_dataset_images(RAW / "mold2")))
    wear_pool = collect_dataset_images(RAW / "paint")
    asbestos_pool = collect_dataset_images(RAW / "asbestos")

    house_damage_pool, house_no_damage_pool = read_house_pools(RAW / "house")
    surface_damage_pool = collect_dataset_images(RAW / "surface damage")

    damage_pool = sorted(set(surface_damage_pool + house_damage_pool))
    no_damage_pool = sorted(set(house_no_damage_pool))

    pools = {
        "damage": damage_pool,
        "crack": crack_pool,
        "mold": mold_pool,
        "wear": wear_pool,
        "asbestos": asbestos_pool,
        "no_damage": no_damage_pool,
    }

    print_debug(pools)

    needed = {cat: 0 for cat in CATEGORIES}
    for property_config in PROFILES.values():
        for phase in ["pre", "post"]:
            for cat, n in property_config[phase].items():
                needed[cat] += n

    for cat in CATEGORIES:
        if len(pools[cat]) < needed[cat]:
            raise ValueError(f"Not enough {cat} images. Needed {needed[cat]}, found {len(pools[cat])}")

    clear_properties()
    used = set()

    for property_id, profile in PROFILES.items():
        prop_dir = PROPERTIES_ROOT / property_id
        pre_dir = prop_dir / "pre_lease"
        post_dir = prop_dir / "post_lease"

        pre_dir.mkdir(parents=True, exist_ok=True)
        post_dir.mkdir(parents=True, exist_ok=True)

        for phase, target_dir in [("pre", pre_dir), ("post", post_dir)]:
            for cat, count in profile[phase].items():
                chosen = sample_without_reuse(pools[cat], count, used)
                copy_images(chosen, target_dir, cat)

        write_old_report(property_id, profile["pre"])

        print(f"{property_id}: created 40 pre images and 40 post images")

    print()
    print("Done. Properties created in data/properties")


if __name__ == "__main__":
    main()