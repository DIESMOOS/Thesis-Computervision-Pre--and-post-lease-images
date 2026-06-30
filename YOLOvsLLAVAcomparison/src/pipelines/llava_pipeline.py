from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
THESIS_CATEGORIES = ["damage", "wear", "alteration", "no_damage"]


def get_image_paths(folder_path: str) -> list[Path]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder}")

    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def empty_category_counts() -> dict:
    return {category: 0 for category in THESIS_CATEGORIES}


def parse_llava_text(text: str) -> dict:
    """
    Temporary rule-based parser.
    Later this receives real LLaVA output text.
    """
    text_lower = text.lower()
    counts = empty_category_counts()

    if any(word in text_lower for word in ["crack", "broken", "hole", "damage"]):
        counts["damage"] = 1

    if any(word in text_lower for word in ["wear", "stain", "peeling", "paint", "discoloration"]):
        counts["wear"] = 1

    if any(word in text_lower for word in ["alteration", "modified", "added", "unauthorized"]):
        counts["alteration"] = 1

    categories_present = [
        category for category, count in counts.items()
        if count > 0 and category != "no_damage"
    ]

    if not categories_present:
        counts["no_damage"] = 1
        categories_present = ["no_damage"]

    return {
        "categories_present": categories_present,
        "category_counts": counts,
        "summary": text
    }


def mock_llava_inference(image_path: Path) -> str:
    """
    Replace this function later with real LLaVA inference.
    """
    return "No visible inspection-relevant issues."


def run_llava_on_folder(folder_path: str) -> list[dict]:
    image_paths = get_image_paths(folder_path)
    results = []

    for image_path in image_paths:
        llava_text = mock_llava_inference(image_path)
        parsed_output = parse_llava_text(llava_text)

        results.append({
            "image_id": image_path.name,
            "model_name": "llava",
            "raw_output": llava_text,
            "parsed_output": parsed_output
        })

    return results


if __name__ == "__main__":
    test_folder = "data/properties/001/post_lease"
    output = run_llava_on_folder(test_folder)

    print(f"Processed {len(output)} images")

    for item in output[:3]:
        print(item)