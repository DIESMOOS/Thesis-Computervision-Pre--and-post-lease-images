from pathlib import Path


def run_llava_on_folder(folder_path: str) -> list:
    """
    Return list of per-image parsed llava outputs.
    Example:
    [
        {
            "image_id": "img1.jpg",
            "parsed_output": {
                "categories_present": ["no_damage"],
                "category_counts": {
                    "damage": 0,
                    "wear": 0,
                    "alteration": 0,
                    "no_damage": 1
                },
                "summary": "No visible inspection-relevant issues."
            }
        }
    ]
    """
    folder = Path(folder_path)
    results = []

    for image_path in folder.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        # TODO: replace mock with real LLaVA inference
        results.append({
            "image_id": image_path.name,
            "parsed_output": {
                "categories_present": ["no_damage"],
                "category_counts": {
                    "damage": 0,
                    "wear": 0,
                    "alteration": 0,
                    "no_damage": 1
                },
                "summary": "No visible inspection-relevant issues."
            }
        })

    return results