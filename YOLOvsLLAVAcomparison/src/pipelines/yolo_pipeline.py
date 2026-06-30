import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"  # later: "models/yolo/best.pt"
CONF_THRESHOLD = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

THESIS_CATEGORIES = ["damage", "wear", "alteration", "no_damage"]

CATEGORY_MAP = {
    "crack": "damage",
    "damage": "damage",
    "broken": "damage",
    "hole": "damage",
    "rust": "wear",
    "paint": "wear",
    "peeling": "wear",
    "stain": "wear",
}

DEFAULT_FALLBACK_LABEL = "damage"


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


def map_label(raw_label: str) -> str:
    raw_label = raw_label.lower().strip()

    for keyword, thesis_label in CATEGORY_MAP.items():
        if keyword in raw_label:
            return thesis_label

    return DEFAULT_FALLBACK_LABEL


def empty_category_counts() -> dict:
    return {category: 0 for category in THESIS_CATEGORIES}


def summarize_detections(detections: list[dict]) -> dict:
    counts = empty_category_counts()

    if not detections:
        counts["no_damage"] = 1
        return {
            "categories_present": ["no_damage"],
            "category_counts": counts,
            "summary": "No visible inspection-relevant issues."
        }

    for det in detections:
        counts[det["label"]] += 1

    categories_present = [
        category for category, count in counts.items()
        if count > 0 and category != "no_damage"
    ]

    return {
        "categories_present": categories_present,
        "category_counts": counts,
        "summary": f"Detected inspection-relevant categories: {', '.join(categories_present)}."
    }


def run_yolo_on_folder(folder_path: str) -> list[dict]:
    image_paths = get_image_paths(folder_path)
    model = YOLO(MODEL_PATH)

    results = []

    for image_path in image_paths:
        result = model(str(image_path), verbose=False)[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence < CONF_THRESHOLD:
                    continue

                class_id = int(box.cls[0])
                raw_label = model.names[class_id]
                mapped_label = map_label(raw_label)

                bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]

                detections.append({
                    "raw_label": raw_label,
                    "label": mapped_label,
                    "confidence": round(confidence, 4),
                    "bbox": bbox
                })

        parsed_output = summarize_detections(detections)

        results.append({
            "image_id": image_path.name,
            "model_name": "yolo",
            "detections": detections,
            "parsed_output": parsed_output
        })

    return results


if __name__ == "__main__":
    test_folder = "data/properties/001/post_lease"
    output = run_yolo_on_folder(test_folder)

    print(f"Processed {len(output)} images")

    for item in output[:3]:
        print(item)