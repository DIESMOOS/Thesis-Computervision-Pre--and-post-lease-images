from src.schemas import ImageResult, DetectionBox
from src.config import CATEGORIES


def empty_counts():
    return {cat: 0 for cat in CATEGORIES}


def normalize_yolo_output(image_id: str, raw_detections: list, model_name: str = "yolo") -> ImageResult:
    """
    raw_detections example:
    [
        {"label": "damage", "confidence": 0.81, "bbox": [10, 20, 40, 60]},
        {"label": "wear", "confidence": 0.72, "bbox": [30, 35, 80, 95]},
    ]
    """
    counts = empty_counts()
    detections = []

    for det in raw_detections:
        label = det["label"]
        if label not in CATEGORIES:
            continue
        counts[label] += 1
        detections.append(
            DetectionBox(
                label=label,
                confidence=float(det["confidence"]),
                bbox=det.get("bbox")
            )
        )

    if sum(counts[c] for c in ["damage", "wear", "alteration"]) == 0:
        counts["no_damage"] = 1

    categories_present = [k for k, v in counts.items() if v > 0]
    summary = build_image_summary(counts)

    return ImageResult(
        image_id=image_id,
        model_name=model_name,
        categories_present=categories_present,
        category_counts=counts,
        detections=detections,
        summary=summary
    )


def normalize_llava_output(image_id: str, parsed_json: dict, model_name: str = "llava") -> ImageResult:
    """
    parsed_json example:
    {
        "categories_present": ["damage"],
        "category_counts": {
            "damage": 1,
            "wear": 0,
            "alteration": 0,
            "no_damage": 0
        },
        "summary": "Visible crack in wall."
    }
    """
    counts = empty_counts()
    incoming_counts = parsed_json.get("category_counts", {})

    for cat in CATEGORIES:
        counts[cat] = int(incoming_counts.get(cat, 0))

    if sum(counts[c] for c in ["damage", "wear", "alteration"]) == 0:
        counts["no_damage"] = 1
    else:
        counts["no_damage"] = 0

    categories_present = [k for k, v in counts.items() if v > 0]

    return ImageResult(
        image_id=image_id,
        model_name=model_name,
        categories_present=categories_present,
        category_counts=counts,
        detections=[],
        summary=parsed_json.get("summary", "")
    )


def build_image_summary(counts: dict) -> str:
    parts = []
    for cat in ["damage", "wear", "alteration"]:
        if counts[cat] > 0:
            parts.append(f"{counts[cat]} {cat}")
    if not parts:
        return "No visible inspection-relevant issues."
    return "Detected: " + ", ".join(parts) + "."