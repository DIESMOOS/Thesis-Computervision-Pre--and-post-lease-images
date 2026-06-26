from src.schemas import ImageResult, DetectionBox
from src.config import CATEGORIES

DEFECT_CATEGORIES = ["damage", "crack", "mold", "wear", "asbestos", "no_damage"]

def empty_counts():
    return {cat: 0 for cat in CATEGORIES}


def normalize_label(label: str) -> str:
    label = str(label).strip().lower().replace(" ", "_")
    if label in {"nodamage", "no_damage", "no defect", "none"}:
        return "no_damage"
    return label


def build_image_summary(counts: dict) -> str:
    parts = []
    for cat in DEFECT_CATEGORIES:
        if counts.get(cat, 0) > 0:
            parts.append(cat)

    if not parts:
        return "No visible inspection relevant issues."

    return "Detected: " + ", ".join(parts) + "."


def normalize_yolo_output(image_id: str, raw_detections: list, model_name: str = "yolo") -> ImageResult:
    counts = empty_counts()
    detections = []

    labels_present = set()

    for det in raw_detections:
        label = normalize_label(det.get("label", ""))

        if label not in CATEGORIES:
            continue

        if label == "no_damage":
            continue

        labels_present.add(label)

        detections.append(
            DetectionBox(
                label=label,
                confidence=float(det.get("confidence", 0.0)),
                bbox=det.get("bbox")
            )
        )

    for label in labels_present:
        counts[label] = 1

    if not labels_present:
        counts["no_damage"] = 1
    else:
        counts["no_damage"] = 0

    categories_present = [k for k, v in counts.items() if v > 0]

    return ImageResult(
        image_id=image_id,
        model_name=model_name,
        categories_present=categories_present,
        category_counts=counts,
        detections=detections,
        summary=build_image_summary(counts)
    )


def normalize_llava_output(image_id: str, parsed_json: dict, model_name: str = "llava") -> ImageResult:
    counts = empty_counts()

    incoming_counts = parsed_json.get("category_counts", {})

    for cat in CATEGORIES:
        counts[cat] = int(incoming_counts.get(cat, 0))

    has_defect = any(counts.get(cat, 0) > 0 for cat in DEFECT_CATEGORIES)

    if has_defect:
        counts["no_damage"] = 0
    else:
        counts["no_damage"] = 1

    categories_present = [k for k, v in counts.items() if v > 0]

    return ImageResult(
        image_id=image_id,
        model_name=model_name,
        categories_present=categories_present,
        category_counts=counts,
        detections=[],
        summary=parsed_json.get("summary", build_image_summary(counts))
    )