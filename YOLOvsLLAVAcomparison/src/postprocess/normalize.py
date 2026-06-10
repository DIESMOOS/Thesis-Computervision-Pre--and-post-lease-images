"""
YOLO pipeline for housing damage/wear detection.

Loads a custom-trained YOLOv8 model (models/best.pt) and runs inference
over a folder of images. Returns structured detection results that are
ready for normalize.py → aggregate_property.py → compare.py.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path
from ultralytics import YOLO

from src.config import (
    YOLO_MODEL_PATH,
    CONF_THRESHOLD,
    IMAGE_EXTS,
    CATEGORIES,
    CATEGORY_MAP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_paths(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        logger.warning("No images found in %s", folder)
    return paths


def map_label(raw_label: str) -> str:
    """Map a raw YOLO class name to one of the thesis categories."""
    key = raw_label.lower().strip()
    # Exact match first
    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]
    # Substring match second
    for keyword, thesis_label in CATEGORY_MAP.items():
        if keyword in key:
            return thesis_label
    # Unknown classes that slip through a custom model are most likely damage
    logger.debug("Unknown label '%s' — defaulting to 'damage'", raw_label)
    return "damage"


def empty_category_counts() -> dict[str, int]:
    return {cat: 0 for cat in CATEGORIES}


def summarize_detections(detections: list[dict]) -> dict:
    counts = empty_category_counts()

    if not detections:
        counts["no_damage"] = 1
        return {
            "categories_present": ["no_damage"],
            "category_counts": counts,
            "summary": "No detections above confidence threshold.",
        }

    for det in detections:
        label = det["label"]
        if label in counts:
            counts[label] += 1

    categories_present = [cat for cat in CATEGORIES if counts[cat] > 0 and cat != "no_damage"]

    return {
        "categories_present": categories_present,
        "category_counts": counts,
        "summary": f"Detected: {', '.join(categories_present)}." if categories_present else "No issues.",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_yolo_on_folder(
    folder_path: str | Path,
    model_path: str | Path = YOLO_MODEL_PATH,
    conf_threshold: float = CONF_THRESHOLD,
) -> list[dict]:
    """
    Run YOLOv8 inference on every image in *folder_path*.

    Args:
        folder_path:    Directory containing inspection images.
        model_path:     Path to the .pt weights file.
        conf_threshold: Minimum confidence to accept a detection.

    Returns:
        List of dicts, one per image:
        {
            "image_id":     str,
            "model_name":   "yolo",
            "detections":   [{"raw_label", "label", "confidence", "bbox"}, ...],
            "parsed_output": {"categories_present", "category_counts", "summary"}
        }
    """
    image_paths = get_image_paths(folder_path)
    model = YOLO(str(model_path))
    logger.info("Loaded YOLO model from %s", model_path)

    results = []

    for image_path in image_paths:
        try:
            result = model(str(image_path), verbose=False)[0]
        except Exception as exc:
            logger.error("YOLO inference failed on %s: %s", image_path.name, exc)
            results.append({
                "image_id": image_path.name,
                "model_name": "yolo",
                "detections": [],
                "parsed_output": summarize_detections([]),
                "error": str(exc),
            })
            continue

        detections: list[dict] = []

        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < conf_threshold:
                    continue

                class_id = int(box.cls[0])
                raw_label = model.names[class_id]
                mapped_label = map_label(raw_label)
                bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]

                detections.append({
                    "raw_label": raw_label,
                    "label": mapped_label,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                })

        results.append({
            "image_id": image_path.name,
            "model_name": "yolo",
            "detections": detections,
            "parsed_output": summarize_detections(detections),
        })

        logger.debug("%s → %d detection(s)", image_path.name, len(detections))

    logger.info("YOLO processed %d images in %s", len(results), folder_path)
    return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/properties/001/post_lease"
    output = run_yolo_on_folder(folder)

    print(f"\nProcessed {len(output)} images\n")
    for item in output[:3]:
        print(json.dumps(item, indent=2))
