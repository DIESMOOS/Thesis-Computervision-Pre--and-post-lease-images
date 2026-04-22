import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# TEMP TEST CONFIG
# ---------------------------------------------------------
# Later replace with your trained model path, for example:
# MODEL_PATH = "models/yolo/best.pt"
MODEL_PATH = "yolov8n.pt"

CONF_THRESHOLD = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------
# TEMP CATEGORY MAP
# ---------------------------------------------------------
# This is only for pipeline testing with a generic YOLO model.
# A generic coco model does not know your thesis classes.
# Later replace this with the actual class mapping from your trained model.
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


def map_label(label: str) -> str:
    """
    Maps raw YOLO class names to thesis categories.
    Later this should match your trained YOLO class names exactly.
    """
    label = label.lower().strip()

    for key, mapped_value in CATEGORY_MAP.items():
        if key in label:
            return mapped_value

    return DEFAULT_FALLBACK_LABEL


def run_yolo_on_folder(folder_path: str) -> list:
    """
    Runs YOLO inference on all images in a folder.

    Returns a list like:
    [
        {
            "image_id": "img1.jpg",
            "detections": [
                {
                    "label": "damage",
                    "confidence": 0.88,
                    "bbox": [10.0, 20.0, 40.0, 50.0]
                }
            ]
        }
    ]
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    model = YOLO(MODEL_PATH)
    results = []

    for image_path in sorted(folder.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        inference_result = model(str(image_path), verbose=False)
        r = inference_result[0]

        detections = []

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                conf = float(box.conf[0])

                if conf < CONF_THRESHOLD:
                    continue

                cls_id = int(box.cls[0])
                raw_label = model.names[cls_id]
                mapped_label = map_label(raw_label)

                xyxy = box.xyxy[0].tolist()
                xyxy = [round(float(x), 2) for x in xyxy]

                detections.append({
                    "label": mapped_label,
                    "confidence": round(conf, 4),
                    "bbox": xyxy
                })

        results.append({
            "image_id": image_path.name,
            "detections": detections
        })

    return results


if __name__ == "__main__":
    # Quick local test
    test_folder = "data/properties/001/post_lease"
    output = run_yolo_on_folder(test_folder)

    print(f"Processed {len(output)} images")
    for item in output[:3]:
        print(item)