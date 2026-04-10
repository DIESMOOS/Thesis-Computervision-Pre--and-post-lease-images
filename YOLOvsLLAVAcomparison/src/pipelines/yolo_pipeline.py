from pathlib import Path


def run_yolo_on_folder(folder_path: str) -> list:
    """
    Return list of per-image raw detections.
    Example:
    [
        {
            "image_id": "img1.jpg",
            "detections": [
                {"label": "damage", "confidence": 0.88, "bbox": [10, 20, 40, 50]}
            ]
        }
    ]
    """
    folder = Path(folder_path)
    results = []

    for image_path in folder.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        # TODO: replace mock with real YOLO inference
        results.append({
            "image_id": image_path.name,
            "detections": []
        })

    return results