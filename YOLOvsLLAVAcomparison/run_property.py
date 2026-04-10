import json
from pathlib import Path

from src.pipelines.yolo_pipeline import run_yolo_on_folder
from src.pipelines.llava_pipeline import run_llava_on_folder
from src.postprocess.normalize import normalize_yolo_output, normalize_llava_output
from src.postprocess.aggregate_property import aggregate_property
from src.postprocess.compare import compare_old_new


def main(property_id: str):
    property_dir = Path("data/properties") / property_id
    new_images_dir = property_dir / "post_lease"
    old_report_path = property_dir / "old_report.json"

    with open(old_report_path, "r", encoding="utf-8") as f:
        old_report = json.load(f)

    # YOLO
    yolo_raw = run_yolo_on_folder(str(new_images_dir))
    yolo_images = [
        normalize_yolo_output(item["image_id"], item["detections"])
        for item in yolo_raw
    ]
    yolo_property = aggregate_property(property_id, "yolo", yolo_images)
    yolo_compare = compare_old_new(old_report, yolo_property)

    # LLaVA
    llava_raw = run_llava_on_folder(str(new_images_dir))
    llava_images = [
        normalize_llava_output(item["image_id"], item["parsed_output"])
        for item in llava_raw
    ]
    llava_property = aggregate_property(property_id, "llava", llava_images)
    llava_compare = compare_old_new(old_report, llava_property)

    output = {
        "property_id": property_id,
        "yolo": {
            "property_result": yolo_property.model_dump(),
            "comparison": yolo_compare.model_dump()
        },
        "llava": {
            "property_result": llava_property.model_dump(),
            "comparison": llava_compare.model_dump()
        }
    }

    out_path = Path("results") / f"{property_id}_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main("001")