import json
import sys
from pathlib import Path

from src.pipelines.yolo_pipeline import run_yolo_on_folder
from src.pipelines.llava_pipeline import run_llava_on_folder as run_llava_rule_guided_on_folder
from src.pipelines.llava_pipeline_zeroshot import run_llava_on_folder as run_llava_zeroshot_on_folder

from src.postprocess.normalize import normalize_yolo_output, normalize_llava_output
from src.postprocess.aggregate_property import aggregate_property
from src.postprocess.compare import compare_old_new


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

def run_yolo(property_id: str, new_images_dir: Path, old_report: dict):
    yolo_raw = run_yolo_on_folder(str(new_images_dir))

    yolo_images = [
        normalize_yolo_output(item["image_id"], item["detections"])
        for item in yolo_raw
    ]

    yolo_property = aggregate_property(property_id, "yolo", yolo_images)
    yolo_compare = compare_old_new(old_report, yolo_property)

    return {
        "property_result": yolo_property.model_dump(),
        "comparison": yolo_compare.model_dump(),
    }


def run_llava(
    property_id: str,
    new_images_dir: Path,
    old_report: dict,
    model_name: str,
    runner,
):
    llava_raw = runner(str(new_images_dir))

    llava_images = [
        normalize_llava_output(item["image_id"], item["parsed_output"])
        for item in llava_raw
    ]

    llava_property = aggregate_property(property_id, model_name, llava_images)
    llava_compare = compare_old_new(old_report, llava_property)

    return {
        "property_result": llava_property.model_dump(),
        "comparison": llava_compare.model_dump(),
    }


def main(property_id: str):
    property_dir = DATA_DIR / "properties" / property_id
    new_images_dir = property_dir / "post_lease"
    old_report_path = property_dir / "old_report.json"

    if not property_dir.exists():
        raise FileNotFoundError(f"Property folder not found: {property_dir}")

    if not old_report_path.exists():
        raise FileNotFoundError(f"Old report not found: {old_report_path}")

    if not new_images_dir.exists():
        raise FileNotFoundError(f"post_lease folder not found: {new_images_dir}")

    with open(old_report_path, "r", encoding="utf-8") as f:
        old_report = json.load(f)

    print(f"Running property {property_id}")

    print("Running YOLOv8...")
    yolo_result = run_yolo(property_id, new_images_dir, old_report)

    print("Running LLaVA zero shot...")
    llava_zeroshot_result = run_llava(
        property_id=property_id,
        new_images_dir=new_images_dir,
        old_report=old_report,
        model_name="llava_zeroshot",
        runner=run_llava_zeroshot_on_folder,
    )

    print("Running LLaVA rule guided...")
    llava_rule_guided_result = run_llava(
        property_id=property_id,
        new_images_dir=new_images_dir,
        old_report=old_report,
        model_name="llava_rule_guided",
        runner=run_llava_rule_guided_on_folder,
    )

    output = {
        "property_id": property_id,
        "yolo": yolo_result,
        "llava_zeroshot": llava_zeroshot_result,
        "llava_rule_guided": llava_rule_guided_result,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{property_id}_result.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {out_path}")


def get_property_id():
    if len(sys.argv) > 1:
        return sys.argv[1]

    available = sorted(
        [p.name for p in (DATA_DIR / "properties").iterdir() if p.is_dir()]
    )
    print("Available properties:", ", ".join(available))

    return input("Enter property ID to analyze: ").strip()


if __name__ == "__main__":
    pid = get_property_id()
    main(pid)