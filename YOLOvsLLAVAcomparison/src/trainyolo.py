from pathlib import Path
from ultralytics import YOLO
import torch

# =========================
# CONFIG
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT_DIR / "data" / "inspection_dataset" / "data.yaml"

BASE_MODEL = "yolov8m.pt"   # better than yolov8n for thesis-quality results
PROJECT_DIR = ROOT_DIR / "models"
RUN_NAME = "75epocs"

EPOCHS = 75
IMG_SIZE = 1024
BATCH_SIZE = 64
SEED = 42
WORKERS = 18
cache = "disk"
patience=20

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Start a Snellius GPU session first.")

    print(f"Using dataset: {DATA_YAML}")
    print(f"Using base model: {BASE_MODEL}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        workers=WORKERS,
        cache=cache,
        seed=SEED,
        pretrained=True,
        patience=patience,
        plots=True,
        val=True,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,
    )

    best_model = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    results_csv = PROJECT_DIR / RUN_NAME / "results.csv"

    print("\nTraining finished.")
    print(f"Best model saved at: {best_model}")
    print(f"Results CSV saved at: {results_csv}")

    return results


if __name__ == "__main__":
    main()