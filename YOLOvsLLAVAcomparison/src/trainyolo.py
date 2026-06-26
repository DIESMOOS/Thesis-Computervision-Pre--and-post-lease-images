import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from ultralytics import YOLO
import torch

# =========================
# CONFIG
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT_DIR / "data" / "inspection_dataset" / "data.yaml"

# Final model used in the thesis
BASE_MODEL = "yolov8m.pt"

PROJECT_DIR = ROOT_DIR / "models"
RUN_NAME = "housing_yolo"

# Training settings (reported in the thesis)
EPOCHS = 75
IMG_SIZE = 1024
BATCH_SIZE = 64
SEED = 42
PATIENCE = 20
WORKERS = 18


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    print(f"Using dataset: {DATA_YAML}")
    print(f"Using base model: {BASE_MODEL}")
    print("CUDA available:", torch.cuda.is_available())

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        patience=PATIENCE,
        plots=True,
        val=True,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=WORKERS,
        cache="disk",
    )

    best_model = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"

    print("\nTraining finished.")
    print(f"Best model saved at: {best_model}")


if __name__ == "__main__":
    main()