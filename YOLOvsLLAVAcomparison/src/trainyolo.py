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
BASE_MODEL = "yolov8n.pt"  # Ultralytics will download this automatically if missing

PROJECT_DIR = ROOT_DIR / "models"
RUN_NAME = "housing_yolo"

#EPOCHS = 75
IMG_SIZE = 640
#BATCH_SIZE = 16
SEED = 42

EPOCHS = 2
BATCH_SIZE = 4

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
        patience=10,
        plots=True,
        val=True,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        cache=True,
    )

    best_model = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"

    print("Training finished.")
    print(f"Best model saved at: {best_model}")


if __name__ == "__main__":
    main()