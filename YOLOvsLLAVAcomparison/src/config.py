from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROPERTIES_DIR = DATA_DIR / "properties"
RESULTS_DIR = BASE_DIR / "results"

CATEGORIES = [
    "damage",
    "crack",
    "mold",
    "wear",
    "asbestos",
    "no_damage",
]

DEFECT_CATEGORIES = [
    "damage",
    "crack",
    "mold",
    "wear",
    "asbestos",
]

YOLO_CONF_THRESHOLD = 0.25

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_MAX_NEW_TOKENS = 120