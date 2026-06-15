"""
Central configuration for the YOLO-vs-LLaVA inspection pipeline.

All paths, constants, and model settings live here.
Import from this module instead of repeating magic strings everywhere.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Taxonomy: aligned with YOLO data.yaml
# ---------------------------------------------------------------------------
CATEGORIES: list[str] = [
    "damage",
    "crack",
    "mold",
    "wear",
    "asbestos",
    "no_damage",
]

EVAL_LABELS: list[str] = [
    "damage",
    "crack",
    "mold",
    "wear",
    "asbestos",
    "no_damage",
]

CATEGORY_MAP: dict[str, str] = {
    "damage": "damage",
    "broken": "damage",
    "hole": "damage",
    "dent": "damage",
    "missing": "damage",
    "water_damage": "damage",
    "fire_damage": "damage",

    "crack": "crack",
    "cracked": "crack",
    "fracture": "crack",
    "split": "crack",

    "mold": "mold",
    "mould": "mold",
    "fungal": "mold",
    "mildew": "mold",

    "wear": "wear",
    "worn": "wear",
    "rust": "wear",
    "paint": "wear",
    "peeling": "wear",
    "stain": "wear",
    "discolor": "wear",
    "discoloration": "wear",
    "deterioration": "wear",

    "asbestos": "asbestos",
    "thick-dark-mark": "asbestos",
    "thick-light-mark": "asbestos",
    "thin-dark-mark": "asbestos",
    "thin-light-mark": "asbestos",

    "no_damage": "no_damage",
    "nodamage": "no_damage",
    "no damage": "no_damage",
}

CLASS_MAP: dict[int, str] = {
    0: "damage",
    1: "crack",
    2: "mold",
    3: "wear",
    4: "asbestos",
}

BINARY_MAP: dict[str, str] = {
    "damage": "damage",
    "crack": "damage",
    "mold": "damage",
    "wear": "damage",
    "asbestos": "damage",
    "no_damage": "not_damage",
}

BINARY_EVAL_LABELS: list[str] = ["damage", "not_damage"]

# ---------------------------------------------------------------------------
# YOLO settings
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH: Path = ROOT / "models" / "best_run" / "weights" / "best.pt"
CONF_THRESHOLD: float = 0.25
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# ---------------------------------------------------------------------------
# LLaVA settings
# ---------------------------------------------------------------------------
LLAVA_MODEL_ID: str = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_MAX_NEW_TOKENS: int = 150

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATA_ROOT: Path = ROOT / "data"
IMG_DIR_TEST: Path = DATA_ROOT / "inspection_dataset" / "images" / "test"
LBL_DIR_TEST: Path = DATA_ROOT / "inspection_dataset" / "labels" / "test"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR: Path = ROOT / "results"