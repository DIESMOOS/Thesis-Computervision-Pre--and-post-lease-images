"""
Central configuration for the YOLO-vs-LLaVA inspection pipeline.

All paths, constants, and model settings live here.
Import from this module instead of repeating magic strings everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: src/config.py → project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
CATEGORIES: list[str] = ["damage", "wear", "alteration", "no_damage"]

# Evaluation uses only the three meaningful labels (no_damage is the null case)
EVAL_LABELS: list[str] = ["damage", "wear", "no_damage"]

# Map YOLO raw class names → thesis categories
CATEGORY_MAP: dict[str, str] = {
    # damage
    "crack":    "damage",
    "damage":   "damage",
    "broken":   "damage",
    "hole":     "damage",
    "mould":    "damage",
    "mold":     "damage",
    "rot":      "damage",
    # wear
    "rust":     "wear",
    "paint":    "wear",
    "peeling":  "wear",
    "stain":    "wear",
    "discolor": "wear",
    # alteration
    "addition": "alteration",
    "modified": "alteration",
}

# Map YOLO dataset class IDs → thesis categories
# Adjust these to match your data.yaml
CLASS_MAP: dict[int, str] = {
    0: "damage",   # crack
    1: "damage",   # broken
    2: "damage",   # hole
    3: "wear",     # peeling_paint
    4: "damage",   # mould
    5: "no_damage",
}

# ---------------------------------------------------------------------------
# YOLO settings
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH: Path = ROOT / "models" / "best.pt"
CONF_THRESHOLD: float = 0.25
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# ---------------------------------------------------------------------------
# LLaVA settings
# ---------------------------------------------------------------------------
LLAVA_MODEL_ID: str = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_MAX_NEW_TOKENS: int = 150   # enough for JSON + summary

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATA_ROOT: Path   = ROOT / "data"
IMG_DIR_TEST: Path = DATA_ROOT / "inspection_dataset" / "images" / "test"
LBL_DIR_TEST: Path = DATA_ROOT / "inspection_dataset" / "labels" / "test"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR: Path = ROOT / "results"
