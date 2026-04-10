from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROPERTIES_DIR = DATA_DIR / "properties"
RESULTS_DIR = BASE_DIR / "results"

CATEGORIES = ["damage", "wear", "alteration", "no_damage"]

YOLO_CONF_THRESHOLD = 0.25

LLAVA_PROMPT = """
Analyze this housing inspection image.
Return only valid JSON with:
- categories_present
- category_counts
- summary

Use only these categories:
damage, wear, alteration, no_damage

Rules:
- If any of damage, wear, or alteration is present, no_damage must be 0.
- If nothing relevant is visible, set no_damage to 1.
- category_counts must contain all 4 categories.
"""