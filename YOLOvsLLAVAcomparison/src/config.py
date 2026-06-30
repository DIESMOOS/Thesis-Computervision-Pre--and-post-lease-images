from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROPERTIES_DIR = DATA_DIR / "properties"
RESULTS_DIR = BASE_DIR / "results"

CATEGORIES = ["damage", "wear", "alteration", "no_damage"]

YOLO_CONF_THRESHOLD = 0.25

LLAVA_PROMPT = """
Analyze this housing inspection image.

Return ONLY valid JSON.

Required JSON structure:
{
  "categories_present": [],
  "category_counts": {
    "damage": 0,
    "wear": 0,
    "alteration": 0,
    "no_damage": 0
  },
  "summary": ""
}

Allowed categories only:
- damage
- wear
- alteration
- no_damage

Rules:
- If damage, wear, or alteration is detected, no_damage must be 0.
- If no visible inspection-relevant issue exists, set no_damage to 1.
- category_counts must always contain all 4 categories.
- Do not include markdown.
- Do not include explanations outside JSON.
"""