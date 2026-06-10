"""
LLaVA-1.6 baseline evaluation script.

Runs the model over the test split and writes per-image predictions to CSV.
Prints accuracy, per-class precision/recall/F1, and a confusion matrix.

Usage (Snellius):
    python -m src.eval.llava_baseline
    python -m src.eval.llava_baseline --limit 50   # quick sanity check
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    LLAVA_MODEL_ID,
    LLAVA_MAX_NEW_TOKENS,
    IMG_DIR_TEST,
    LBL_DIR_TEST,
    CLASS_MAP,
    EVAL_LABELS,
    RESULTS_DIR,
)
from src.pipelines.llava_pipeline import run_llava_on_image

logger = logging.getLogger(__name__)

OUT_CSV = RESULTS_DIR / "llava_baseline_results.csv"


# ---------------------------------------------------------------------------
# Ground-truth helper
# ---------------------------------------------------------------------------

def get_true_label(label_path: Path) -> str:
    """
    Read a YOLO-format label file and return the dominant thesis category.
    Priority: damage > wear > no_damage.
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "no_damage"

    classes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            cls_id = int(line.split()[0])
            classes.append(CLASS_MAP.get(cls_id, "damage"))

    if "damage" in classes:
        return "damage"
    if "wear" in classes:
        return "wear"
    return "no_damage"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(limit: int | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMG_DIR_TEST.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if limit:
        image_paths = image_paths[:limit]

    logger.info("Evaluating %d images …", len(image_paths))

    rows: list[dict] = []

    for i, image_path in enumerate(image_paths, 1):
        label_path = LBL_DIR_TEST / f"{image_path.stem}.txt"
        true_label = get_true_label(label_path)

        try:
            raw_output, parsed = run_llava_on_image(image_path)
            # Determine predicted label: highest-count non-no_damage wins,
            # otherwise fall back to no_damage.
            counts = parsed["category_counts"]
            active = {cat: cnt for cat, cnt in counts.items() if cat != "no_damage" and cnt > 0}
            pred_label = max(active, key=active.get) if active else "no_damage"
        except Exception as exc:
            logger.error("Failed on %s: %s", image_path.name, exc)
            raw_output = ""
            pred_label = "no_damage"

        rows.append({
            "image": image_path.name,
            "true": true_label,
            "pred": pred_label,
            "raw_llava": raw_output[:300],          # truncate for readability
        })

        if i % 10 == 0:
            logger.info("  %d / %d done", i, len(image_paths))

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    logger.info("Results saved to %s", OUT_CSV)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()

    accuracy = (df["true"] == df["pred"]).mean()

    print("\n" + "=" * 60)
    print(f"LLaVA-1.6 Baseline  |  model: {LLAVA_MODEL_ID}")
    print("=" * 60)
    print(f"Images evaluated : {len(df)}")
    print(f"Overall accuracy : {accuracy:.3f}")
    print()

    print(classification_report(y_true, y_pred, labels=EVAL_LABELS, zero_division=0))

    print("Confusion matrix  (rows = true, cols = pred)")
    cm = confusion_matrix(y_true, y_pred, labels=EVAL_LABELS)
    header = f"{'':12s}" + "".join(f"{lbl:>12s}" for lbl in EVAL_LABELS)
    print(header)
    for lbl, row in zip(EVAL_LABELS, cm):
        print(f"{lbl:12s}" + "".join(f"{v:12d}" for v in row))

    print("\nFull results in:", OUT_CSV)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="LLaVA-1.6 baseline evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    args = parser.parse_args()

    evaluate(limit=args.limit)
