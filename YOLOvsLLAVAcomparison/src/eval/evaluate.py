"""
Unified evaluation script: compares YOLO and LLaVA on the test split.

Outputs:
  results/eval_yolo_results.csv
  results/eval_llava_results.csv
  results/eval_comparison.csv        ← side-by-side per-image diff
  results/eval_summary.txt           ← readable metrics summary

Usage:
    python -m src.eval.evaluate
    python -m src.eval.evaluate --model yolo    # run only YOLO
    python -m src.eval.evaluate --model llava   # run only LLaVA
    python -m src.eval.evaluate --limit 20      # quick smoke-test
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    CLASS_MAP,
    EVAL_LABELS,
    IMG_DIR_TEST,
    LBL_DIR_TEST,
    RESULTS_DIR,
    YOLO_MODEL_PATH,
    CONF_THRESHOLD,
)

logger = logging.getLogger(__name__)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ground-truth
# ---------------------------------------------------------------------------

def get_true_label(label_path: Path) -> str:
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
# Per-model runners
# ---------------------------------------------------------------------------

def eval_yolo(image_paths: list[Path]) -> pd.DataFrame:
    from ultralytics import YOLO
    from src.pipelines.yolo_pipeline import map_label, summarize_detections

    model = YOLO(str(YOLO_MODEL_PATH))
    rows = []

    for image_path in image_paths:
        label_path = LBL_DIR_TEST / f"{image_path.stem}.txt"
        true_label = get_true_label(label_path)

        try:
            result = model(str(image_path), verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue
                    raw = model.names[int(box.cls[0])]
                    detections.append({"label": map_label(raw), "confidence": conf})

            summary = summarize_detections(detections)
            cats = [c for c in summary["categories_present"] if c != "no_damage"]
            pred_label = cats[0] if cats else "no_damage"
        except Exception as exc:
            logger.error("YOLO failed on %s: %s", image_path.name, exc)
            pred_label = "no_damage"

        rows.append({"image": image_path.name, "true": true_label, "pred": pred_label})

    return pd.DataFrame(rows)


def eval_llava(image_paths: list[Path]) -> pd.DataFrame:
    from src.pipelines.llava_pipeline import run_llava_on_image

    rows = []

    for image_path in image_paths:
        label_path = LBL_DIR_TEST / f"{image_path.stem}.txt"
        true_label = get_true_label(label_path)

        try:
            _, parsed = run_llava_on_image(image_path)
            counts = parsed["category_counts"]
            active = {k: v for k, v in counts.items() if k != "no_damage" and v > 0}
            pred_label = max(active, key=active.get) if active else "no_damage"
        except Exception as exc:
            logger.error("LLaVA failed on %s: %s", image_path.name, exc)
            pred_label = "no_damage"

        rows.append({"image": image_path.name, "true": true_label, "pred": pred_label})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics printer
# ---------------------------------------------------------------------------

def print_metrics(df: pd.DataFrame, model_name: str) -> str:
    y_true, y_pred = df["true"].tolist(), df["pred"].tolist()
    acc = (df["true"] == df["pred"]).mean()

    lines = [
        f"\n{'=' * 60}",
        f"Model: {model_name.upper()}",
        f"Images evaluated : {len(df)}",
        f"Overall accuracy : {acc:.3f}",
        "",
        classification_report(y_true, y_pred, labels=EVAL_LABELS, zero_division=0),
        "Confusion matrix  (rows = true, cols = pred)",
    ]

    cm = confusion_matrix(y_true, y_pred, labels=EVAL_LABELS)
    header = f"{'':12s}" + "".join(f"{lbl:>12s}" for lbl in EVAL_LABELS)
    lines.append(header)
    for lbl, row in zip(EVAL_LABELS, cm):
        lines.append(f"{lbl:12s}" + "".join(f"{v:12d}" for v in row))

    block = "\n".join(lines)
    print(block)
    return block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(model: str = "both", limit: int | None = None) -> None:
    image_paths = sorted(
        p for p in IMG_DIR_TEST.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit:
        image_paths = image_paths[:limit]

    logger.info("Test images: %d", len(image_paths))

    summary_blocks = []

    if model in ("yolo", "both"):
        logger.info("Running YOLO evaluation …")
        df_yolo = eval_yolo(image_paths)
        df_yolo.to_csv(RESULTS_DIR / "eval_yolo_results.csv", index=False)
        summary_blocks.append(print_metrics(df_yolo, "yolo"))

    if model in ("llava", "both"):
        logger.info("Running LLaVA evaluation …")
        df_llava = eval_llava(image_paths)
        df_llava.to_csv(RESULTS_DIR / "eval_llava_results.csv", index=False)
        summary_blocks.append(print_metrics(df_llava, "llava"))

    if model == "both":
        df_cmp = df_yolo.rename(columns={"pred": "pred_yolo"}).merge(
            df_llava.rename(columns={"pred": "pred_llava"})[["image", "pred_llava"]],
            on="image",
        )
        df_cmp["agreement"] = df_cmp["pred_yolo"] == df_cmp["pred_llava"]
        df_cmp.to_csv(RESULTS_DIR / "eval_comparison.csv", index=False)
        agree_rate = df_cmp["agreement"].mean()
        block = f"\nModel agreement: {agree_rate:.1%} of images"
        print(block)
        summary_blocks.append(block)

    summary_path = RESULTS_DIR / "eval_summary.txt"
    summary_path.write_text("\n".join(summary_blocks))
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="YOLO vs LLaVA evaluation")
    parser.add_argument("--model", choices=["yolo", "llava", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate(model=args.model, limit=args.limit)
