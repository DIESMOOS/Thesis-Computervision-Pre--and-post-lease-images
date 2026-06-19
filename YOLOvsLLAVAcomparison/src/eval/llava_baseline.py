"""
LLaVA-1.6 evaluation script.

Evaluates run_llava_on_images() (true batched inference) against the YOLO
test split. Labels: damage, crack, mold, wear, asbestos, no_damage.

Outputs (models/llava_runs/<run_name>/)
-----------------------------------------
  predictions.csv
  summary.csv                     metric table with 95 % bootstrap CIs
  confusion_matrix_multiclass.csv
  confusion_matrix_binary.csv
  classification_report_multiclass.txt
  classification_report_binary.txt
  raw_outputs/<stem>.json         full raw + parsed output per image
  run_config.json

Usage (Snellius)
-----------------
  python -m src.eval.llava_baseline
  python -m src.eval.llava_baseline --limit 50
  python -m src.eval.llava_baseline --batch-size 128 --name run_v3
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CLASS_MAP,
    EVAL_LABELS,
    IMAGE_EXTS,
    IMG_DIR_TEST,
    LBL_DIR_TEST,
    LLAVA_MAX_NEW_TOKENS,
    LLAVA_MODEL_ID,
    RESULTS_DIR,
)
from src.pipelines.llava_pipeline import run_llava_on_images

ROOT_DIR = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

BINARY_LABELS = ["damage", "not_damage"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_dir(run_name: str | None = None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("llava_%Y%m%d_%H%M%S")
    run_dir = ROOT_DIR / "models" / "llava_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_outputs").mkdir(exist_ok=True)
    return run_dir


def get_true_label(label_path: Path) -> str:
    """
    Read a YOLO-format label file and return the dominant category.
    Priority: crack > mold > asbestos > wear > damage > no_damage.
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "no_damage"
    found: list[str] = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls_id = int(line.split()[0])
        lbl = CLASS_MAP.get(cls_id)
        if lbl:
            found.append(lbl)
    for priority in ["crack", "mold", "asbestos", "wear", "damage"]:
        if priority in found:
            return priority
    return "no_damage"


def choose_pred_label(parsed: dict) -> str:
    cats = parsed.get("categories_present", [])
    if cats and cats[0] in EVAL_LABELS:
        return cats[0]
    counts = parsed.get("category_counts", {})
    for lbl in EVAL_LABELS:
        if counts.get(lbl, 0) > 0:
            return lbl
    return "no_damage"


def to_binary(label: str) -> str:
    return "not_damage" if label == "no_damage" else "damage"


def bootstrap_ci(
    y_true: list, y_pred: list, metric_fn,
    n_bootstrap: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    scores = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            continue
    if not scores:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(scores)),
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )


def print_confusion_matrix(y_true: list, y_pred: list, labels: list[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    w = 14
    print(f"{'':>{w}}" + "".join(f"{lbl:>{w}}" for lbl in labels))
    for lbl, row in zip(labels, cm):
        print(f"{lbl:>{w}}" + "".join(f"{v:>{w}d}" for v in row))


def save_confusion_matrix(
    y_true: list, y_pred: list, labels: list[str], out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    limit: int | None = None,
    bootstrap: int = 1000,
    run_name: str | None = None,
    batch_size: int = 128,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(run_name)
    raw_dir = run_dir / "raw_outputs"
    out_csv = run_dir / "predictions.csv"
    out_summary = run_dir / "summary.csv"

    image_paths = sorted(
        p for p in IMG_DIR_TEST.glob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    logger.info("Run folder     : %s", run_dir)
    logger.info("Images to eval : %d", len(image_paths))
    logger.info("Batch size     : %d", batch_size)
    logger.info("Model          : %s", LLAVA_MODEL_ID)

    # -----------------------------------------------------------------------
    # Inference — all images, in batches
    # -----------------------------------------------------------------------
    all_outputs = run_llava_on_images(image_paths, batch_size=batch_size)

    # -----------------------------------------------------------------------
    # Build rows + save raw outputs
    # -----------------------------------------------------------------------
    rows: list[dict] = []

    for i, (image_path, (raw_output, parsed)) in enumerate(
        zip(image_paths, all_outputs), 1
    ):
        label_path = LBL_DIR_TEST / f"{image_path.stem}.txt"
        true_label = get_true_label(label_path)
        pred_label = choose_pred_label(parsed)

        raw_file = raw_dir / f"{image_path.stem}.json"
        raw_file.write_text(json.dumps({
            "image":       image_path.name,
            "true":        true_label,
            "pred":        pred_label,
            "true_binary": to_binary(true_label),
            "pred_binary": to_binary(pred_label),
            "raw_llava":   raw_output,
            "parsed":      parsed,
        }, indent=2))

        rows.append({
            "image":           image_path.name,
            "true":            true_label,
            "pred":            pred_label,
            "true_binary":     to_binary(true_label),
            "pred_binary":     to_binary(pred_label),
            "raw_output_file": str(raw_file.relative_to(run_dir)),
            "raw_llava":       raw_output[:500],
        })

        correct = "✓" if pred_label == true_label else "✗"
        logger.info(
            "[%d/%d] %-60s true=%-10s pred=%-10s %s",
            i, len(image_paths), image_path.name,
            true_label, pred_label, correct,
        )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    y_true     = df["true"].tolist()
    y_pred     = df["pred"].tolist()
    y_true_bin = df["true_binary"].tolist()
    y_pred_bin = df["pred_binary"].tolist()

    multiclass_metrics = {
        "accuracy":          lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "macro_f1":          lambda yt, yp: f1_score(yt, yp, labels=EVAL_LABELS, average="macro", zero_division=0),
        "macro_precision":   lambda yt, yp: precision_score(yt, yp, labels=EVAL_LABELS, average="macro", zero_division=0),
        "macro_recall":      lambda yt, yp: recall_score(yt, yp, labels=EVAL_LABELS, average="macro", zero_division=0),
    }

    binary_metrics = {
        "accuracy":                  lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy":         lambda yt, yp: balanced_accuracy_score(yt, yp),
        "f1_damage":                 lambda yt, yp: f1_score(yt, yp, pos_label="damage", average="binary", zero_division=0),
        "precision_damage":          lambda yt, yp: precision_score(yt, yp, pos_label="damage", zero_division=0),
        "recall_damage_sensitivity": lambda yt, yp: recall_score(yt, yp, pos_label="damage", zero_division=0),
    }

    summary_rows: list[dict] = []

    for name, fn in multiclass_metrics.items():
        point = fn(y_true, y_pred)
        mean, low, high = bootstrap_ci(y_true, y_pred, fn, bootstrap)
        summary_rows.append({"task": "multiclass", "metric": name,
                              "point_estimate": round(point, 4),
                              "bootstrap_mean": round(mean, 4),
                              "ci_95_low": round(low, 4),
                              "ci_95_high": round(high, 4)})

    for name, fn in binary_metrics.items():
        point = fn(y_true_bin, y_pred_bin)
        mean, low, high = bootstrap_ci(y_true_bin, y_pred_bin, fn, bootstrap)
        summary_rows.append({"task": "binary", "metric": name,
                              "point_estimate": round(point, 4),
                              "bootstrap_mean": round(mean, 4),
                              "ci_95_low": round(low, 4),
                              "ci_95_high": round(high, 4)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_summary, index=False)

    mc_report  = classification_report(y_true, y_pred, labels=EVAL_LABELS, zero_division=0)
    bin_report = classification_report(y_true_bin, y_pred_bin, labels=BINARY_LABELS, zero_division=0)

    (run_dir / "classification_report_multiclass.txt").write_text(mc_report)
    (run_dir / "classification_report_binary.txt").write_text(bin_report)
    save_confusion_matrix(y_true, y_pred, EVAL_LABELS,
                          run_dir / "confusion_matrix_multiclass.csv")
    save_confusion_matrix(y_true_bin, y_pred_bin, BINARY_LABELS,
                          run_dir / "confusion_matrix_binary.csv")

    run_config = {
        "run_name":         run_dir.name,
        "model":            LLAVA_MODEL_ID,
        "max_new_tokens":   LLAVA_MAX_NEW_TOKENS,
        "batch_size":       batch_size,
        "images_evaluated": len(df),
        "limit":            limit,
        "bootstrap":        bootstrap,
        "image_dir":        str(IMG_DIR_TEST),
        "label_dir":        str(LBL_DIR_TEST),
        "eval_labels":      EVAL_LABELS,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    df.to_csv(RESULTS_DIR / "llava_latest_predictions.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "llava_latest_summary.csv", index=False)

    # -----------------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"LLaVA-1.6  |  {LLAVA_MODEL_ID}")
    print(sep)
    print(f"Images evaluated : {len(df)}")
    print(f"Batch size used  : {batch_size}")
    print(f"Run folder       : {run_dir}")
    print()

    print("Ground-truth distribution:")
    print(df["true"].value_counts().reindex(EVAL_LABELS, fill_value=0).to_string())
    print()
    print("Prediction distribution:")
    print(df["pred"].value_counts().reindex(EVAL_LABELS, fill_value=0).to_string())
    print()

    print("Multiclass classification report:")
    print(mc_report)
    print("Multiclass confusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(y_true, y_pred, EVAL_LABELS)
    print()

    print("Binary (damage vs not_damage) classification report:")
    print(bin_report)
    print("Binary confusion matrix:")
    print_confusion_matrix(y_true_bin, y_pred_bin, BINARY_LABELS)
    print()

    print("Summary with 95 % bootstrap CIs:")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved to: {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate LLaVA-1.6 on housing inspection images"
    )
    parser.add_argument("--limit",      type=int, default=None,
                        help="Evaluate only the first N images")
    parser.add_argument("--bootstrap",  type=int, default=1000,
                        help="Bootstrap resamples for CIs (default: 1000)")
    parser.add_argument("--name",       type=str, default=None,
                        help="Run folder name (default: llava_YYYYMMDD_HHMMSS)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Images per model.generate() call (default: 128)")
    args = parser.parse_args()

    evaluate(
        limit=args.limit,
        bootstrap=args.bootstrap,
        run_name=args.name,
        batch_size=args.batch_size,
    )
