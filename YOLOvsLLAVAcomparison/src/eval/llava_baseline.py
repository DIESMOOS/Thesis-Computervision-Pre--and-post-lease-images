"""
LLaVA-1.6 evaluation script.

Uses src/pipelines/llava_pipeline.py for inference and evaluates predictions
against the YOLO test split using the YOLO-aligned labels:
damage, crack, mold, wear, asbestos, no_damage.

Outputs are saved in:
models/llava_runs/<run_name>/
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
    IMG_DIR_TEST,
    LBL_DIR_TEST,
    RESULTS_DIR,
    CLASS_MAP,
    EVAL_LABELS,
    IMAGE_EXTS,
    LLAVA_MODEL_ID,
)

from src.pipelines.llava_pipeline import run_llava_on_images

ROOT_DIR = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def make_run_dir(run_name: str | None = None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("llava_%Y%m%d_%H%M%S")

    run_dir = ROOT_DIR / "models" / "llava_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_outputs").mkdir(parents=True, exist_ok=True)
    return run_dir


def get_true_label(label_path: Path) -> str:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "no_damage"

    labels = []

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        cls_id = int(line.split()[0])
        labels.append(CLASS_MAP.get(cls_id, "damage"))

    for label in ["crack", "mold", "asbestos", "wear", "damage"]:
        if label in labels:
            return label

    return "no_damage"


def choose_pred_label(parsed: dict, raw_output: str) -> str:
    cats = parsed.get("categories_present", [])

    if cats:
        label = cats[0]
        if label in EVAL_LABELS:
            return label

    counts = parsed.get("category_counts", {})

    for label in EVAL_LABELS:
        if counts.get(label, 0) > 0:
            return label

    return "no_damage"


def to_binary(label: str) -> str:
    return "not_damage" if label == "no_damage" else "damage"


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap: int = 1000, seed: int = 42):
    rng = random.Random(seed)
    n = len(y_true)

    if n == 0:
        return np.nan, np.nan, np.nan

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
        return np.nan, np.nan, np.nan

    return (
        float(np.mean(scores)),
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )


def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = f"{'':14s}" + "".join(f"{label:>14s}" for label in labels)
    print(header)

    for label, row in zip(labels, cm):
        print(f"{label:14s}" + "".join(f"{value:14d}" for value in row))


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_path)


def evaluate(
    limit: int | None = None,
    bootstrap: int = 1000,
    run_name: str | None = None,
    batch_size: int = 128,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(run_name)

    out_csv = run_dir / "predictions.csv"
    out_summary = run_dir / "summary.csv"
    raw_dir = run_dir / "raw_outputs"

    image_paths = sorted(
        p for p in IMG_DIR_TEST.glob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )

    if limit is not None:
        image_paths = image_paths[:limit]

    logger.info("Evaluating %d images", len(image_paths))
    logger.info("Run directory: %s", run_dir)
    logger.info("Batch size: %d", batch_size)

    rows = []

    for start_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start_idx:start_idx + batch_size]

        try:
            batch_outputs = run_llava_on_images(batch_paths)
        except Exception as exc:
            logger.error("Batch failed: %s", exc)
            batch_outputs = [("", {}) for _ in batch_paths]

        for offset, image_path in enumerate(batch_paths):
            i = start_idx + offset + 1

            label_path = LBL_DIR_TEST / f"{image_path.stem}.txt"
            true_label = get_true_label(label_path)

            raw_output, parsed = batch_outputs[offset]
            pred_label = choose_pred_label(parsed, raw_output)
            print(f"Using batch size: {batch_size}")

            raw_file = raw_dir / f"{image_path.stem}.json"
            raw_file.write_text(
                json.dumps(
                    {
                        "image": image_path.name,
                        "true": true_label,
                        "pred": pred_label,
                        "true_binary": to_binary(true_label),
                        "pred_binary": to_binary(pred_label),
                        "raw_llava": raw_output,
                        "parsed": parsed,
                    },
                    indent=2,
                )
            )

            rows.append(
                {
                    "image": image_path.name,
                    "true": true_label,
                    "pred": pred_label,
                    "true_binary": to_binary(true_label),
                    "pred_binary": to_binary(pred_label),
                    "raw_output_file": str(raw_file.relative_to(run_dir)),
                    "raw_llava": raw_output[:500],
                }
            )

            if i % 10 == 0:
                logger.info("%d / %d done", i, len(image_paths))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()
    y_true_bin = df["true_binary"].tolist()
    y_pred_bin = df["pred_binary"].tolist()

    summary_rows = []

    multiclass_metrics = {
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "macro_f1": lambda yt, yp: f1_score(
            yt, yp, labels=EVAL_LABELS, average="macro", zero_division=0
        ),
        "macro_precision": lambda yt, yp: precision_score(
            yt, yp, labels=EVAL_LABELS, average="macro", zero_division=0
        ),
        "macro_recall": lambda yt, yp: recall_score(
            yt, yp, labels=EVAL_LABELS, average="macro", zero_division=0
        ),
    }

    for name, fn in multiclass_metrics.items():
        point = fn(y_true, y_pred)
        mean, low, high = bootstrap_ci(y_true, y_pred, fn, bootstrap)
        summary_rows.append(
            {
                "task": "multiclass",
                "metric": name,
                "point_estimate": point,
                "bootstrap_mean": mean,
                "ci_95_low": low,
                "ci_95_high": high,
            }
        )

    binary_labels = ["damage", "not_damage"]

    binary_metrics = {
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "f1_damage": lambda yt, yp: f1_score(
            yt, yp, pos_label="damage", average="binary", zero_division=0
        ),
        "precision_damage": lambda yt, yp: precision_score(
            yt, yp, pos_label="damage", zero_division=0
        ),
        "recall_damage_sensitivity": lambda yt, yp: recall_score(
            yt, yp, pos_label="damage", zero_division=0
        ),
    }

    for name, fn in binary_metrics.items():
        point = fn(y_true_bin, y_pred_bin)
        mean, low, high = bootstrap_ci(y_true_bin, y_pred_bin, fn, bootstrap)
        summary_rows.append(
            {
                "task": "binary_damage_vs_not_damage",
                "metric": name,
                "point_estimate": point,
                "bootstrap_mean": mean,
                "ci_95_low": low,
                "ci_95_high": high,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_summary, index=False)

    multiclass_report = classification_report(
        y_true,
        y_pred,
        labels=EVAL_LABELS,
        zero_division=0,
    )

    binary_report = classification_report(
        y_true_bin,
        y_pred_bin,
        labels=binary_labels,
        zero_division=0,
    )

    (run_dir / "classification_report_multiclass.txt").write_text(multiclass_report)
    (run_dir / "classification_report_binary.txt").write_text(binary_report)

    save_confusion_matrix(
        y_true,
        y_pred,
        EVAL_LABELS,
        run_dir / "confusion_matrix_multiclass.csv",
    )

    save_confusion_matrix(
        y_true_bin,
        y_pred_bin,
        binary_labels,
        run_dir / "confusion_matrix_binary.csv",
    )

    run_config = {
        "run_name": run_dir.name,
        "model": LLAVA_MODEL_ID,
        "images_evaluated": len(df),
        "limit": limit,
        "bootstrap": bootstrap,
        "batch_size": batch_size,
        "image_dir": str(IMG_DIR_TEST),
        "label_dir": str(LBL_DIR_TEST),
        "labels": EVAL_LABELS,
        "predictions_file": str(out_csv),
        "summary_file": str(out_summary),
        "medical_imaging_inspired_options": [
            "Fine-tune LLaVA on medical or housing inspection images",
            "Use retrieval-augmented prompting",
            "Use few-shot examples",
            "Use chain-of-thought-style structured reasoning",
            "Use domain-specific models instead of vanilla LLaVA",
        ],
    }

    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    df.to_csv(RESULTS_DIR / "llava_baseline_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "llava_medical_style_summary.csv", index=False)

    print("\n" + "=" * 70)
    print(f"LLaVA-1.6 evaluation | model: {LLAVA_MODEL_ID}")
    print("=" * 70)
    print(f"Images evaluated  : {len(df)}")
    print(f"Run folder        : {run_dir}")
    print(f"Predictions saved : {out_csv}")
    print(f"Summary saved     : {out_summary}")
    print()

    print("Ground-truth distribution:")
    print(df["true"].value_counts().reindex(EVAL_LABELS, fill_value=0))
    print()

    print("Prediction distribution:")
    print(df["pred"].value_counts().reindex(EVAL_LABELS, fill_value=0))
    print()

    print("Multiclass classification report:")
    print(multiclass_report)

    print("Multiclass confusion matrix:")
    print_confusion_matrix(y_true, y_pred, EVAL_LABELS)
    print()

    print("Binary damage vs not_damage classification report:")
    print(binary_report)

    print("Binary confusion matrix:")
    print_confusion_matrix(y_true_bin, y_pred_bin, binary_labels)
    print()

    print("Summary with 95% bootstrap CIs:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate LLaVA-1.6 on YOLO-aligned housing labels")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    evaluate(
        limit=args.limit,
        bootstrap=args.bootstrap,
        run_name=args.name,
        batch_size=args.batch_size,
    )