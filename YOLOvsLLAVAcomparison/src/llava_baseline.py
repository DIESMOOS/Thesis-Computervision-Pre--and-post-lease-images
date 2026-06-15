"""
LLaVA-1.6 baseline evaluation script.

Runs the model over the test split and writes per-image predictions to CSV.
Prints accuracy, per-class precision/recall/F1, and a confusion matrix.

Usage (Snellius):
    python -m src.eval.llava_baseline
    python -m src.eval.llava_baseline --limit 50   # quick sanity check

Direct usage:
    .venv/bin/python src/llava_baseline.py
    .venv/bin/python src/llava_baseline.py --limit 50

Medical-imaging style evaluation:
- multi-class accuracy, precision, recall, F1
- macro F1
- balanced accuracy
- binary damage vs not_damage evaluation
- confusion matrices
- bootstrap 95% confidence intervals
"""

import argparse
import logging
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


DATASET_DIR = PROJECT_ROOT / "data" / "inspection_dataset"
IMG_DIR_TEST = DATASET_DIR / "images" / "test"
LBL_DIR_TEST = DATASET_DIR / "labels" / "test"

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_CSV = RESULTS_DIR / "llava_baseline_results.csv"
OUT_SUMMARY = RESULTS_DIR / "llava_medical_style_summary.csv"

LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_MAX_NEW_TOKENS = 64

# Same class structure as data.yaml:
# 0: damage
# 1: crack
# 2: mold
# 3: wear
# 4: asbestos
YOLO_TO_EVAL_CLASS = {
    0: "damage",
    1: "crack",
    2: "mold",
    3: "wear",
    4: "asbestos",
}

EVAL_LABELS = ["damage", "crack", "mold", "wear", "asbestos", "no_damage"]

logger = logging.getLogger(__name__)


def get_true_label(label_path: Path) -> str:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "no_damage"

    mapped = []

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        cls_id = int(line.split()[0])
        mapped.append(YOLO_TO_EVAL_CLASS.get(cls_id, "damage"))

    # If multiple objects exist, use priority order.
    for label in ["mold", "asbestos", "crack", "damage", "wear"]:
        if label in mapped:
            return label

    return "no_damage"


def parse_llava_output(text: str) -> str:
    text_l = text.lower().strip()

    match = re.search(
        r"label\s*:\s*(damage|crack|mold|mould|wear|asbestos|no_damage|no damage)",
        text_l,
    )

    if match:
        label = match.group(1).replace(" ", "_")
        if label == "mould":
            return "mold"
        return label

    if "asbestos" in text_l:
        return "asbestos"
    if "mold" in text_l or "mould" in text_l:
        return "mold"
    if "crack" in text_l or "cracked" in text_l:
        return "crack"
    if "wear" in text_l or "worn" in text_l or "stain" in text_l or "discoloration" in text_l:
        return "wear"
    if "no visible" in text_l or "no damage" in text_l or "clean" in text_l or "no issue" in text_l:
        return "no_damage"
    if "damage" in text_l or "broken" in text_l or "hole" in text_l or "defect" in text_l:
        return "damage"

    return "no_damage"


def to_binary(label: str) -> str:
    return "damage" if label != "no_damage" else "not_damage"


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42):
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


def load_llava():
    logger.info("Loading processor: %s", LLAVA_MODEL_ID)
    processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)

    logger.info("Loading LLaVA-1.6 model once: %s", LLAVA_MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()
    return processor, model


def run_llava_on_image(image_path: Path, processor, model) -> str:
    image = Image.open(image_path).convert("RGB")

    prompt = (
        "[INST] <image>\n"
        "You are evaluating a housing inspection image. "
        "Classify the image as exactly one of these labels: "
        "damage, crack, mold, wear, asbestos, no_damage.\n\n"
        "Use crack only for visible cracks.\n"
        "Use mold only for visible mold or mould.\n"
        "Use asbestos only for visible asbestos-like material marks.\n"
        "Use wear for discoloration, paint wear, stains, aging, or surface deterioration.\n"
        "Use damage for other visible damage such as holes, broken material, fire damage, or water damage.\n"
        "Use no_damage when no inspection-relevant issue is visible.\n\n"
        "Answer in this exact format:\n"
        "label: <damage/crack/mold/wear/asbestos/no_damage>\n"
        "reason: <short reason>\n"
        "[/INST]"
    )

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=LLAVA_MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only newly generated answer, not the prompt.
    generated_ids = output_ids[:, input_len:]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return response


def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = f"{'':14s}" + "".join(f"{label:>14s}" for label in labels)
    print(header)

    for label, row in zip(labels, cm):
        print(f"{label:14s}" + "".join(f"{value:14d}" for value in row))


def evaluate(limit: int | None = None, bootstrap: int = 1000) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMG_DIR_TEST.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if limit is not None:
        image_paths = image_paths[:limit]

    logger.info("Dataset path: %s", DATASET_DIR)
    logger.info("Test images : %s", IMG_DIR_TEST)
    logger.info("Test labels : %s", LBL_DIR_TEST)
    logger.info("Evaluating %d images", len(image_paths))

    processor, model = load_llava()
    rows = []

    for i, image_path in enumerate(image_paths, 1):
        label_path = LBL_DIR_TEST / f"{image_path.stem}.txt"
        true_label = get_true_label(label_path)

        try:
            raw_output = run_llava_on_image(image_path, processor, model)
            pred_label = parse_llava_output(raw_output)
        except Exception as exc:
            logger.error("Failed on %s: %s", image_path.name, exc)
            raw_output = ""
            pred_label = "no_damage"

        rows.append({
            "image": image_path.name,
            "true": true_label,
            "pred": pred_label,
            "true_binary": to_binary(true_label),
            "pred_binary": to_binary(pred_label),
            "raw_llava": raw_output[:500],
        })

        if i % 10 == 0:
            logger.info("%d / %d done", i, len(image_paths))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()

    y_true_bin = df["true_binary"].tolist()
    y_pred_bin = df["pred_binary"].tolist()

    summary_rows = []

    metrics = {
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

    for name, fn in metrics.items():
        point = fn(y_true, y_pred)
        mean, low, high = bootstrap_ci(y_true, y_pred, fn, n_bootstrap=bootstrap)
        summary_rows.append({
            "task": "multiclass",
            "metric": name,
            "point_estimate": point,
            "bootstrap_mean": mean,
            "ci_95_low": low,
            "ci_95_high": high,
        })

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
        mean, low, high = bootstrap_ci(y_true_bin, y_pred_bin, fn, n_bootstrap=bootstrap)
        summary_rows.append({
            "task": "binary_damage_vs_not_damage",
            "metric": name,
            "point_estimate": point,
            "bootstrap_mean": mean,
            "ci_95_low": low,
            "ci_95_high": high,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 70)
    print(f"LLaVA-1.6 medical-style evaluation | model: {LLAVA_MODEL_ID}")
    print("=" * 70)
    print(f"Dataset path      : {DATASET_DIR}")
    print(f"Images evaluated  : {len(df)}")
    print(f"Predictions saved : {OUT_CSV}")
    print(f"Summary saved     : {OUT_SUMMARY}")
    print()

    print("Ground-truth distribution:")
    print(df["true"].value_counts().reindex(EVAL_LABELS, fill_value=0))
    print()

    print("Prediction distribution:")
    print(df["pred"].value_counts().reindex(EVAL_LABELS, fill_value=0))
    print()

    print("Multiclass classification report:")
    print(classification_report(y_true, y_pred, labels=EVAL_LABELS, zero_division=0))

    print("Multiclass confusion matrix:")
    print_confusion_matrix(y_true, y_pred, EVAL_LABELS)
    print()

    print("Binary damage vs not_damage classification report:")
    print(classification_report(y_true_bin, y_pred_bin, labels=binary_labels, zero_division=0))

    print("Binary confusion matrix:")
    print_confusion_matrix(y_true_bin, y_pred_bin, binary_labels)
    print()

    print("Medical-style summary with 95% bootstrap CIs:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="LLaVA-1.6 medical-style baseline evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    args = parser.parse_args()

    evaluate(limit=args.limit, bootstrap=args.bootstrap)