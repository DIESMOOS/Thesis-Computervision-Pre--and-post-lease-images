import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
except ModuleNotFoundError:

    print("Installing transformers...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "transformers",
        "accelerate",
        "pillow",
        "torch",
        "--default-timeout=100"
    ])

    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


from pathlib import Path
from PIL import Image
import pandas as pd
import torch
MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

IMG_DIR = Path("data/inspection_dataset/images/test")
LBL_DIR = Path("data/inspection_dataset/labels/test")

OUT_CSV = Path("results/llava_baseline_results.csv")
OUT_CSV.parent.mkdir(exist_ok=True)

LABELS = ["damage", "wear", "no_damage"]

CLASS_MAP = {
    0: "damage",
    1: "damage",
    2: "damage",
    3: "wear",
    4: "damage",
    5: "no_damage",
}

PROMPT = """
You are inspecting a housing image.

Choose exactly one category:
damage
wear
no_damage

Return only one category name.
"""

processor = LlavaNextProcessor.from_pretrained(MODEL)

model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


def get_true_label(label_path: Path) -> str:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "no_damage"

    classes = []

    for line in label_path.read_text().splitlines():
        if line.strip():
            cls_id = int(line.split()[0])
            classes.append(CLASS_MAP.get(cls_id, "damage"))

    if "damage" in classes:
        return "damage"
    if "wear" in classes:
        return "wear"
    return "no_damage"


def map_llava_output(text: str) -> str:
    t = text.lower()

    if "no_damage" in t or "no damage" in t or "no visible damage" in t:
        return "no_damage"
    if "wear" in t or "paint" in t or "discolor" in t or "deterioration" in t:
        return "wear"
    if "damage" in t or "crack" in t or "hole" in t or "broken" in t or "mold" in t:
        return "damage"

    return "no_damage"


def run_llava(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }]

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    raw = processor.decode(output[0], skip_special_tokens=True)
    return raw


image_paths = (
    list(IMG_DIR.glob("*.jpg")) +
    list(IMG_DIR.glob("*.jpeg")) +
    list(IMG_DIR.glob("*.png"))
)

print("Images found:", len(image_paths))

rows = []

for image_path in image_paths[:1]:
    label_path = LBL_DIR / f"{image_path.stem}.txt"

    true_label = get_true_label(label_path)
    raw_output = run_llava(image_path)
    pred_label = map_llava_output(raw_output)

    rows.append({
        "image": str(image_path),
        "true": true_label,
        "pred": pred_label,
        "raw_llava": raw_output
    })

    print(image_path.name, "true:", true_label, "pred:", pred_label)


df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

correct = (df["true"] == df["pred"]).sum()
total = len(df)
accuracy = correct / total if total > 0 else 0

print()
print("Images evaluated:", total)
print("Saved to:", OUT_CSV)
print()
print("Accuracy:", round(accuracy, 3))

print()
print("Per-class results:")

for label in LABELS:
    tp = ((df["true"] == label) & (df["pred"] == label)).sum()
    fp = ((df["true"] != label) & (df["pred"] == label)).sum()
    fn = ((df["true"] == label) & (df["pred"] != label)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(label)
    print(" precision:", round(precision, 3))
    print(" recall:   ", round(recall, 3))
    print(" f1:       ", round(f1, 3))
    print()

print("Confusion matrix:")
print("rows=true, cols=pred")
print(["true/pred"] + LABELS)

for true_label in LABELS:
    row = [true_label]
    for pred_label in LABELS:
        row.append(
            int(
                ((df["true"] == true_label) & (df["pred"] == pred_label)).sum()
            )
        )
    print(row)