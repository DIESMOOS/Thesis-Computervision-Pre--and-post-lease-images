"""
Optimized LLaVA-1.6 pipeline for housing inspection.
"""

import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from src.config import (
    CATEGORIES,
    IMAGE_EXTS,
    LLAVA_MODEL_ID,
    LLAVA_MAX_NEW_TOKENS,
)

logger = logging.getLogger(__name__)

USER_PROMPT = """
You are an expert housing inspection assistant.

Classify the image into exactly one of these labels:
damage, crack, mold, wear, asbestos, no_damage.

Definitions:
- no_damage: normal surface, no clear inspection-relevant defect.
- crack: clear line-shaped break, split, fracture, or visible crack.
- mold: damp spots, mildew, fungal growth, black mold-like patches.
- wear: peeling paint, discoloration, rust, stains, aging, gradual surface deterioration.
- damage: holes, broken material, dents, missing material, severe water/fire damage.
- asbestos: suspicious fibrous material or asbestos-like surface markings.

Important:
- Choose crack if a line-shaped break is visible.
- Choose mold if damp/fungal/black spots are visible.
- Choose wear if the issue is peeling, staining, discoloration, or gradual aging.
- Choose damage only for broken, missing, holed, dented, or severely damaged material.
- Choose no_damage only if no clear defect is visible.

Return ONLY valid JSON in this exact format:
{
  "observations": "short visual description",
  "label": "damage|crack|mold|wear|asbestos|no_damage",
  "summary": "short reason"
}
"""

_processor: LlavaNextProcessor | None = None
_model: LlavaNextForConditionalGeneration | None = None


def _load_model() -> tuple[LlavaNextProcessor, LlavaNextForConditionalGeneration]:
    global _processor, _model

    if _processor is not None and _model is not None:
        return _processor, _model

    logger.info("Loading LLaVA model: %s", LLAVA_MODEL_ID)

    _processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    _model.eval()
    return _processor, _model


def get_image_paths(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def empty_category_counts() -> dict[str, int]:
    return {cat: 0 for cat in CATEGORIES}


def normalize_label(label: str) -> str:
    text = str(label).lower().strip()
    text = text.replace(" ", "_").replace("-", "_")

    parts = re.split(r"[|,;/]+", text)
    parts = [p.strip() for p in parts if p.strip()]

    cleaned = []
    for part in parts:
        if part == "mould":
            part = "mold"
        if part in ["nodamage", "no__damage", "none", "normal"]:
            part = "no_damage"
        cleaned.append(part)

    priority = [
        "asbestos",
        "mold",
        "crack",
        "damage",
        "wear",
        "no_damage",
    ]

    for label_name in priority:
        if label_name in cleaned:
            return label_name

    for label_name in priority:
        if label_name in text:
            return label_name

    return "no_damage"


def fallback_label(text: str) -> str:
    t = text.lower()

    if any(w in t for w in ["asbestos", "asbestos-like"]):
        return "asbestos"

    if any(w in t for w in [
        "mold", "mould", "mildew", "fungal",
        "black spot", "black spots",
        "damp spot", "damp spots"
    ]):
        return "mold"

    if any(w in t for w in [
        "crack", "cracked", "fracture", "split",
        "visible crack", "line-shaped break", "line shaped break"
    ]):
        return "crack"

    if any(w in t for w in [
        "hole", "broken", "dent", "missing material",
        "water damage", "fire damage", "damaged surface",
        "severe deterioration"
    ]):
        return "damage"

    if any(w in t for w in [
        "wear", "worn", "peeling", "paint damage",
        "stain", "stained", "discolor", "discolour",
        "rust", "aging", "ageing", "deteriorat"
    ]):
        return "wear"

    if any(w in t for w in [
        "no visible defect",
        "no visible damage",
        "no inspection-relevant defect",
        "no defect",
        "no defects",
        "appears intact",
        "looks normal",
        "normal surface",
        "normal wall",
        "unblemished surface",
        "no clear defect",
        "no clear damage",
    ]):
        return "no_damage"

    return "no_damage"


def make_parsed(label: str, summary: str = "", observations: str = "") -> dict:
    label = normalize_label(label)

    counts = empty_category_counts()
    counts[label] = 1

    if label != "no_damage":
        counts["no_damage"] = 0

    return {
        "categories_present": [label],
        "category_counts": counts,
        "observations": observations.strip(),
        "summary": summary.strip() or "No summary provided.",
    }


def fallback_label(text: str) -> str:
    t = text.lower()

    # If LLaVA clearly says there is no defect, respect that first.
    no_damage_phrases = [
        "no visible defect",
        "no visible damage",
        "no inspection-relevant defect",
        "no defect",
        "no defects",
        "appears intact",
        "looks normal",
        "normal surface",
        "normal wall",
        "unblemished surface",
        "no clear defect",
        "no clear damage",
    ]

    if any(w in t for w in no_damage_phrases):
        return "no_damage"

    if any(w in t for w in ["asbestos", "asbestos-like"]):
        return "asbestos"

    if any(w in t for w in [
        "mold", "mould", "mildew", "fungal",
        "black spot", "black spots", "damp spot", "damp spots"
    ]):
        return "mold"

    if any(w in t for w in [
        "crack", "cracked", "fracture", "split",
        "line-shaped break", "line shaped break",
        "visible crack"
    ]):
        return "crack"

    if any(w in t for w in [
        "hole", "broken", "dent", "missing material",
        "water damage", "fire damage", "damaged surface"
    ]):
        return "damage"

    if any(w in t for w in [
        "wear", "worn", "peeling", "paint damage",
        "stain", "stained", "discolor", "discolour",
        "rust", "aging", "ageing", "deteriorat"
    ]):
        return "wear"

    return "no_damage"


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{[\s\S]*\}", text)

    if match:
        try:
            parsed = json.loads(match.group())

            label = parsed.get("label", "no_damage")
            summary = parsed.get("summary", "")
            observations = parsed.get("observations", "")

            label = normalize_label(label)

            combined = f"{observations} {summary} {text}"

            # Only override no_damage if the text clearly contains a defect.
            if label == "no_damage":
                label = fallback_label(combined)

            return make_parsed(label, summary, observations)

        except Exception as exc:
            logger.warning("Could not parse JSON output: %s", exc)

    label = fallback_label(text)
    return make_parsed(label, text[:300], text[:300])


def run_llava_on_image(image_path: Path) -> tuple[str, dict]:
    processor, model = _load_model()
    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=prompt_text,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=LLAVA_MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids = output_ids[:, input_len:]

    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    parsed = _extract_json(raw_text)
    return raw_text, parsed

def run_llava_on_images(image_paths: list[Path]) -> list[tuple[str, dict]]:
    processor, model = _load_model()

    images = [
        Image.open(image_path).convert("RGB")
        for image_path in image_paths
    ]

    conversations = []

    for _ in image_paths:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }
            ]
        )

    prompt_texts = [
        processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        for conversation in conversations
    ]

    inputs = processor(
        images=images,
        text=prompt_texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=LLAVA_MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids = output_ids[:, input_len:]

    raw_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    results = []

    for raw_text in raw_texts:
        raw_text = raw_text.strip()
        parsed = _extract_json(raw_text)
        results.append((raw_text, parsed))

    return results

def run_llava_on_folder(folder_path: str | Path) -> list[dict]:
    image_paths = get_image_paths(folder_path)
    results = []

    for image_path in image_paths:
        try:
            raw_text, parsed = run_llava_on_image(image_path)
        except Exception as exc:
            logger.error("LLaVA failed on %s: %s", image_path.name, exc)
            raw_text = ""
            parsed = make_parsed("no_damage", f"Inference error: {exc}", "")

        results.append(
            {
                "image_id": image_path.name,
                "model_name": "llava_optimized",
                "raw_output": raw_text,
                "parsed_output": parsed,
            }
        )

    logger.info("Processed %d images with optimized LLaVA", len(results))
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/properties/001/post_lease"
    output = run_llava_on_folder(folder)

    print(f"Processed {len(output)} images")
    for item in output[:3]:
        print(json.dumps(item, indent=2))