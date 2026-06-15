"""
LLaVA-1.6 pipeline for housing inspection.

Runs LLaVA on images and returns structured predictions using YOLO-aligned labels:
damage, crack, mold, wear, asbestos, no_damage.
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


USER_PROMPT = """\
You are a housing inspection assistant.

Classify the image into exactly one of these labels:
damage, crack, mold, wear, asbestos, no_damage.

Definitions:
- crack: visible cracks, fractures, splits, or line-shaped breaks.
- mold: visible mold, mould, mildew, fungal growth, or damp-related black spots.
- asbestos: asbestos-like material marks, suspicious fibrous material, or asbestos-related surface patterns.
- wear: surface degradation, peeling paint, stains, discoloration, rust, aging, or gradual deterioration.
- damage: other visible damage such as holes, broken surfaces, dents, missing material, fire damage, or water damage.
- no_damage: no visible inspection-relevant issue.

Rules:
- If a crack is visible, choose crack.
- If mold or mould is visible, choose mold.
- If asbestos-like material is visible, choose asbestos.
- Choose no_damage when the image only shows a normal wall, floor, ceiling, pipe, surface, or room feature without a clear visible defect. Do not mark minor shadows, texture, lighting, normal edges, or ordinary material patterns as damage.
- Choose no_damage only when the area looks fine.

Return only this JSON object:
{
  "label": "one_of_damage_crack_mold_wear_asbestos_no_damage",
  "summary": "one short sentence"
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


def _make_parsed(label: str, summary: str) -> dict:
    label = normalize_label(label)
    counts = empty_category_counts()
    counts[label] = 1

    if label != "no_damage":
        counts["no_damage"] = 0

    return {
        "categories_present": [label],
        "category_counts": counts,
        "summary": summary.strip() or "No summary provided.",
    }


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = label.replace(" ", "_")

    if label == "mould":
        return "mold"

    if label in CATEGORIES:
        return label

    return "no_damage"


def _keyword_label(text: str) -> str:
    t = text.lower()

    # First check for explicit no-damage statements
    if any(w in t for w in [
        "no visible issue",
        "no visible damage",
        "no defect",
        "no signs of",
        "appears intact",
        "looks fine",
        "clean",
        "normal condition"
    ]):
        return "no_damage"

    # Specific issue classes first.
    if any(w in t for w in ["mold", "mould", "mildew", "fungal", "black spots"]):
        return "mold"

    if any(w in t for w in ["asbestos", "asbestos-like", "fibrous"]):
        return "asbestos"

    if any(w in t for w in ["crack", "cracked", "fracture", "split"]):
        return "crack"

    if any(w in t for w in [
        "hole",
        "broken",
        "dent",
        "missing material",
        "fire damage",
        "water damage",
        "damaged surface"
    ]):
        return "damage"

    if any(w in t for w in [
        "wear",
        "worn",
        "stain",
        "peeling",
        "paint",
        "discolor",
        "discolour",
        "rust",
        "aging",
        "deteriorat"
    ]):
        return "wear"

    if any(w in t for w in [
        "no visible",
        "no issue",
        "no damage",
        "looks fine",
        "clean",
        "intact"
    ]):
        return "no_damage"

    if "damage" in t or "defect" in t:
        return "damage"

    return "no_damage"


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()

    match = re.search(r"\{.*?\}", text, re.DOTALL)

    if match:
        try:
            parsed = json.loads(match.group())

            label = parsed.get("label", "no_damage")
            summary = parsed.get("summary", "")

            return _make_parsed(label, summary)

        except Exception:
            pass

    # fallback only if JSON parsing fails
    label = _keyword_label(text)

    return _make_parsed(label, text[:300])


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


def run_llava_on_folder(folder_path: str | Path) -> list[dict]:
    image_paths = get_image_paths(folder_path)
    results = []

    for image_path in image_paths:
        try:
            raw_text, parsed = run_llava_on_image(image_path)
        except Exception as exc:
            logger.error("LLaVA failed on %s: %s", image_path.name, exc)
            raw_text = ""
            parsed = _make_parsed("no_damage", f"Inference error: {exc}")

        results.append(
            {
                "image_id": image_path.name,
                "model_name": "llava",
                "raw_output": raw_text,
                "parsed_output": parsed,
            }
        )

    logger.info("Processed %d images with LLaVA", len(results))
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/properties/001/post_lease"
    output = run_llava_on_folder(folder)

    print(f"Processed {len(output)} images")
    for item in output[:3]:
        print(json.dumps(item, indent=2))