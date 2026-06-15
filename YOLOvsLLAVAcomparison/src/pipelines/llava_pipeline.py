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

Use a careful inspection procedure inspired by medical image review:
1. Inspect the image for visible abnormalities.
2. Compare it with the examples.
3. Decide whether a real defect is visible.
4. Select exactly one final label.

Allowed labels:
damage, crack, mold, wear, asbestos, no_damage.

Few-shot examples:

Example 1:
Observation: A normal wall or floor surface is visible. There are no clear cracks, stains, mold, holes, or broken parts.
Label: no_damage

Example 2:
Observation: A clear line-shaped break is visible in the wall or surface.
Label: crack

Example 3:
Observation: Dark damp spots or fungal-looking patches are visible.
Label: mold

Example 4:
Observation: Paint is peeling, discolored, stained, rusty, or worn from normal use.
Label: wear

Example 5:
Observation: A surface is broken, has a hole, missing material, dent, fire damage, or water damage.
Label: damage

Example 6:
Observation: Suspicious fibrous material or asbestos-like surface markings are visible.
Label: asbestos

Important rules:
- If no clear defect is visible, choose no_damage.
- Do not classify shadows, lighting differences, texture, corners, joints, or ordinary material patterns as defects.
- Do not choose crack unless a clear line-shaped break is visible.
- Do not choose mold unless visible damp spots, mildew, or fungal growth are present.
- Do not choose damage unless there is clear broken material, a hole, missing material, or severe deterioration.
- The final label must be exactly one allowed label.

Return ONLY valid JSON:
{
  "observations": "short description of visible relevant findings",
  "label": "damage|crack|mold|wear|asbestos|no_damage",
  "summary": "short reason for the chosen label"
}

Important hierarchy:
- If the defect is a line shaped break, choose crack, not damage.
- If the defect is mold or damp black spots, choose mold, not damage.
- If the defect is peeling, discoloration, rust, or gradual surface aging, choose wear, not damage.
- Use damage only for other visible defects such as holes, broken surfaces, dents, missing material, water damage, or fire damage.
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
    label = str(label).lower().strip()
    label = label.replace(" ", "_").replace("-", "_")

    if label == "mould":
        return "mold"

    if label in ["nodamage", "no_damage", "no__damage"]:
        return "no_damage"

    if label in CATEGORIES:
        return label

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
    ]):
        return "no_damage"

    if any(w in t for w in ["asbestos", "asbestos-like", "fibrous insulation"]):
        return "asbestos"

    if any(w in t for w in ["mold", "mould", "mildew", "fungal", "black spots"]):
        return "mold"

    if any(w in t for w in ["crack", "cracked", "fracture", "split"]):
        return "crack"

    if any(w in t for w in [
        "hole",
        "broken",
        "dent",
        "missing material",
        "water damage",
        "fire damage",
        "damaged surface",
    ]):
        return "damage"

    if any(w in t for w in [
        "wear",
        "worn",
        "peeling",
        "stain",
        "discolor",
        "discolour",
        "rust",
        "aging",
        "deteriorat",
    ]):
        return "wear"

    return "no_damage"


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*?\}", text, re.DOTALL)

    if match:
        try:
            parsed = json.loads(match.group())

            label = parsed.get("label", "no_damage")
            summary = parsed.get("summary", "")
            observations = parsed.get("observations", "")

            return make_parsed(label, summary, observations)

        except Exception as exc:
            logger.warning("Could not parse JSON output: %s", exc)

    label = fallback_label(text)
    return make_parsed(label, text[:300], "")


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
            parsed = make_parsed("no_damage", f"Inference error: {exc}", "")

        results.append({
            "image_id": image_path.name,
            "model_name": "llava_optimized",
            "raw_output": raw_text,
            "parsed_output": parsed,
        })

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