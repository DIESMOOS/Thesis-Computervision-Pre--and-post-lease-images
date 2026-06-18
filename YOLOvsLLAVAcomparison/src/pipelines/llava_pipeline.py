"""
Improved LLaVA-1.6 pipeline for housing inspection.

Medical-imaging-inspired improvements:
- structured review procedure
- few-shot textual examples
- self-check step
- strict single-label JSON output
- negation-aware fallback parsing
- batch inference support
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

ALLOWED_LABELS = ["damage", "crack", "mold", "wear", "asbestos", "no_damage"]

USER_PROMPT = """
You are an expert housing inspection assistant.

Your task is to classify the image into exactly ONE label:
damage, crack, mold, wear, asbestos, no_damage.

Use a careful inspection procedure inspired by medical image review:
1. Inspect the visible surface carefully.
2. Look for small abnormalities, not only severe defects.
3. Compare the visible pattern with the label definitions.
4. Check whether your answer is contradicted by the image.
5. Return exactly one final label.

Label definitions:
- crack: a clear line-shaped break, split, fracture, or visible crack.
- mold: damp spots, mildew, fungal growth, black mold-like patches.
- wear: peeling paint, discoloration, rust, staining, aging, or gradual surface deterioration.
- damage: holes, broken material, dents, missing material, water damage, fire damage, severe deterioration.
- asbestos: suspicious fibrous material, asbestos-like board/surface markings, or material markings resembling asbestos inspection examples.
- no_damage: normal surface with no visible inspection-relevant abnormality.

Few-shot examples:
Example 1:
Observation: A plain wall or floor is visible. No cracks, spots, stains, peeling, holes, or suspicious markings are visible.
Reasoning: The surface appears normal.
Final label: no_damage

Example 2:
Observation: A line-shaped break is visible on the surface.
Reasoning: The main visual abnormality is a linear break.
Final label: crack

Example 3:
Observation: Dark damp patches or fungal-looking spots are visible.
Reasoning: The pattern looks like mold or mildew.
Final label: mold

Example 4:
Observation: Paint is peeling, stained, discolored, rusty, or gradually deteriorated.
Reasoning: The issue looks like surface aging or use-related deterioration.
Final label: wear

Example 5:
Observation: There is a hole, broken material, missing material, dent, or severe deterioration.
Reasoning: The object or surface is physically damaged.
Final label: damage

Example 6:
Observation: The surface has suspicious material markings or asbestos-like texture.
Reasoning: The visual pattern resembles asbestos-related material markings.
Final label: asbestos

Important rules:
- Choose no_damage only if the image is clearly normal.
- If there is any visible abnormality, choose the closest defect label.
- Do not output multiple labels.
- Do not use labels like "wear|mold".
- The final label must be exactly one of the allowed labels.

Return ONLY valid JSON:
{
  "observations": "short visual description",
  "reasoning": "short inspection reasoning",
  "self_check": "briefly check whether another label would fit better",
  "label": "damage|crack|mold|wear|asbestos|no_damage",
  "uncertain": true/false
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

    for part in parts:
        if part == "mould":
            part = "mold"

        if part in ["nodamage", "no__damage", "none", "normal"]:
            part = "no_damage"

        if part in ALLOWED_LABELS:
            return part

    if text == "mould":
        return "mold"

    if text in ["nodamage", "no__damage", "none", "normal"]:
        return "no_damage"

    if text in ALLOWED_LABELS:
        return text

    return "no_damage"


def remove_negated_phrases(text: str) -> str:
    t = text.lower()

    negated_patterns = [
        r"no clear [^.]*",
        r"no visible [^.]*",
        r"no signs? of [^.]*",
        r"no evidence of [^.]*",
        r"does not show [^.]*",
        r"not visible [^.]*",
        r"without [^.]*",
    ]

    for pattern in negated_patterns:
        t = re.sub(pattern, " ", t)

    return t


def fallback_label(text: str) -> str:
    t_original = text.lower()
    t = remove_negated_phrases(t_original)

    if any(w in t for w in ["asbestos", "asbestos_like", "asbestos-like"]):
        return "asbestos"

    if any(w in t for w in [
        "mold", "mould", "mildew", "fungal",
        "black spot", "black spots",
        "damp spot", "damp spots",
        "dark damp", "fungus"
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
        "severe deterioration", "physical damage"
    ]):
        return "damage"

    if any(w in t for w in [
        "wear", "worn", "peeling", "paint damage",
        "stain", "stained", "discolor", "discolour",
        "rust", "aging", "ageing", "deteriorat"
    ]):
        return "wear"

    if any(w in t_original for w in [
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


def make_parsed(
    label: str,
    observations: str = "",
    reasoning: str = "",
    self_check: str = "",
    uncertain: bool = False,
) -> dict:
    label = normalize_label(label)

    counts = empty_category_counts()
    counts[label] = 1

    if label != "no_damage":
        counts["no_damage"] = 0

    return {
        "categories_present": [label],
        "category_counts": counts,
        "observations": observations.strip(),
        "reasoning": reasoning.strip(),
        "self_check": self_check.strip(),
        "uncertain": bool(uncertain),
    }


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{[\s\S]*\}", text)

    if match:
        try:
            parsed = json.loads(match.group())

            observations = str(parsed.get("observations", ""))
            reasoning = str(parsed.get("reasoning", ""))
            self_check = str(parsed.get("self_check", ""))
            raw_label = str(parsed.get("label", "no_damage"))
            uncertain = bool(parsed.get("uncertain", False))

            label = normalize_label(raw_label)

            combined = f"{observations} {reasoning} {self_check}"

            # If model says no_damage but describes a defect, override.
            if label == "no_damage":
                fallback = fallback_label(combined)
                if fallback != "no_damage":
                    label = fallback

            return make_parsed(
                label=label,
                observations=observations,
                reasoning=reasoning,
                self_check=self_check,
                uncertain=uncertain,
            )

        except Exception as exc:
            logger.warning("Could not parse JSON output: %s", exc)

    label = fallback_label(text)
    return make_parsed(
        label=label,
        observations=text[:300],
        reasoning="Fallback parser used.",
        self_check="JSON parsing failed.",
        uncertain=True,
    )


def _build_prompt():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]


def run_llava_on_image(image_path: Path) -> tuple[str, dict]:
    processor, model = _load_model()
    image = Image.open(image_path).convert("RGB")

    prompt_text = processor.apply_chat_template(
        [_build_prompt()],
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

    conversations = [_build_prompt() for _ in image_paths]

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
            parsed = make_parsed(
                label="no_damage",
                observations="",
                reasoning=f"Inference error: {exc}",
                self_check="Inference failed.",
                uncertain=True,
            )

        results.append(
            {
                "image_id": image_path.name,
                "model_name": "llava_medical_inspired",
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