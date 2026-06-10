"""
LLaVA-1.6 pipeline for housing damage/wear detection.

Loads llava-hf/llava-v1.6-mistral-7b-hf once (singleton pattern) and runs
per-image inference. The model is prompted to return a structured JSON block
so we can skip fragile regex parsing.

Usage:
    from src.pipelines.llava_pipeline import run_llava_on_folder
    results = run_llava_on_folder("data/properties/001/post_lease")
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

# ---------------------------------------------------------------------------
# Prompt  — ask the model for structured JSON so parsing is reliable
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert housing inspector. "
    "Analyse the image and classify any visible issues."
)

USER_PROMPT = """\
Look at this housing inspection image carefully.

Classify what you see into one or more of these categories:
  - damage   : structural issues such as cracks, holes, broken surfaces, mould
  - wear     : surface degradation such as peeling paint, stains, discolouration, rust
  - alteration : unauthorised modifications or additions
  - no_damage : the area looks fine

Return ONLY a JSON object with this exact structure and nothing else:
{
  "categories_present": ["damage"],
  "category_counts": {"damage": 1, "wear": 0, "alteration": 0, "no_damage": 0},
  "summary": "One-sentence description of what you see."
}
"""

# ---------------------------------------------------------------------------
# Model singleton  (loaded at most once per process)
# ---------------------------------------------------------------------------

_processor: LlavaNextProcessor | None = None
_model: LlavaNextForConditionalGeneration | None = None


def _load_model() -> tuple[LlavaNextProcessor, LlavaNextForConditionalGeneration]:
    global _processor, _model

    if _processor is not None and _model is not None:
        return _processor, _model

    logger.info("Loading LLaVA-1.6 model: %s …", LLAVA_MODEL_ID)

    _processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        # Reduces VRAM by ~40 % on single-GPU Snellius nodes
        load_in_4bit=torch.cuda.is_available(),
    )
    _model.eval()

    logger.info("LLaVA model loaded (dtype=%s, device=%s)", dtype, next(_model.parameters()).device)
    return _processor, _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_paths(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def empty_category_counts() -> dict[str, int]:
    return {cat: 0 for cat in CATEGORIES}


def _extract_json(raw_text: str) -> dict:
    """
    Pull the first JSON object out of the model's raw output.
    Falls back to rule-based parsing if no JSON is found.
    """
    # Strip the echoed prompt — LLaVA-Next repeats the conversation before the answer
    # The answer always starts after the last [/INST] token
    if "[/INST]" in raw_text:
        raw_text = raw_text.split("[/INST]")[-1]

    # Try to find a JSON block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return _validate_and_fix(parsed)
        except json.JSONDecodeError:
            pass

    # Fallback: keyword scan
    logger.warning("Could not parse JSON from LLaVA output — falling back to keyword scan")
    return _keyword_parse(raw_text)


def _validate_and_fix(parsed: dict) -> dict:
    """Ensure all required keys exist and counts are sane."""
    counts = empty_category_counts()
    incoming = parsed.get("category_counts", {})
    for cat in CATEGORIES:
        counts[cat] = max(0, int(incoming.get(cat, 0)))

    # Recompute categories_present from counts (don't trust the model's list)
    active = [cat for cat in CATEGORIES if counts[cat] > 0 and cat != "no_damage"]
    if not active:
        counts["no_damage"] = 1
        active = ["no_damage"]
    else:
        counts["no_damage"] = 0

    return {
        "categories_present": active,
        "category_counts": counts,
        "summary": str(parsed.get("summary", "")).strip() or "No summary provided.",
    }


def _keyword_parse(text: str) -> dict:
    t = text.lower()
    counts = empty_category_counts()

    if any(w in t for w in ["crack", "broken", "hole", "mould", "mold", "damage", "structural"]):
        counts["damage"] = 1
    if any(w in t for w in ["wear", "stain", "peel", "paint", "discolor", "rust", "deteriorat"]):
        counts["wear"] = 1
    if any(w in t for w in ["alteration", "unauthor", "modif", "added", "new install"]):
        counts["alteration"] = 1

    active = [cat for cat in CATEGORIES if counts[cat] > 0 and cat != "no_damage"]
    if not active:
        counts["no_damage"] = 1
        active = ["no_damage"]

    return {
        "categories_present": active,
        "category_counts": counts,
        "summary": text[:200].strip(),
    }


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def run_llava_on_image(image_path: Path) -> dict:
    """
    Run LLaVA-1.6 on a single image and return the parsed output dict.
    """
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

    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=LLAVA_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,       # must be None when do_sample=False
            top_p=None,
        )

    raw_text = processor.decode(output_ids[0], skip_special_tokens=True)
    return raw_text, _extract_json(raw_text)


# ---------------------------------------------------------------------------
# Folder-level runner
# ---------------------------------------------------------------------------

def run_llava_on_folder(folder_path: str | Path) -> list[dict]:
    """
    Run LLaVA-1.6 on every image in *folder_path*.

    Returns:
        List of dicts, one per image:
        {
            "image_id":     str,
            "model_name":   "llava",
            "raw_output":   str,
            "parsed_output": {"categories_present", "category_counts", "summary"}
        }
    """
    image_paths = get_image_paths(folder_path)
    results = []

    for image_path in image_paths:
        try:
            raw_text, parsed = run_llava_on_image(image_path)
        except Exception as exc:
            logger.error("LLaVA inference failed on %s: %s", image_path.name, exc)
            parsed = {
                "categories_present": ["no_damage"],
                "category_counts": empty_category_counts() | {"no_damage": 1},
                "summary": f"Inference error: {exc}",
            }
            raw_text = ""

        results.append({
            "image_id": image_path.name,
            "model_name": "llava",
            "raw_output": raw_text,
            "parsed_output": parsed,
        })

        logger.debug("%s → %s", image_path.name, parsed["categories_present"])

    logger.info("LLaVA processed %d images in %s", len(results), folder_path)
    return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/properties/001/post_lease"
    output = run_llava_on_folder(folder)

    print(f"\nProcessed {len(output)} images\n")
    for item in output[:3]:
        print(json.dumps(item, indent=2))
