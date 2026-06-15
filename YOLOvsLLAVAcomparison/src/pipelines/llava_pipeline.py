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
        raise NotADirectoryError(f"Path is not a folder: {folder}")

    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def empty_category_counts() -> dict:
    return {category: 0 for category in THESIS_CATEGORIES}


def parse_llava_text(text: str) -> dict:
    """
    Temporary rule-based parser.
    Later this receives real LLaVA output text.
    """
    text_lower = text.lower()
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
        "categories_present": categories_present,
        "category_counts": counts,
        "summary": text
    }


def mock_llava_inference(image_path: Path) -> str:
    """
    Replace this function later with real LLaVA inference.
    """
    return "No visible inspection-relevant issues."


def run_llava_on_folder(folder_path: str) -> list[dict]:
    image_paths = get_image_paths(folder_path)
    results = []

    for image_path in image_paths:
        llava_text = mock_llava_inference(image_path)
        parsed_output = parse_llava_text(llava_text)

        results.append({
            "image_id": image_path.name,
            "model_name": "llava",
            "raw_output": llava_text,
            "parsed_output": parsed_output
        })

    return results


if __name__ == "__main__":
    test_folder = "data/properties/001/post_lease"
    output = run_llava_on_folder(test_folder)

    print(f"Processed {len(output)} images")

    for item in output[:3]:
        print(item)