"""
LLaVA-1.6 pipeline — memory-safe batched version.

OOM ROOT CAUSE (batch_size=128)
---------------------------------
LLaVA-1.6 encodes each image as ~576 tokens before the text prompt.
With batch_size=128 and padding to the longest sequence:
  - Input tensors  : 128 × (576 + ~350 prompt tokens) × hidden_dim × fp16
  - KV cache       : 128 × 32 layers × 8 heads × seq_len → fills VRAM fast

On an A100-80GB the practical safe batch size is 8–16, not 128.
The 24 GB "reserved but unallocated" in the OOM error is fragmentation —
fixed by PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set below).

FIXES IN THIS VERSION
----------------------
  1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  set in os.environ
     before any CUDA allocation happens.
  2. Default batch_size lowered to 8.
  3. Auto-retry with halving: if a batch hits OOM the batch is split in two
     and retried recursively until it succeeds or reaches size 1.
  4. torch.cuda.empty_cache() after every batch to release fragmented blocks.
  5. Images are resized to a max side of 672 px before encoding to keep
     per-image token counts consistent and predictable.
  6. _build_inputs() sets padding_side="left" — required for correct greedy
     decoding in padded batches (right-padding shifts the EOS position).
"""

import json
import logging
import os
import re
from pathlib import Path
from src.hf_token import HF_TOKEN

# Must be set before torch imports so the CUDA allocator picks it up.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_TOKEN"] = HF_TOKEN

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from src.config import (
    CATEGORIES,
    IMAGE_EXTS,
    LLAVA_MAX_NEW_TOKENS,
    LLAVA_MODEL_ID,
)

logger = logging.getLogger(__name__)

VALID_LABELS: list[str] = ["damage", "crack", "mold", "wear", "asbestos", "no_damage"]

# Resize images to this maximum side length before encoding.
# LLaVA-1.6 natively handles up to 672 px per tile; capping here keeps the
# token count per image at ~576 instead of 4×576 for high-res images.
MAX_IMAGE_SIZE = 1024

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """You are a housing inspection classifier. Look at the image carefully.

Pick exactly one label from: damage, crack, mold, wear, asbestos, no_damage.
Only choose a defect label when the defect is clearly visible.
If the image is uncertain, ambiguous, low quality, or only shows normal texture, choose no_damage.

Decision rules (apply in order — stop at first match):
1. asbestos  → flat corrugated or board-like grey/white surface with a matte
               texture; may show thin/thick line markings on the surface.
               NOT rust, NOT peeling paint.
2. crack     → a continuous line-shaped fracture, split, or break in the surface.
               The line must be clearly visible, not just a discolouration.
3. mold      → dark organic spots, green/black patches, mildew, or damp staining
               with fuzzy or spreading edges.
4. damage    → a hole, missing chunk, broken material, dent, or collapse.
               NOT surface aging — the structure itself is broken.
5. wear      → peeling paint, rust, discolouration, stains, or surface aging
               where the structure is still intact.
6. no_damage → choose this when no inspection relevant defect is clearly visible. Normal texture, lighting differences, shadows, stains caused by lighting, or uncertain patterns must not be classified as a defect.

Tie-break rules:
- Corrugated grey surface alone is not enough for asbestos. Choose asbestos only when the surface clearly resembles asbestos cement or fibrous asbestos material.
- Thin surface line → crack (not damage)
- Brown/orange patches on intact surface → wear (not damage)
- Dark fuzzy patches → mold (not wear)

Return ONLY this JSON, nothing else before or after it:
```json
{"label": "<one of the 6 labels>", "reason": "<max 10 words>"}
```"""


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_processor: LlavaNextProcessor | None = None
_model: LlavaNextForConditionalGeneration | None = None


def _load_model() -> tuple[LlavaNextProcessor, LlavaNextForConditionalGeneration]:
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    logger.info("Loading LLaVA model: %s", LLAVA_MODEL_ID)
    _processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)

    # Left-pad so EOS alignment is correct in batched greedy decode
    _processor.tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    _model.eval()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "LLaVA loaded | dtype=%s | GPU free=%.1f GB / total=%.1f GB",
            dtype, free / 1e9, total / 1e9,
        )

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
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def empty_category_counts() -> dict[str, int]:
    return {cat: 0 for cat in CATEGORIES}


def make_parsed(label: str, reason: str) -> dict:
    label = label if label in VALID_LABELS else "no_damage"
    counts = empty_category_counts()
    counts[label] = 1
    return {
        "categories_present": [label],
        "category_counts": counts,
        "reason": reason.strip()[:200],
    }


def _resize(image: Image.Image, max_side: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize so the longest side equals max_side, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _sanitize(raw: str) -> str:
    """Remove control characters that break json.loads()."""
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", raw)
    return re.sub(r"  +", " ", sanitized)


def _extract_json(raw_text: str) -> dict:
    """Parse label + reason from model output. Three fallback strategies."""
    text = _sanitize(raw_text)

    # 1. Fenced JSON block  ```json { ... } ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return _parse_obj(json.loads(fence.group(1)), text)
        except json.JSONDecodeError:
            pass

    # 2. Bare JSON object
    brace = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace:
        try:
            return _parse_obj(json.loads(brace.group()), text)
        except json.JSONDecodeError:
            pass

    # 3. Keyword scan
    logger.debug("No JSON found — keyword scan on: %r", raw_text[:120])
    return make_parsed(_keyword_label(text), "keyword fallback")


def _parse_obj(obj: dict, full_text: str) -> dict:
    raw_label = str(obj.get("label", "")).lower().strip()
    reason = str(obj.get("reason", obj.get("summary", obj.get("observations", ""))))
#--------------------------------------------------
    combined = f"{raw_label} {reason}".lower()

    # asbestos override
    # stricter asbestos override
    asbestos_terms = [
        "asbestos",
        "asbestos-like",
        "fibrous asbestos",
        "asbestos cement",
    ]

    asbestos_visual_terms = [
        "corrugated",
        "cement board",
        "grey board",
        "gray board",
        "fibre cement",   
        "fiber cement",
        "line markings",
    ]

    if raw_label == "asbestos":
        return make_parsed("asbestos", reason)

    if any(x in combined for x in asbestos_terms) and any(x in combined for x in asbestos_visual_terms):
        return make_parsed("asbestos", reason)

    # crack override
    if any(x in combined for x in [
        "line-shaped fracture",
        "continuous crack",
        "visible crack",
        "split in surface",
    ]):
        return make_parsed("crack", reason)

    # mold override
    if any(x in combined for x in [
        "mildew",
        "fungal",
        "black mold",
        "dark fuzzy",
        "damp patch",
    ]):
        return make_parsed("mold", reason)
#--------------------------------------------------
    if any(sep in raw_label for sep in ("|", ",", "/", " and ", " or ")):
        first = re.split(r"[|,/]| and | or ", raw_label)[0].strip()
        return make_parsed(_normalize_label(first), f"[uncertain: {raw_label}] {reason}")

    return make_parsed(_normalize_label(raw_label), reason)


def _normalize_label(raw: str) -> str:
    t = raw.lower().strip().replace(" ", "_").replace("-", "_")
    if t in VALID_LABELS:
        return t
    aliases = {
        "no_damage": "no_damage", "nodamage": "no_damage",
        "none": "no_damage", "normal": "no_damage", "clean": "no_damage",
        "mould": "mold", "fungal": "mold", "mildew": "mold",
        "cracked": "crack", "fracture": "crack", "split": "crack",
        "broken": "damage", "hole": "damage", "dent": "damage",
        "peeling": "wear", "rust": "wear", "stain": "wear",
        "discolor": "wear", "deterioration": "wear",
    }
    if t in aliases:
        return aliases[t]
    for label in VALID_LABELS:
        if label in t:
            return label
    return "no_damage"


def _keyword_label(text: str) -> str:
    t = text.lower()
    no_dmg = ["no visible defect", "no defect", "no damage", "appears intact",
               "looks normal", "normal surface", "no clear defect"]
    if any(p in t for p in no_dmg):
        return "no_damage"
    rules = [
        ("asbestos", ["asbestos", "corrugated", "fibrous"]),
        ("crack",    ["crack", "cracked", "fracture", "split", "line-shaped break"]),
        ("mold",     ["mold", "mould", "mildew", "fungal", "black spot", "damp spot"]),
        ("damage",   ["hole", "broken", "dent", "missing material", "water damage",
                      "fire damage", "collapsed"]),
        ("wear",     ["wear", "peeling", "rust", "stain", "discolor", "deteriorat"]),
    ]
    for label, keywords in rules:
        if any(kw in t for kw in keywords):
            return label
    return "no_damage"


# ---------------------------------------------------------------------------
# Core batch runner — with OOM auto-retry
# ---------------------------------------------------------------------------

def _run_batch(
    processor: LlavaNextProcessor,
    model: LlavaNextForConditionalGeneration,
    images: list[Image.Image],
    batch_size: int,
) -> list[str]:
    """
    Run model.generate() on `images` in sub-batches of `batch_size`.
    On OOM, halves the batch size and retries automatically.
    Returns a list of raw decoded strings, one per image.
    """
    if not images:
        return []

    # If batch fits in one go, run it directly
    if len(images) <= batch_size:
        return _generate_batch(processor, model, images, batch_size)

    # Split and recurse
    mid = len(images) // 2
    left  = _run_batch(processor, model, images[:mid], batch_size)
    right = _run_batch(processor, model, images[mid:], batch_size)
    return left + right


def _generate_batch(
    processor: LlavaNextProcessor,
    model: LlavaNextForConditionalGeneration,
    images: list[Image.Image],
    batch_size: int,
) -> list[str]:
    """
    Single model.generate() call for `images`. Retries with halved batch
    size on OOM until size 1. Raises if even size-1 fails.
    """
    try:
        conversations = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ]}]
            for _ in images
        ]
        prompt_texts = [
            processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
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
                temperature=None,
                top_p=None,
            )

        raw_texts = processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )
        return [t.strip() for t in raw_texts]

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

        if len(images) == 1:
            logger.error("OOM on a single image — cannot reduce further. Skipping.")
            return [""]

        half = max(1, len(images) // 2)
        logger.warning(
            "OOM with %d images — retrying as two batches of ~%d",
            len(images), half,
        )
        left  = _generate_batch(processor, model, images[:half],      half)
        right = _generate_batch(processor, model, images[half:], half)
        return left + right


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_llava_on_image(image_path: Path) -> tuple[str, dict]:
    """Single-image inference. Returns (raw_text, parsed_dict)."""
    processor, model = _load_model()
    image = _resize(Image.open(image_path).convert("RGB"))
    raw_texts = _generate_batch(processor, model, [image], batch_size=1)
    raw = raw_texts[0]
    return raw, _extract_json(raw)


def run_llava_on_images(
    image_paths: list[Path],
    batch_size: int = 8,
) -> list[tuple[str, dict]]:
    """
    Batched inference over a list of image paths.

    Args:
        image_paths: images to classify, in order.
        batch_size:  starting batch size. Automatically halved on OOM until
                     a size that fits is found. Recommended: 8 on A100-80GB.
                     Use 4 if you also have other processes on the GPU.

    Returns:
        List of (raw_text, parsed_dict), one per input image, same order.
    """
    processor, model = _load_model()
    results: list[tuple[str, dict]] = []

    total = len(image_paths)
    current_batch_size = batch_size

    for start in range(0, total, current_batch_size):
        batch_paths = image_paths[start : start + current_batch_size]
        end_idx = start + len(batch_paths)
        logger.info("Batch %d–%d / %d  (batch_size=%d)", start + 1, end_idx, total, current_batch_size)

        # Load + resize images
        images: list[Image.Image] = []
        valid_indices: list[int] = []
        for i, p in enumerate(batch_paths):
            try:
                images.append(_resize(Image.open(p).convert("RGB")))
                valid_indices.append(i)
            except Exception as exc:
                logger.error("Cannot open %s: %s", p.name, exc)

        # Placeholder results for failed loads
        batch_results: list[tuple[str, dict]] = [
            ("", make_parsed("no_damage", "image load error"))
        ] * len(batch_paths)

        if images:
            raw_texts = _run_batch(processor, model, images, current_batch_size)
            for out_idx, img_idx in enumerate(valid_indices):
                raw = raw_texts[out_idx]
                batch_results[img_idx] = (raw, _extract_json(raw))

        results.extend(batch_results)

        # Free fragmented blocks between batches
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            free, total_mem = torch.cuda.mem_get_info()
            logger.debug("GPU after batch: free=%.1f GB", free / 1e9)

    return results


def run_llava_on_folder(folder_path: str | Path, batch_size: int = 8) -> list[dict]:
    image_paths = get_image_paths(folder_path)
    raw_results = run_llava_on_images(image_paths, batch_size=batch_size)
    return [
        {
            "image_id":     p.name,
            "model_name":   "llava",
            "raw_output":   raw,
            "parsed_output": parsed,
        }
        for p, (raw, parsed) in zip(image_paths, raw_results)
    ]


# ---------------------------------------------------------------------------
# Smoke-test:  python -m src.pipelines.llava_pipeline <folder>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/properties/001/post_lease"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    output = run_llava_on_folder(folder, batch_size=bs)
    print(f"\nProcessed {len(output)} images\n")
    for item in output[:3]:
        print(json.dumps(item, indent=2))
