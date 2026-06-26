from src.schemas import PropertyResult
from src.config import CATEGORIES

DEFECT_CATEGORIES = ["damage", "crack", "mold", "wear", "asbestos"]


def aggregate_property(property_id: str, model_name: str, image_results: list) -> PropertyResult:
    total_counts = {cat: 0 for cat in CATEGORIES}

    for img in image_results:
        for cat in CATEGORIES:
            total_counts[cat] += int(img.category_counts.get(cat, 0))

    categories_present = [k for k, v in total_counts.items() if v > 0]

    inspection_recommended = any(total_counts.get(cat, 0) > 0 for cat in DEFECT_CATEGORIES)

    return PropertyResult(
        property_id=property_id,
        model_name=model_name,
        images_analyzed=len(image_results),
        categories_present=categories_present,
        category_counts_total=total_counts,
        image_results=image_results,
        property_summary=build_property_summary(total_counts),
        inspection_recommended=inspection_recommended
    )


def build_property_summary(counts: dict) -> str:
    parts = []

    for cat in DEFECT_CATEGORIES:
        if counts.get(cat, 0) > 0:
            parts.append(f"{counts[cat]} images with {cat}")

    if not parts:
        return "No visible inspection relevant issues across the property."

    return "Across the property, detected " + ", ".join(parts) + "."