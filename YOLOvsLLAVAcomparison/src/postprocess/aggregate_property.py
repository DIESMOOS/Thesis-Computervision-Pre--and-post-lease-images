from src.schemas import PropertyResult
from src.config import CATEGORIES


def aggregate_property(property_id: str, model_name: str, image_results: list) -> PropertyResult:
    total_counts = {cat: 0 for cat in CATEGORIES}

    for img in image_results:
        for cat in CATEGORIES:
            total_counts[cat] += img.category_counts.get(cat, 0)

    categories_present = [k for k, v in total_counts.items() if v > 0]

    property_summary = build_property_summary(total_counts)
    inspection_recommended = total_counts["damage"] > 0 or total_counts["alteration"] > 0

    return PropertyResult(
        property_id=property_id,
        model_name=model_name,
        images_analyzed=len(image_results),
        categories_present=categories_present,
        category_counts_total=total_counts,
        image_results=image_results,
        property_summary=property_summary,
        inspection_recommended=inspection_recommended
    )


def build_property_summary(counts: dict) -> str:
    if counts["damage"] == 0 and counts["wear"] == 0 and counts["alteration"] == 0:
        return "No visible inspection-relevant issues across the property."

    parts = []
    for cat in ["damage", "wear", "alteration"]:
        if counts[cat] > 0:
            parts.append(f"{counts[cat]} {cat}")

    return "Across the property, detected " + ", ".join(parts) + "."