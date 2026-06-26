from src.schemas import ComparisonResult
from src.config import CATEGORIES


DEFECT_CATEGORIES = [
    "damage",
    "crack",
    "mold",
    "wear",
    "asbestos",
]


def compare_old_new(old_report: dict, new_property_result) -> ComparisonResult:
    old_counts_raw = old_report.get("category_counts", {})

    old_counts = {
        cat: int(old_counts_raw.get(cat, 0))
        for cat in CATEGORIES
    }

    new_counts = {
        cat: int(new_property_result.category_counts_total.get(cat, 0))
        for cat in CATEGORIES
    }

    delta = {
        cat: new_counts.get(cat, 0) - old_counts.get(cat, 0)
        for cat in CATEGORIES
    }

    inspection_recommended = (
        any(delta.get(cat, 0) > 0 for cat in DEFECT_CATEGORIES)
        or new_property_result.inspection_recommended
    )

    return ComparisonResult(
        old_report_counts=old_counts,
        new_report_counts=new_counts,
        delta=delta,
        inspection_recommended=inspection_recommended
    )