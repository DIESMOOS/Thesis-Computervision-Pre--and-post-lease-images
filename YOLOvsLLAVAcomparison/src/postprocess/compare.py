from src.schemas import ComparisonResult
from src.config import CATEGORIES


def compare_old_new(old_report: dict, new_property_result) -> ComparisonResult:
    old_counts = old_report.get("category_counts", {cat: 0 for cat in CATEGORIES})
    new_counts = new_property_result.category_counts_total

    delta = {
        cat: int(new_counts.get(cat, 0)) - int(old_counts.get(cat, 0))
        for cat in CATEGORIES
    }

    inspection_recommended = (
        delta["damage"] > 0
        or delta["alteration"] > 0
        or new_property_result.inspection_recommended
    )

    return ComparisonResult(
        old_report_counts=old_counts,
        new_report_counts=new_counts,
        delta=delta,
        inspection_recommended=inspection_recommended
    )