from typing import List, Dict, Optional
from pydantic import BaseModel


CATEGORIES = ["damage", "wear", "alteration", "no_damage"]


class DetectionBox(BaseModel):
    label: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]


class ImageResult(BaseModel):
    image_id: str
    model_name: str
    categories_present: List[str]
    category_counts: Dict[str, int]
    detections: List[DetectionBox]
    summary: str


class PropertyResult(BaseModel):
    property_id: str
    model_name: str
    images_analyzed: int
    categories_present: List[str]
    category_counts_total: Dict[str, int]
    image_results: List[ImageResult]
    property_summary: str
    inspection_recommended: bool


class ComparisonResult(BaseModel):
    old_report_counts: Dict[str, int]
    new_report_counts: Dict[str, int]
    delta: Dict[str, int]
    inspection_recommended: bool