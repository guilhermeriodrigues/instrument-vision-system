from dataclasses import dataclass
from typing import List


@dataclass
class OCRItem:
    text: str
    value: float
    center_x: float
    center_y: float
    confidence: float


@dataclass
class ScalePoint:
    value: float
    angle: float
    confidence: float


@dataclass
class ScaleModel:
    unit: str
    points: List[ScalePoint]
    confidence: float