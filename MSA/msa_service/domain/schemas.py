from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class PredictionResult:
    taskId: str
    usedModalities: List[str]
    missingModalities: List[str]
    emotionLabel: Optional[str]
    sentimentPolarity: str
    score: float
    confidence: float
    message: str
    error: Optional[str]

    def to_dict(self):
        return asdict(self)

