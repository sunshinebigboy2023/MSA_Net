from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

TASK_PENDING = "PENDING"
TASK_PREPROCESSING = "PREPROCESSING"
TASK_EXTRACTING = "EXTRACTING"
TASK_INFERRING = "INFERRING"
TASK_SUCCESS = "SUCCESS"
TASK_FAILED = "FAILED"


@dataclass
class ServiceTask:
    task_id: str
    payload: Dict[str, Any]
    status: str = TASK_PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

