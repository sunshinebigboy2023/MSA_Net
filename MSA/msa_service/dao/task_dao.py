from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional
import uuid

from msa_service.domain.tasks import ServiceTask, TASK_SUCCESS


class InMemoryTaskDao:
    def __init__(self):
        self._tasks: Dict[str, ServiceTask] = {}
        self._lock = Lock()

    def create(self, payload: Dict[str, Any]) -> ServiceTask:
        task = ServiceTask(task_id=str(uuid.uuid4()), payload=payload)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> Optional[ServiceTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def set_status(self, task_id: str, status: str, error: Optional[str] = None):
        with self._lock:
            task = self._tasks[task_id]
            task.status = status
            task.error = error

    def set_result(self, task_id: str, result: Dict[str, Any]):
        with self._lock:
            task = self._tasks[task_id]
            task.status = TASK_SUCCESS
            task.result = result
            task.error = None
