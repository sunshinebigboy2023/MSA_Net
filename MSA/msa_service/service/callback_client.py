from __future__ import annotations

from typing import Any, Dict

import requests


class CallbackClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def complete(self, payload: Dict[str, Any]) -> None:
        response = requests.post(
            f"{self.base_url}/analysis/callback",
            json=payload,
            headers={"X-MSA-Callback-Token": self.token},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
