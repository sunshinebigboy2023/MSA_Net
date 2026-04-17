from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class MissingDependencyError(RuntimeError):
    dependency: str
    message: str

    def __str__(self):
        return self.message


class TextFeatureExtractor:
    def __init__(self, model_name: str = "deberta-large", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model_dir = self._resolve_model_dir(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=False)
        self.model = AutoModel.from_pretrained(str(self.model_dir)).to(self.device).eval()

    @staticmethod
    def _resolve_model_dir(model_name: str) -> Path:
        root = Path(__file__).resolve().parents[2] / "tools" / "transformers"
        nested = root / model_name
        if nested.exists():
            return nested
        if (root / "config.json").exists():
            return root
        raise FileNotFoundError(f"Could not find local transformer weights for {model_name!r} under {root}")

    def extract(self, text: str) -> np.ndarray:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("text must not be empty")

        words = cleaned.split()
        inputs = self.tokenizer(
            [words],
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True).hidden_states
            hidden = torch.stack(outputs)[-4:].sum(dim=0)
            attention = inputs["attention_mask"][0].bool()
            valid = hidden[0][attention]
            if valid.shape[0] > 2:
                valid = valid[1:-1]
            pooled = valid.mean(dim=0)
        return pooled.detach().cpu().numpy().astype(np.float32)

