from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import WhisperForConditionalGeneration, WhisperProcessor


_LANGUAGE_PROMPTS = {
    "zh": "chinese",
    "cn": "chinese",
    "chinese": "chinese",
    "中文": "chinese",
    "en": "english",
    "english": "english",
    "英文": "english",
}


class WhisperTranscriptionService:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        language: Optional[str] = None,
        task: str = "transcribe",
        load_model: bool = True,
    ):
        self.model_dir = Path(model_dir) if model_dir else self._resolve_model_dir()
        self.device = torch.device(device)
        self.language = language
        self.task = task
        self.processor = None
        self.model = None
        if load_model:
            self._load()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        if self.processor is None or self.model is None:
            self._load()

        waveform, sample_rate = self._load_audio(audio_path)
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        language_prompt = self._language_prompt(language or self.language)
        generate_kwargs = {}
        if language_prompt:
            generate_kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                language=language_prompt,
                task=self.task,
            )

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **generate_kwargs)

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return text

    @staticmethod
    def _language_prompt(language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        return _LANGUAGE_PROMPTS.get(language.strip().lower())

    def _load(self):
        self.processor = WhisperProcessor.from_pretrained(str(self.model_dir))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(self.model_dir)).to(self.device).eval()

    @staticmethod
    def _load_audio(audio_path: str):
        source = Path(audio_path)
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = sf.read(str(source), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
            waveform = AF.resample(tensor, sample_rate, 16000).numpy()
            sample_rate = 16000
        return waveform, sample_rate

    @staticmethod
    def _resolve_model_dir() -> Path:
        env_path = os.environ.get("MSA_WHISPER_MODEL_DIR")
        if env_path:
            candidate = Path(env_path)
            if candidate.exists():
                return candidate

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        preferred = [
            tools_dir / "whisper-medium",
            tools_dir / "whisper medium",
            tools_dir / "whisiper medium",
            tools_dir / "whisper",
        ]
        for candidate in preferred:
            if (candidate / "config.json").exists():
                return candidate

        for candidate in tools_dir.iterdir():
            if (candidate / "config.json").exists():
                try:
                    if '"model_type": "whisper"' in (candidate / "config.json").read_text(encoding="utf-8"):
                        return candidate
                except OSError:
                    continue

        raise FileNotFoundError("Local Whisper model directory not found under tools/.")
