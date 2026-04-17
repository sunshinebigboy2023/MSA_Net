from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf


class MediaService:
    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or self._resolve_ffmpeg()
        self.ffprobe_path = str(Path(self.ffmpeg_path).with_name("ffprobe.exe"))

    def extract_audio(self, media_path: str, output_path: str, sample_rate: int = 16000) -> str:
        source = Path(media_path)
        if not source.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
        return str(output)

    def has_audio_stream(self, media_path: str) -> bool:
        ffprobe = self.ffprobe_path if Path(self.ffprobe_path).exists() else shutil.which("ffprobe")
        if not ffprobe:
            return True
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            media_path,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        return bool(completed.stdout.strip())

    @staticmethod
    def has_audible_signal(audio_path: str, min_rms: float = 1e-4) -> bool:
        try:
            audio, _ = sf.read(audio_path, dtype="float32")
        except RuntimeError:
            return False
        if audio.size == 0:
            return False
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms >= min_rms

    @classmethod
    def _resolve_ffmpeg(cls) -> str:
        candidates = cls._ffmpeg_candidates()
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate

        found = shutil.which("ffmpeg")
        if found:
            return found

        raise FileNotFoundError("ffmpeg executable not found. Install ffmpeg or set MSA_FFMPEG_PATH.")

    @staticmethod
    def _ffmpeg_candidates() -> Iterable[str]:
        env_path = os.environ.get("MSA_FFMPEG_PATH")
        if env_path:
            yield env_path

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            winget_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_root.exists():
                for path in winget_root.rglob("ffmpeg.exe"):
                    yield str(path)
