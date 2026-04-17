from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class OpenFaceService:
    def __init__(self, executable_path: Optional[str] = None):
        self.executable_path = Path(executable_path) if executable_path else self._resolve_executable()

    def extract_aligned_faces(self, video_path: str, output_dir: str) -> Path:
        source = Path(video_path)
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.executable_path),
            "-f",
            str(source),
            "-out_dir",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)

        aligned_dir = output / f"{source.stem}_aligned"
        has_faces = aligned_dir.exists() and any(
            [
                *aligned_dir.glob("*.bmp"),
                *aligned_dir.glob("*.jpg"),
                *aligned_dir.glob("*.png"),
            ]
        )
        if completed.returncode != 0 and not has_faces:
            raise RuntimeError(f"OpenFace failed: {completed.stderr.strip() or completed.stdout.strip()}")
        if not aligned_dir.exists():
            raise RuntimeError(f"OpenFace did not create aligned face directory: {aligned_dir}")
        return aligned_dir

    @staticmethod
    def _resolve_executable() -> Path:
        executable = Path(__file__).resolve().parents[2] / "tools" / "openface" / "FeatureExtraction.exe"
        if not executable.exists():
            raise FileNotFoundError(f"OpenFace FeatureExtraction.exe not found: {executable}")
        return executable
