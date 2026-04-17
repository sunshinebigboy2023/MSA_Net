from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class FeatureDirs:
    audio: Path
    text: Path
    video: Path


@dataclass(frozen=True)
class SimsLabel:
    text: str
    label: Optional[float]
    split: str
    video_id: str
    clip_id: str


def feature_dirs(output_root: Path) -> FeatureDirs:
    return FeatureDirs(
        audio=output_root / "wav2vec-large-c-UTT",
        text=output_root / "deberta-large-4-UTT",
        video=output_root / "manet_UTT",
    )


def iter_sims_videos(dataset_root: Path) -> Iterable[Path]:
    raw_root = dataset_root / "Raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"SIMS Raw directory not found: {raw_root}")
    videos = []
    for path in raw_root.glob("video_*/*.mp4"):
        if path.name.startswith("._"):
            continue
        videos.append(path)
    return sorted(videos, key=lambda item: (item.parent.name, item.name))


def sample_id_for_video(video_path: Path) -> str:
    return f"{video_path.parent.name}_{video_path.stem}"


def sample_id_for_label(video_id: str, clip_id: str) -> str:
    clip = str(clip_id).strip()
    try:
        clip = f"{int(clip):04d}"
    except ValueError:
        clip = clip.zfill(4)
    return f"{video_id.strip()}_{clip}"


def load_sims_labels(dataset_root: Path) -> dict[str, SimsLabel]:
    label_path = dataset_root / "label.csv"
    if not label_path.exists():
        return {}
    labels: dict[str, SimsLabel] = {}
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            clip_id = (row.get("clip_id") or "").strip()
            if not video_id or not clip_id:
                continue
            raw_label = (row.get("label") or "").strip()
            try:
                label = float(raw_label) if raw_label else None
            except ValueError:
                label = None
            labels[sample_id_for_label(video_id, clip_id)] = SimsLabel(
                text=(row.get("text") or "").strip(),
                label=label,
                split=(row.get("split") or "").strip(),
                video_id=video_id,
                clip_id=clip_id,
            )
    return labels


def _save_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(feature, dtype=np.float32).reshape(-1))


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class SimsFeatureBatchExtractor:
    def __init__(
        self,
        dataset_root: Path,
        output_root: Path,
        temp_root: Path,
        modalities: set[str],
        force: bool = False,
    ):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.temp_root = temp_root
        self.modalities = modalities
        self.force = force
        self.dirs = feature_dirs(output_root)
        self.labels = load_sims_labels(dataset_root)
        self.metadata_path = output_root / "metadata.jsonl"
        self.errors_path = output_root / "errors.jsonl"
        self.transcripts_dir = output_root / "transcripts"
        self.texts_dir = output_root / "texts"
        self._media_service = None
        self._transcription_service = None
        self._text_extractor = None
        self._audio_feature_service = None
        self._video_feature_service = None

    @property
    def media_service(self):
        if self._media_service is None:
            from msa_service.service.media_service import MediaService

            self._media_service = MediaService()
        return self._media_service

    @property
    def transcription_service(self):
        if self._transcription_service is None:
            from msa_service.service.transcription_service import WhisperTranscriptionService

            self._transcription_service = WhisperTranscriptionService()
        return self._transcription_service

    @property
    def text_extractor(self):
        if self._text_extractor is None:
            from msa_service.service.feature_service import TextFeatureExtractor

            self._text_extractor = TextFeatureExtractor()
        return self._text_extractor

    @property
    def audio_feature_service(self):
        if self._audio_feature_service is None:
            from msa_service.service.audio_feature_service import Wav2VecAudioFeatureService

            self._audio_feature_service = Wav2VecAudioFeatureService()
        return self._audio_feature_service

    @property
    def video_feature_service(self):
        if self._video_feature_service is None:
            from msa_service.service.video_feature_service import ManetVideoFeatureService

            self._video_feature_service = ManetVideoFeatureService()
        return self._video_feature_service

    def prepare_dirs(self) -> None:
        for directory in (
            self.dirs.audio,
            self.dirs.text,
            self.dirs.video,
            self.transcripts_dir,
            self.texts_dir,
            self.temp_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def run(self, limit: Optional[int] = None, start_after: Optional[str] = None) -> dict:
        self.prepare_dirs()
        videos = list(iter_sims_videos(self.dataset_root))
        if start_after:
            videos = [video for video in videos if sample_id_for_video(video) > start_after]
        if limit is not None:
            videos = videos[:limit]

        summary = {"total": len(videos), "success": 0, "failed": 0, "skipped": 0}
        for index, video_path in enumerate(videos, start=1):
            sample_id = sample_id_for_video(video_path)
            print(f"[{index}/{len(videos)}] {sample_id}", flush=True)
            started_at = time.perf_counter()
            try:
                status = self.extract_one(video_path)
                if status.get("skipped"):
                    summary["skipped"] += 1
                else:
                    summary["success"] += 1
                status["durationMs"] = int((time.perf_counter() - started_at) * 1000)
                _append_jsonl(self.metadata_path, status)
            except Exception as exc:
                summary["failed"] += 1
                error = {
                    "sampleId": sample_id,
                    "videoPath": str(video_path),
                    "error": str(exc),
                    "durationMs": int((time.perf_counter() - started_at) * 1000),
                }
                _append_jsonl(self.errors_path, error)
                print(f"  failed: {exc}", flush=True)
        return summary

    def extract_one(self, video_path: Path) -> dict:
        sample_id = sample_id_for_video(video_path)
        audio_output = self.dirs.audio / f"{sample_id}.npy"
        text_output = self.dirs.text / f"{sample_id}.npy"
        video_output = self.dirs.video / f"{sample_id}.npy"
        transcript_output = self.transcripts_dir / f"{sample_id}.txt"
        text_record_output = self.texts_dir / f"{sample_id}.txt"
        label = self.labels.get(sample_id)
        label_text = label.text if label and label.text else None
        needed_audio = "audio" in self.modalities and (self.force or not audio_output.exists())
        needed_text = "text" in self.modalities and (self.force or not text_output.exists())
        needed_video = "video" in self.modalities and (self.force or not video_output.exists())

        if not any((needed_audio, needed_text, needed_video)):
            return {
                "sampleId": sample_id,
                "videoPath": str(video_path),
                "skipped": True,
                "features": self._feature_status(audio_output, text_output, video_output),
            }

        status = {
            "sampleId": sample_id,
            "videoPath": str(video_path),
            "skipped": False,
            "features": {},
        }
        if label:
            status["label"] = label.label
            status["split"] = label.split

        needs_audio_file = needed_audio or (needed_text and not label_text)
        audio_path = None
        if needs_audio_file:
            audio_path = self._extract_audio(video_path, sample_id)
        if needed_text:
            if label_text:
                text_record_output.write_text(label_text, encoding="utf-8")
                _save_feature(text_output, self.text_extractor.extract(label_text))
                status["textSource"] = "label.csv"
            else:
                if audio_path is None:
                    audio_path = self._extract_audio(video_path, sample_id)
                transcript = self.transcription_service.transcribe(str(audio_path))
                transcript_output.write_text(transcript, encoding="utf-8")
                if not transcript:
                    raise ValueError(f"empty transcript for {sample_id}")
                _save_feature(text_output, self.text_extractor.extract(transcript))
                status["textSource"] = "whisper"
        if needed_audio:
            if audio_path is None:
                audio_path = self._extract_audio(video_path, sample_id)
            _save_feature(audio_output, self.audio_feature_service.extract(str(audio_path)))

        if needed_video:
            openface_dir = self.temp_root / "openface" / sample_id
            aligned_dir = self.video_feature_service.openface_service.extract_aligned_faces(
                str(video_path),
                str(openface_dir),
            )
            _save_feature(video_output, self.video_feature_service.extract_from_aligned_dir(aligned_dir))

        status["features"] = self._feature_status(audio_output, text_output, video_output)
        if transcript_output.exists():
            status["transcriptPath"] = str(transcript_output)
        if text_record_output.exists():
            status["textPath"] = str(text_record_output)
        return status

    def _extract_audio(self, video_path: Path, sample_id: str) -> Path:
        audio_path = self.temp_root / "audio" / f"{sample_id}.wav"
        if audio_path.exists() and not self.force:
            return audio_path
        if not self.media_service.has_audio_stream(str(video_path)):
            raise ValueError(f"no audio stream for {sample_id}")
        self.media_service.extract_audio(str(video_path), str(audio_path))
        if not self.media_service.has_audible_signal(str(audio_path)):
            raise ValueError(f"silent audio for {sample_id}")
        return audio_path

    @staticmethod
    def _feature_status(audio_output: Path, text_output: Path, video_output: Path) -> dict:
        return {
            "audio": str(audio_output) if audio_output.exists() else None,
            "text": str(text_output) if text_output.exists() else None,
            "video": str(video_output) if video_output.exists() else None,
        }


def _parse_modalities(value: str) -> set[str]:
    if value == "all":
        return {"audio", "text", "video"}
    aliases = {"a": "audio", "t": "text", "v": "video"}
    modalities = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        modalities.add(aliases.get(item, item))
    valid = {"audio", "text", "video"}
    invalid = modalities - valid
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid modalities: {sorted(invalid)}")
    if not modalities:
        raise argparse.ArgumentTypeError("at least one modality is required")
    return modalities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract SIMS features into MOSI-like feature folders")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset") / "SIMS")
    parser.add_argument("--output-root", type=Path, default=Path("dataset") / "SIMS" / "features")
    parser.add_argument("--temp-root", type=Path, default=Path("temp") / "sims_feature_extraction")
    parser.add_argument("--modalities", type=_parse_modalities, default={"audio", "text", "video"})
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos")
    parser.add_argument("--start-after", type=str, default=None, help="Resume after a sample id, e.g. video_0001_0001")
    parser.add_argument("--force", action="store_true", help="Overwrite existing features")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extractor = SimsFeatureBatchExtractor(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        temp_root=args.temp_root,
        modalities=args.modalities,
        force=args.force,
    )
    summary = extractor.run(limit=args.limit, start_after=args.start_after)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
