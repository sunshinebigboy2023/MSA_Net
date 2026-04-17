from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from msa_service.domain.tasks import TASK_EXTRACTING, TASK_FAILED, TASK_INFERRING, TASK_PREPROCESSING
from msa_service.dao.task_dao import InMemoryTaskDao
from msa_service.service.audio_feature_service import FeatureExtractionUnavailable, Wav2VecAudioFeatureService
from msa_service.service.feature_service import TextFeatureExtractor
from msa_service.service.media_service import MediaService
from msa_service.service.predictor_service import MoMKEPredictorRegistry, condition_from_features
from msa_service.service.language_service import dataset_for_language, detect_text_language
from msa_service.service.transcription_service import WhisperTranscriptionService
from msa_service.service.video_feature_service import ManetVideoFeatureService


class AnalysisService:
    def __init__(self, checkpoint_path: str | None = None):
        self.predictor = MoMKEPredictorRegistry.from_checkpoint_source(checkpoint_path)
        self.task_dao = InMemoryTaskDao()
        self._text_extractor: Optional[TextFeatureExtractor] = None
        self._media_service: Optional[MediaService] = None
        self._transcription_service: Optional[WhisperTranscriptionService] = None
        self._audio_feature_service: Optional[Wav2VecAudioFeatureService] = None
        self._video_feature_service: Optional[ManetVideoFeatureService] = None

    @property
    def text_extractor(self) -> TextFeatureExtractor:
        if self._text_extractor is None:
            self._text_extractor = TextFeatureExtractor()
        return self._text_extractor

    @property
    def media_service(self) -> MediaService:
        if self._media_service is None:
            self._media_service = MediaService()
        return self._media_service

    @property
    def transcription_service(self) -> WhisperTranscriptionService:
        if self._transcription_service is None:
            self._transcription_service = WhisperTranscriptionService()
        return self._transcription_service

    @property
    def audio_feature_service(self) -> Wav2VecAudioFeatureService:
        if getattr(self, "_audio_feature_service", None) is None:
            self._audio_feature_service = Wav2VecAudioFeatureService()
        return self._audio_feature_service

    @property
    def video_feature_service(self) -> ManetVideoFeatureService:
        if getattr(self, "_video_feature_service", None) is None:
            self._video_feature_service = ManetVideoFeatureService()
        return self._video_feature_service

    def submit(self, payload: Dict[str, Any]):
        return self.task_dao.create(payload)

    def run_task(self, task_id: str) -> Dict[str, Any]:
        task = self.task_dao.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")

        started_at = time.perf_counter()
        try:
            self.task_dao.set_status(task_id, TASK_PREPROCESSING)
            features, metadata = self._resolve_features(task_id, task.payload)
            self.task_dao.set_status(task_id, TASK_INFERRING)
            result = self.predictor.predict_from_features(
                audio_feature=features.get("audio"),
                text_feature=features.get("text"),
                video_feature=features.get("video"),
                task_id=task_id,
                dataset=metadata.get("modelDataset"),
            ).to_dict()
            result.update(metadata)
            result["modelCondition"] = condition_from_features(
                audio_feature=features.get("audio"),
                text_feature=features.get("text"),
                video_feature=features.get("video"),
            )
            result["rawInputs"] = self._raw_inputs(task.payload)
            result["processingTimeMs"] = int((time.perf_counter() - started_at) * 1000)
            self.task_dao.set_result(task_id, result)
            return result
        except Exception as exc:
            self.task_dao.set_status(task_id, TASK_FAILED, error=str(exc))
            raise

    def _resolve_features(self, task_id: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Any]]:
        features: Dict[str, Optional[np.ndarray]] = {"audio": None, "text": None, "video": None}
        metadata: Dict[str, Any] = {
            "transcript": None,
            "language": "unknown",
            "modelDataset": "CMUMOSI",
            "featureStatus": {"audio": "missing", "text": "missing", "video": "missing"},
        }
        audio_file = self._resolve_audio_source(task_id, payload)
        language_text: Optional[str] = None

        text = payload.get("text")
        if text:
            language_text = text
            self.task_dao.set_status(task_id, TASK_EXTRACTING)
            features["text"] = self.text_extractor.extract(text)
            metadata["featureStatus"]["text"] = "extracted"
        elif payload.get("textFeaturePath"):
            features["text"] = np.load(payload["textFeaturePath"]).astype(np.float32).reshape(-1)
            metadata["featureStatus"]["text"] = "provided"
        elif audio_file:
            transcript = self._transcribe_audio(task_id, audio_file)
            metadata["transcript"] = transcript
            language_text = transcript
            self.task_dao.set_status(task_id, TASK_EXTRACTING)
            features["text"] = self.text_extractor.extract(transcript)
            metadata["featureStatus"]["text"] = "extracted_from_transcript"

        if payload.get("audioFeaturePath"):
            features["audio"] = np.load(payload["audioFeaturePath"]).astype(np.float32).reshape(-1)
            metadata["featureStatus"]["audio"] = "provided"
        elif audio_file:
            features["audio"] = self._safe_extract_audio_feature(audio_file)
            metadata["featureStatus"]["audio"] = "extracted" if features["audio"] is not None else "unavailable"

        if payload.get("videoFeaturePath"):
            features["video"] = np.load(payload["videoFeaturePath"]).astype(np.float32).reshape(-1)
            metadata["featureStatus"]["video"] = "provided"
        elif payload.get("videoFile"):
            features["video"] = self._safe_extract_video_feature(payload["videoFile"])
            metadata["featureStatus"]["video"] = "extracted" if features["video"] is not None else "unavailable"

        if not any(value is not None for value in features.values()):
            raise ValueError("At least one modality input is required")

        language = self._resolve_language(payload, language_text)
        metadata["language"] = language
        metadata["modelDataset"] = dataset_for_language(language)

        return features, metadata

    @staticmethod
    def _resolve_language(payload: Dict[str, Any], text: Optional[str]) -> str:
        requested = str(payload.get("language") or "").strip().lower()
        if requested in {"zh", "cn", "chinese", "中文", "sims"}:
            return "zh"
        if requested in {"en", "english", "英文", "cmumosi", "mosi"}:
            return "en"
        return detect_text_language(text)

    def _resolve_audio_source(self, task_id: str, payload: Dict[str, Any]) -> Optional[str]:
        audio_file = payload.get("audioFile")
        video_file = payload.get("videoFile")
        if audio_file:
            return audio_file

        self.task_dao.set_status(task_id, TASK_PREPROCESSING)
        if video_file:
            try:
                media_service = self.media_service
                if hasattr(media_service, "has_audio_stream") and not media_service.has_audio_stream(video_file):
                    return None
                audio_file = str(Path("temp") / task_id / "audio.wav")
                media_service.extract_audio(video_file, audio_file)
                if hasattr(media_service, "has_audible_signal") and not media_service.has_audible_signal(audio_file):
                    return None
                return audio_file
            except (FileNotFoundError, RuntimeError):
                return None
        return None

    def _transcribe_audio(self, task_id: str, audio_file: str) -> str:
        self.task_dao.set_status(task_id, TASK_EXTRACTING)
        transcript = self.transcription_service.transcribe(audio_file)
        if not transcript:
            raise ValueError("Whisper transcription returned empty text")
        return transcript

    def _safe_extract_audio_feature(self, audio_file: str) -> Optional[np.ndarray]:
        try:
            return self.audio_feature_service.extract(audio_file)
        except FeatureExtractionUnavailable:
            return None

    def _safe_extract_video_feature(self, video_file: str) -> Optional[np.ndarray]:
        try:
            return self.video_feature_service.extract(video_file)
        except FeatureExtractionUnavailable:
            return None

    def get_task(self, task_id: str):
        return self.task_dao.get(task_id)

    def get_result(self, task_id: str):
        task = self.task_dao.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        return task.result

    @staticmethod
    def _raw_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("text", "audioFile", "videoFile", "audioFeaturePath", "textFeaturePath", "videoFeaturePath")
        return {key: payload.get(key) for key in keys if payload.get(key)}


def dump_result_json(result: Dict[str, Any], output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
