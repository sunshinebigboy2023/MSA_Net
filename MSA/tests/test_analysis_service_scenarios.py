import unittest

import numpy as np


class _FakeTextExtractor:
    def __init__(self):
        self.seen = []

    def extract(self, text):
        self.seen.append(text)
        return np.ones((1024,), dtype=np.float32)


class _FakeMediaService:
    def __init__(self, has_audio=True):
        self.has_audio = has_audio
        self.extract_calls = []

    def has_audio_stream(self, media_path):
        return self.has_audio

    def extract_audio(self, media_path, output_path):
        self.extract_calls.append((media_path, output_path))
        return output_path


class _UnavailableMediaService:
    def has_audio_stream(self, media_path):
        raise FileNotFoundError("ffmpeg executable not found")

    def extract_audio(self, media_path, output_path):
        raise AssertionError("extract_audio should not be called when ffmpeg is unavailable")


class _FakeTranscriptionService:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path):
        self.calls.append(audio_path)
        return "asr text"


class _FakeAudioFeatureService:
    def __init__(self):
        self.calls = []

    def extract(self, audio_path):
        self.calls.append(audio_path)
        return np.full((512,), 2.0, dtype=np.float32)


class _FakeVideoFeatureService:
    def __init__(self):
        self.calls = []

    def extract(self, video_path):
        self.calls.append(video_path)
        return np.full((1024,), 3.0, dtype=np.float32)


class _FakePrediction:
    def __init__(self, used):
        self.used = used

    def to_dict(self):
        return {"message": "success", "usedModalities": self.used}


class _FakePredictor:
    def __init__(self):
        self.calls = []

    def predict_from_features(self, audio_feature=None, text_feature=None, video_feature=None, task_id=None, dataset=None):
        used = []
        if audio_feature is not None:
            used.append("audio")
        if text_feature is not None:
            used.append("text")
        if video_feature is not None:
            used.append("video")
        self.calls.append((audio_feature, text_feature, video_feature))
        return _FakePrediction(used)


def _build_service(has_audio=True):
    from msa_service.dao.task_dao import InMemoryTaskDao
    from msa_service.service.analysis_service import AnalysisService

    service = AnalysisService.__new__(AnalysisService)
    service.predictor = _FakePredictor()
    service.task_dao = InMemoryTaskDao()
    service._text_extractor = _FakeTextExtractor()
    service._media_service = _FakeMediaService(has_audio=has_audio)
    service._transcription_service = _FakeTranscriptionService()
    service._audio_feature_service = _FakeAudioFeatureService()
    service._video_feature_service = _FakeVideoFeatureService()
    return service


class AnalysisServiceScenarioTests(unittest.TestCase):
    def test_text_only_uses_text_modality(self):
        service = _build_service()

        task = service.submit({"text": "manual text"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["usedModalities"], ["text"])
        self.assertEqual(service._text_extractor.seen, ["manual text"])
        self.assertEqual(service._transcription_service.calls, [])

    def test_audio_only_uses_audio_and_asr_text(self):
        service = _build_service()

        task = service.submit({"audioFile": "sample.wav"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["usedModalities"], ["audio", "text"])
        self.assertEqual(service._audio_feature_service.calls, ["sample.wav"])
        self.assertEqual(service._transcription_service.calls, ["sample.wav"])

    def test_video_with_audio_uses_extracted_audio_and_asr_text(self):
        service = _build_service(has_audio=True)

        task = service.submit({"videoFile": "sample.mp4"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["usedModalities"], ["audio", "text", "video"])
        self.assertEqual(len(service._media_service.extract_calls), 1)
        self.assertEqual(len(service._audio_feature_service.calls), 1)
        self.assertEqual(len(service._transcription_service.calls), 1)
        self.assertEqual(service._video_feature_service.calls, ["sample.mp4"])

    def test_video_without_audio_uses_video_only(self):
        service = _build_service(has_audio=False)

        task = service.submit({"videoFile": "silent.mp4"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["usedModalities"], ["video"])
        self.assertEqual(service._media_service.extract_calls, [])
        self.assertEqual(service._audio_feature_service.calls, [])
        self.assertEqual(service._transcription_service.calls, [])
        self.assertEqual(service._video_feature_service.calls, ["silent.mp4"])

    def test_video_with_manual_text_prefers_manual_text_over_asr(self):
        service = _build_service(has_audio=True)

        task = service.submit({"videoFile": "sample.mp4", "text": "user provided text"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["usedModalities"], ["audio", "text", "video"])
        self.assertEqual(service._text_extractor.seen, ["user provided text"])
        self.assertEqual(service._transcription_service.calls, [])
        self.assertEqual(len(service._audio_feature_service.calls), 1)

    def test_video_with_unavailable_ffmpeg_still_uses_text_and_video(self):
        service = _build_service(has_audio=True)
        service._media_service = _UnavailableMediaService()

        task = service.submit({"videoFile": "sample.mp4", "text": "user provided text"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["usedModalities"], ["text", "video"])
        self.assertEqual(service._text_extractor.seen, ["user provided text"])
        self.assertEqual(service._transcription_service.calls, [])
        self.assertEqual(service._audio_feature_service.calls, [])
        self.assertEqual(service._video_feature_service.calls, ["sample.mp4"])
