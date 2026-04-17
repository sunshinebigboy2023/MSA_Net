import unittest

import numpy as np


class _FakeTextExtractor:
    def __init__(self):
        self.seen = []

    def extract(self, text):
        self.seen.append(text)
        return np.ones((1024,), dtype=np.float32)


class _FakeMediaService:
    def __init__(self):
        self.calls = []

    def extract_audio(self, video_path, output_path):
        self.calls.append((video_path, output_path))
        return output_path


class _FakeTranscriptionService:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path):
        self.calls.append(audio_path)
        return "transcribed text"


class _FakeAudioFeatureService:
    def extract(self, audio_path):
        return None


class _FakeVideoFeatureService:
    def extract(self, video_path):
        return None


class _FakePrediction:
    def to_dict(self):
        return {"message": "success", "usedModalities": ["text"]}


class _FakePredictor:
    def __init__(self):
        self.text_feature = None

    def predict_from_features(self, audio_feature=None, text_feature=None, video_feature=None, task_id=None, dataset=None):
        self.text_feature = text_feature
        return _FakePrediction()


class AnalysisServiceMediaTests(unittest.TestCase):
    def test_video_file_uses_whisper_transcript_as_text_feature(self):
        from msa_service.dao.task_dao import InMemoryTaskDao
        from msa_service.service.analysis_service import AnalysisService

        service = AnalysisService.__new__(AnalysisService)
        service.predictor = _FakePredictor()
        service.task_dao = InMemoryTaskDao()
        service._text_extractor = _FakeTextExtractor()
        service._media_service = _FakeMediaService()
        service._transcription_service = _FakeTranscriptionService()
        service._audio_feature_service = _FakeAudioFeatureService()
        service._video_feature_service = _FakeVideoFeatureService()

        task = service.submit({"videoFile": "dataset/sims/0001.mp4"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["message"], "success")
        self.assertEqual(service._text_extractor.seen, ["transcribed text"])
        self.assertEqual(len(service._media_service.calls), 1)
        self.assertEqual(len(service._transcription_service.calls), 1)
        self.assertIsNotNone(service.predictor.text_feature)
