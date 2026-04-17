import unittest

import numpy as np


class _FakeTextExtractor:
    def extract(self, text):
        return np.ones((1024,), dtype=np.float32)


class _FakeMediaService:
    def has_audio_stream(self, media_path):
        return True

    def has_audible_signal(self, audio_path):
        return True

    def extract_audio(self, media_path, output_path):
        return output_path


class _FakeTranscriptionService:
    def transcribe(self, audio_path):
        return "asr transcript"


class _FakeAudioFeatureService:
    def extract(self, audio_path):
        return np.ones((512,), dtype=np.float32)


class _FakeVideoFeatureService:
    def extract(self, video_path):
        return np.ones((1024,), dtype=np.float32)


class _FakePrediction:
    def to_dict(self):
        return {
            "taskId": "placeholder",
            "usedModalities": ["audio", "text", "video"],
            "missingModalities": [],
            "emotionLabel": None,
            "sentimentPolarity": "negative",
            "score": -0.1,
            "confidence": 0.52,
            "message": "success",
            "error": None,
        }


class _FakePredictor:
    def predict_from_features(self, audio_feature=None, text_feature=None, video_feature=None, task_id=None, dataset=None):
        return _FakePrediction()


class AnalysisResultMetadataTests(unittest.TestCase):
    def test_result_includes_transcript_feature_status_raw_inputs_and_timing(self):
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

        self.assertEqual(result["transcript"], "asr transcript")
        self.assertEqual(result["featureStatus"]["audio"], "extracted")
        self.assertEqual(result["featureStatus"]["text"], "extracted_from_transcript")
        self.assertEqual(result["featureStatus"]["video"], "extracted")
        self.assertEqual(result["modelCondition"], "atv")
        self.assertEqual(result["rawInputs"]["videoFile"], "dataset/sims/0001.mp4")
        self.assertIsInstance(result["processingTimeMs"], int)
