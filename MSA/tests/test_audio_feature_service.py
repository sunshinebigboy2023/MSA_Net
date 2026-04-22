import os
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch


class AudioFeatureServiceTests(unittest.TestCase):
    def test_extract_resamples_non_16khz_audio(self):
        from msa_service.service.audio_feature_service import Wav2VecAudioFeatureService

        class FakeModel:
            def __init__(self):
                self.input_shape = None

            def feature_extractor(self, tensor):
                self.input_shape = tuple(tensor.shape)
                return torch.ones((1, 512, 4), dtype=torch.float32)

            def feature_aggregator(self, features):
                return features

        class ResamplingService(Wav2VecAudioFeatureService):
            def __init__(self):
                self.device = torch.device("cpu")
                self.fake_model = FakeModel()

            def _load_model(self):
                return self.fake_model

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "sample.wav")
            sf.write(audio_path, np.ones((8000,), dtype=np.float32), 8000)

            service = ResamplingService()
            feature = service.extract(audio_path)

        self.assertEqual(feature.shape, (512,))
        self.assertEqual(service.fake_model.input_shape, (1, 16000))

    def test_extract_audio_feature_returns_512_dim_embedding(self):
        from msa_service.service.audio_feature_service import Wav2VecAudioFeatureService
        from msa_service.service.media_service import MediaService

        audio_path = os.path.join(os.getcwd(), "temp", "sims_smoke", "audio.wav")
        if not os.path.exists(audio_path):
            video_path = os.path.join(os.getcwd(), "dataset", "sims", "0001.mp4")
            if not os.path.exists(video_path):
                self.skipTest("dataset/sims/0001.mp4 is not available")
            MediaService().extract_audio(video_path, audio_path)

        feature = Wav2VecAudioFeatureService().extract(audio_path)

        self.assertEqual(feature.shape, (512,))
