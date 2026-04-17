import os
import unittest


class AudioFeatureServiceTests(unittest.TestCase):
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
