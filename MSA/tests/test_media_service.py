import os
import tempfile
import unittest


class MediaServiceTests(unittest.TestCase):
    def test_extract_audio_from_video_creates_16k_wav(self):
        from msa_service.service.media_service import MediaService

        video_path = os.path.join(os.getcwd(), "dataset", "sims", "0001.mp4")
        if not os.path.exists(video_path):
            self.skipTest("dataset/sims/0001.mp4 is not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "audio.wav")

            wav_path = MediaService().extract_audio(video_path, output_path)

            self.assertEqual(wav_path, output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)
