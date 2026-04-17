import os
import unittest


class VideoFeatureServiceTests(unittest.TestCase):
    def test_extract_video_feature_returns_1024_dim_embedding(self):
        from msa_service.service.video_feature_service import ManetVideoFeatureService

        video_path = os.path.join(os.getcwd(), "dataset", "sims", "0001.mp4")
        if not os.path.exists(video_path):
            self.skipTest("dataset/sims/0001.mp4 is not available")

        feature = ManetVideoFeatureService(batch_size=8).extract(video_path)

        self.assertEqual(feature.shape, (1024,))
