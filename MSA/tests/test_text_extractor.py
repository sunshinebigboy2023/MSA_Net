import unittest


class TextExtractorTests(unittest.TestCase):
    def test_extract_text_feature_returns_expected_dim(self):
        from msa_service.service.feature_service import TextFeatureExtractor

        extractor = TextFeatureExtractor()
        feature = extractor.extract("this movie was surprisingly good")

        self.assertEqual(feature.shape, (1024,))
