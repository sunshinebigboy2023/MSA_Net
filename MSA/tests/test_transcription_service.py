import unittest


class TranscriptionServiceTests(unittest.TestCase):
    def test_resolves_local_whisper_model_directory(self):
        from msa_service.service.transcription_service import WhisperTranscriptionService

        service = WhisperTranscriptionService(load_model=False)

        self.assertTrue(service.model_dir.exists())
        self.assertEqual(service.model_dir.name, "whisiper medium")
