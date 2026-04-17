import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class OpenFaceServiceTests(unittest.TestCase):
    def test_returns_aligned_dir_when_openface_nonzero_but_faces_exist(self):
        from msa_service.service.openface_service import OpenFaceService

        class Completed:
            returncode = 1
            stderr = "OpenFace returned a warning-like nonzero code"
            stdout = ""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "sample.mp4"
            video.write_bytes(b"fake video")
            output = root / "openface"
            aligned = output / "sample_aligned"
            aligned.mkdir(parents=True)
            (aligned / "frame_det_00_000001.bmp").write_bytes(b"fake image")

            with patch("msa_service.service.openface_service.subprocess.run", return_value=Completed()):
                service = OpenFaceService(executable_path=str(root / "FeatureExtraction.exe"))
                result = service.extract_aligned_faces(str(video), str(output))

        self.assertEqual(result, aligned)

