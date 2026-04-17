import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DT_MSA_DIR = ROOT_DIR / "DT-MSA"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DT_MSA_DIR) not in sys.path:
    sys.path.insert(0, str(DT_MSA_DIR))


class DTMSASIMSConfigTests(unittest.TestCase):
    def test_config_exposes_sims_and_removes_iemocap(self):
        config = importlib.import_module("config")

        self.assertIn("SIMS", config.PATH_TO_FEATURES)
        self.assertIn("SIMS", config.PATH_TO_LABEL)
        self.assertNotIn("IEMOCAPFour", config.PATH_TO_FEATURES)
        self.assertNotIn("IEMOCAPSix", config.PATH_TO_FEATURES)
        self.assertNotIn("IEMOCAPFour", config.PATH_TO_LABEL)
        self.assertNotIn("IEMOCAPSix", config.PATH_TO_LABEL)


class SIMSDatasetTests(unittest.TestCase):
    def _write_feature(self, root: Path, sample_id: str, value: float, dim: int) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / f"{sample_id}.npy", np.full((dim,), value, dtype=np.float32))

    def test_sims_dataset_reads_label_csv_and_split_level_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "SIMS"
            feature_root = dataset_root / "features"
            audio_root = feature_root / "wav2vec-large-c-UTT"
            text_root = feature_root / "deberta-large-4-UTT"
            video_root = feature_root / "manet_UTT"
            dataset_root.mkdir(parents=True)
            (dataset_root / "label.csv").write_text(
                "\n".join(
                    [
                        "video_id,clip_id,text,col4,col5,col6,col7,label,split",
                        "video_0001,1,开心,1,1,1,1,1,train",
                        "video_0001,2,难过,-1,-1,-1,-1,-1,test",
                        "video_0002,1,一般,0,0,0,0,0,valid",
                    ]
                ),
                encoding="utf-8",
            )

            for sample_id, value in [
                ("video_0001_0001", 1.0),
                ("video_0001_0002", 2.0),
                ("video_0002_0001", 3.0),
            ]:
                self._write_feature(audio_root, sample_id, value, 2)
                self._write_feature(text_root, sample_id, value, 3)
                self._write_feature(video_root, sample_id, value, 4)

            dataloader_sims = importlib.import_module("dataloader_sims")
            dataset = dataloader_sims.SIMSDataset(
                label_path=str(dataset_root / "label.csv"),
                audio_root=str(audio_root),
                text_root=str(text_root),
                video_root=str(video_root),
            )

            self.assertEqual(dataset.trainVids, ["train_video_0001"])
            self.assertEqual(dataset.valVids, ["valid_video_0002"])
            self.assertEqual(dataset.testVids, ["test_video_0001"])
            self.assertEqual(dataset.get_featDim(), (2, 3, 4))

            audio, text, video, audio_guest, text_guest, video_guest, qmask, umask, labels, vid = dataset[0]
            self.assertEqual(vid, "train_video_0001")
            self.assertEqual(audio.shape, (1, 2))
            self.assertEqual(text.shape, (1, 3))
            self.assertEqual(video.shape, (1, 4))
            self.assertTrue(np.allclose(audio_guest.numpy(), 0))
            self.assertTrue(np.allclose(text_guest.numpy(), 0))
            self.assertTrue(np.allclose(video_guest.numpy(), 0))
            self.assertEqual(qmask.tolist(), [0.0])
            self.assertEqual(umask.tolist(), [1.0])
            self.assertEqual(labels.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
