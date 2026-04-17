import tempfile
import unittest
from pathlib import Path

import numpy as np


class SimsFeatureBatchTests(unittest.TestCase):
    def test_iter_sims_videos_skips_macos_resource_files(self):
        from msa_service.scripts.extract_sims_features import iter_sims_videos

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SIMS"
            video_dir = root / "Raw" / "video_0001"
            video_dir.mkdir(parents=True)
            real_video = video_dir / "0001.mp4"
            real_video.write_bytes(b"real")
            resource_video = video_dir / "._0001.mp4"
            resource_video.write_bytes(b"resource")

            videos = list(iter_sims_videos(root))

        self.assertEqual(videos, [real_video])

    def test_sample_id_combines_parent_and_clip_stem(self):
        from msa_service.scripts.extract_sims_features import sample_id_for_video

        path = Path("dataset/SIMS/Raw/video_0001/0002.mp4")

        self.assertEqual(sample_id_for_video(path), "video_0001_0002")

    def test_sample_id_from_label_zero_pads_clip_id(self):
        from msa_service.scripts.extract_sims_features import sample_id_for_label

        self.assertEqual(sample_id_for_label("video_0001", "2"), "video_0001_0002")

    def test_feature_dirs_match_mosi_names(self):
        from msa_service.scripts.extract_sims_features import feature_dirs

        dirs = feature_dirs(Path("dataset/SIMS/features"))

        self.assertEqual(dirs.audio.name, "wav2vec-large-c-UTT")
        self.assertEqual(dirs.text.name, "deberta-large-4-UTT")
        self.assertEqual(dirs.video.name, "manet_UTT")

    def test_load_sims_labels_maps_text_label_and_split(self):
        from msa_service.scripts.extract_sims_features import load_sims_labels

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "label.csv").write_text(
                "video_id,clip_id,text,col4,col5,col6,col7,label,split\n"
                "video_0001,2,你这是嫁入豪门啊！,1,1,0.8,1,1,train\n",
                encoding="utf-8",
            )

            labels = load_sims_labels(root)

        self.assertEqual(labels["video_0001_0002"].text, "你这是嫁入豪门啊！")
        self.assertEqual(labels["video_0001_0002"].label, 1.0)
        self.assertEqual(labels["video_0001_0002"].split, "train")

    def test_text_feature_uses_label_csv_without_whisper(self):
        from msa_service.scripts.extract_sims_features import SimsFeatureBatchExtractor

        class FakeTextExtractor:
            def __init__(self):
                self.seen = []

            def extract(self, text):
                self.seen.append(text)
                return np.ones((1024,), dtype=np.float32)

        class FailingTranscription:
            def transcribe(self, audio_path):
                raise AssertionError("Whisper should not be called when label text exists")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "SIMS"
            video_dir = dataset_root / "Raw" / "video_0001"
            video_dir.mkdir(parents=True)
            video_path = video_dir / "0002.mp4"
            video_path.write_bytes(b"fake video")
            (dataset_root / "label.csv").write_text(
                "video_id,clip_id,text,col4,col5,col6,col7,label,split\n"
                "video_0001,2,表格里的文本,1,1,1,1,1,train\n",
                encoding="utf-8",
            )
            extractor = SimsFeatureBatchExtractor(
                dataset_root=dataset_root,
                output_root=dataset_root / "features",
                temp_root=root / "temp",
                modalities={"text"},
                force=True,
            )
            extractor._text_extractor = FakeTextExtractor()
            extractor._transcription_service = FailingTranscription()
            extractor.prepare_dirs()

            status = extractor.extract_one(video_path)

            self.assertEqual(extractor._text_extractor.seen, ["表格里的文本"])
            self.assertEqual(status["textSource"], "label.csv")
            self.assertEqual(status["label"], 1.0)
            self.assertEqual(status["split"], "train")
            self.assertTrue((dataset_root / "features" / "deberta-large-4-UTT" / "video_0001_0002.npy").exists())
