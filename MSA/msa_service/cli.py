from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from msa_service.service.analysis_service import AnalysisService, dump_result_json


def _default_checkpoint_source():
    models_dir = Path(os.getcwd()) / "models"
    matches = glob.glob(str(models_dir / "**" / "*test-condition-*.pth"), recursive=True)
    if matches:
        return str(models_dir)
    matches = glob.glob(os.path.join(os.getcwd(), "*CMUMOSI*.pth"))
    if matches:
        return matches[0]
    raise FileNotFoundError("No CMUMOSI checkpoint found. Please pass --checkpoint explicitly.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone MoMKE CMUMOSI inference CLI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a MoMKE checkpoint or a directory containing condition checkpoints")
    parser.add_argument("--text", type=str, default=None, help="Raw text input")
    parser.add_argument("--text-feature-path", type=str, default=None, help="Path to extracted text feature .npy")
    parser.add_argument("--audio-feature-path", type=str, default=None, help="Path to extracted audio feature .npy")
    parser.add_argument("--video-feature-path", type=str, default=None, help="Path to extracted video feature .npy")
    parser.add_argument("--audio-file", type=str, default=None, help="Raw audio file path")
    parser.add_argument("--video-file", type=str, default=None, help="Raw video file path")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    service = AnalysisService(args.checkpoint or _default_checkpoint_source())
    task = service.submit(
        {
            "text": args.text,
            "textFeaturePath": args.text_feature_path,
            "audioFeaturePath": args.audio_feature_path,
            "videoFeaturePath": args.video_feature_path,
            "audioFile": args.audio_file,
            "videoFile": args.video_file,
        }
    )
    result = service.run_task(task.task_id)
    if args.output:
        dump_result_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
