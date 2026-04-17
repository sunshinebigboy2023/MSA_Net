# *_*coding:utf-8 *_*
import os
import socket
from pathlib import Path


def get_host_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.0.0.1", 8080))
        ip = sock.getsockname()[0]
    finally:
        sock.close()
    return ip


ROOT_DIR = Path(__file__).resolve().parent


def _first_existing_dir(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_DATASET_ROOTS = {
    "CMUMOSI": _first_existing_dir(ROOT_DIR / "dataset" / "CMUMOSI", ROOT_DIR / "GCNet" / "dataset" / "CMUMOSI"),
    "CMUMOSEI": _first_existing_dir(ROOT_DIR / "dataset" / "CMUMOSEI", ROOT_DIR / "GCNet" / "dataset" / "CMUMOSEI"),
    "SIMS": _first_existing_dir(ROOT_DIR / "dataset" / "SIMS", ROOT_DIR / "GCNet" / "dataset" / "SIMS"),
}

PATH_TO_RAW_AUDIO = {
    name: str(root / "subaudio")
    for name, root in _DATASET_ROOTS.items()
}

PATH_TO_RAW_FACE = {
    "CMUMOSI": str(_DATASET_ROOTS["CMUMOSI"] / "openface_face"),
    "CMUMOSEI": str(_DATASET_ROOTS["CMUMOSEI"] / "openface_face"),
    "SIMS": str(_DATASET_ROOTS["SIMS"] / "openface_face"),
}

PATH_TO_TRANSCRIPTIONS = {
    name: str(root / "transcription.csv")
    for name, root in _DATASET_ROOTS.items()
}

PATH_TO_FEATURES = {
    name: str(root / "features")
    for name, root in _DATASET_ROOTS.items()
}

PATH_TO_LABEL = {
    "CMUMOSI": str(_first_existing_path(ROOT_DIR / "dataset" / "CMUMOSI" / "CMUMOSI_features_raw_2way.pkl", ROOT_DIR / "GCNet" / "dataset" / "CMUMOSI" / "CMUMOSI_features_raw_2way.pkl")),
    "CMUMOSEI": str(_first_existing_path(ROOT_DIR / "dataset" / "CMUMOSEI" / "CMUMOSEI_features_raw_2way.pkl", ROOT_DIR / "GCNet" / "dataset" / "CMUMOSEI" / "CMUMOSEI_features_raw_2way.pkl")),
    "SIMS": str(_first_existing_path(ROOT_DIR / "dataset" / "SIMS" / "label.csv", ROOT_DIR / "GCNet" / "dataset" / "SIMS" / "label.csv")),
}

PATH_TO_PRETRAINED_MODELS = str(ROOT_DIR / "tools")
PATH_TO_OPENSMILE = str(ROOT_DIR / "tools" / "opensmile-2.3.0")
PATH_TO_FFMPEG = str(_first_existing_path(
    ROOT_DIR / "tools" / "ffmpeg-4.4.1-i686-static" / "ffmpeg",
    ROOT_DIR / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ROOT_DIR / "tools" / "ffmpeg.exe",
))

SAVED_ROOT = str(ROOT_DIR / "saved")
DATA_DIR = str(Path(SAVED_ROOT) / "data")
MODEL_DIR = str(Path(SAVED_ROOT) / "model")
LOG_DIR = str(Path(SAVED_ROOT) / "log")
NPZ_DIR = str(Path(SAVED_ROOT) / "npz")

_SAVE_DIRS = {
    "data": DATA_DIR,
    "model": MODEL_DIR,
    "log": LOG_DIR,
    "npz": NPZ_DIR,
}


def get_save_dir(kind: str) -> str:
    if kind not in _SAVE_DIRS:
        raise KeyError(f"Unsupported save dir kind: {kind}")
    return _SAVE_DIRS[kind]
