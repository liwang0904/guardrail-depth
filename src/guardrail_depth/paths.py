"""Filesystem layout.

Small, versioned files (manifests, configs) live in the repo. Anything large
(raw downloads, ground-truth images, generated images, model weights) goes under
directories that can be redirected with environment variables, so on the Markov
cluster they land in course storage instead of the 25 GB home quota:

    GD_DATA_ROOT    raw downloads + ground-truth images   (default: <repo>/data)
    GD_OUTPUT_ROOT  generated images, logs, metrics        (default: <repo>/outputs)
    GD_CACHE        auxiliary weights such as SSCD         (default: $GD_DATA_ROOT/cache)
    HF_HOME         Hugging Face model cache (read by huggingface_hub itself)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"


def data_root() -> Path:
    return Path(os.environ.get("GD_DATA_ROOT", REPO_ROOT / "data"))


def raw_dir() -> Path:
    return data_root() / "raw"


def gt_image_dir() -> Path:
    return data_root() / "gt_images"


def output_root() -> Path:
    return Path(os.environ.get("GD_OUTPUT_ROOT", REPO_ROOT / "outputs"))


def cache_dir() -> Path:
    return Path(os.environ.get("GD_CACHE", data_root() / "cache"))
