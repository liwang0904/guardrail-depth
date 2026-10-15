"""Model registry (configs/models.yaml) and text-to-image pipeline loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import yaml

from .paths import CONFIG_DIR


@lru_cache(maxsize=1)
def registry() -> dict:
    return yaml.safe_load(open(CONFIG_DIR / "models.yaml"))


@dataclass
class T2ISpec:
    key: str
    hf_id: str
    revision: str
    family: str
    variant: str | None
    resolution: int
    scheduler: str
    steps: int
    guidance: float


def t2i_spec(key: str) -> T2ISpec:
    cfg = registry()["text_to_image"][key]
    s = cfg["settings"]
    return T2ISpec(key, cfg["hf_id"], cfg["revision"], cfg["family"], cfg.get("variant"),
                   cfg["resolution"], s["scheduler"], s["steps"], s["guidance"])


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # fp16 everywhere a GPU is present: the 2080 Ti on Markov has no bf16, and the
    # SD1/SD2 VAEs are stable in fp16. CPU runs (tests) stay in fp32.
    return torch.float32 if device.type == "cpu" else torch.float16


def load_t2i(key: str, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Load a diffusers pipeline for a registry key. Returns (pipe, spec)."""
    from diffusers import DDIMScheduler, DiffusionPipeline

    spec = t2i_spec(key)
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    kwargs = dict(revision=spec.revision, torch_dtype=dtype)
    if spec.variant and dtype == torch.float16:
        kwargs["variant"] = spec.variant
    if spec.family in ("sd1", "sd2"):
        # Copyright research needs the raw output: prior work (Wen et al., NeMo, Pham et al.)
        # disables the NSFW checker, which otherwise blacks out unrelated images.
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    pipe = DiffusionPipeline.from_pretrained(spec.hf_id, **kwargs)
    if spec.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, spec
