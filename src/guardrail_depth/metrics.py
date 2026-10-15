"""Output-level metrics.

  SSCD          copy detection (Pizzi et al. CVPR'22); generated vs. ground-truth training image,
                as in Somepalli et al. CVPR'23, Wen et al. ICLR'24 and NeMo. "Copy" if > 0.5.
  CLIPScorer    CLIP-I (image-image), CLIP-T / prompt adherence (image-text, Reissinger et al. 2025),
                and zero-shot attribution over a label set (cheap local proxy for the VLM judge).
  diversity     mean pairwise similarity across seeds (memorized prompts collapse to one image).
  kid           kernel inception distance on any feature set (Track 2 style shift vs. clean baseline).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .models import registry
from .paths import cache_dir

COPY_THRESHOLD = 0.5  # Somepalli et al. CVPR'23


def _download(url: str, dest: Path, expected_bytes: int | None = None) -> Path:
    if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if expected_bytes and tmp.stat().st_size != expected_bytes:
        raise IOError(f"{url}: got {tmp.stat().st_size} bytes, expected {expected_bytes}")
    tmp.rename(dest)
    return dest


class SSCD:
    """sscd_disc_mixup TorchScript model with NeMo's preprocessing (320x320, ImageNet norm)."""

    def __init__(self, device: torch.device | str = "cpu"):
        spec = registry()["auxiliary"]["sscd"]
        path = _download(spec["url"], cache_dir() / "sscd_disc_mixup.torchscript.pt", spec["bytes"])
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location="cpu").to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize([320, 320]), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, images: list[Image.Image], batch_size: int = 16) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.tf(im.convert("RGB")) for im in images[i:i + batch_size]]).to(self.device)
            feats.append(torch.nn.functional.normalize(self.model(x).float(), dim=1).cpu())
        return torch.cat(feats)


class CLIPScorer:
    def __init__(self, model: str = "ViT-L-14", pretrained: str = "openai", device: torch.device | str = "cpu"):
        import open_clip

        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.name = f"{model}/{pretrained}"

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image], batch_size: int = 16) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im.convert("RGB")) for im in images[i:i + batch_size]]).to(self.device)
            feats.append(torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=1).cpu())
        return torch.cat(feats)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        feats = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            feats.append(torch.nn.functional.normalize(self.model.encode_text(tok).float(), dim=1).cpu())
        return torch.cat(feats)

    def zero_shot(self, image_feats: torch.Tensor, labels: list[str], template: str = "a picture of {}") -> torch.Tensor:
        """Softmax over labels (CLIP logit scale 100)."""
        text = self.embed_texts([template.format(l) for l in labels])
        return (100.0 * image_feats @ text.T).softmax(dim=-1)


def paired_cosine(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return (a * b).sum(-1).numpy()


def diversity(feats: torch.Tensor) -> float:
    """Mean pairwise cosine similarity among generations of one prompt (higher = less diverse)."""
    n = feats.shape[0]
    if n < 2:
        return float("nan")
    sim = feats @ feats.T
    return float((sim.sum() - sim.diag().sum()) / (n * (n - 1)))


def kid(feats_a: np.ndarray, feats_b: np.ndarray, n_subsets: int = 50, subset_size: int = 100, seed: int = 0) -> tuple[float, float]:
    """Unbiased KID (polynomial kernel, Binkowski et al. 2018); returns (mean, std) over subsets."""
    rng = np.random.default_rng(seed)
    d = feats_a.shape[1]
    m = min(subset_size, len(feats_a), len(feats_b))
    vals = []
    for _ in range(n_subsets):
        x = feats_a[rng.choice(len(feats_a), m, replace=False)]
        y = feats_b[rng.choice(len(feats_b), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))
