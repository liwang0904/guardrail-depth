"""Pre-download model weights (pinned revisions) into the Hugging Face cache.

    python scripts/download_models.py --list                      # sizes, no download
    python scripts/download_models.py sd14 clip_eval_small sscd   # laptop pilot (~3.7 GB)
    python scripts/download_models.py sd14 sdxl playground25 judge_vlm rewriter_reasoning clip_eval sscd

Only the fp16 diffusers components are fetched for text-to-image models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guardrail_depth.models import registry  # noqa: E402

T2I_PATTERNS = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "feature_extractor/*",
                "*/config.json", "*/*.fp16.safetensors"]


def t2i_patterns(cfg: dict) -> list[str]:
    if cfg.get("variant") == "fp16":
        return T2I_PATTERNS
    return ["model_index.json", "*/*.json", "*/*.txt", "*/*.model", "*.safetensors"]


def remote_size(repo: str, revision: str, patterns: list[str]) -> float:
    from fnmatch import fnmatch

    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    total = sum(s.size or 0 for s in info.siblings if any(fnmatch(s.rfilename, p) for p in patterns))
    return total / 1e9


def main() -> None:
    reg = registry()
    keys = list(reg["text_to_image"]) + list(reg["auxiliary"])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help=f"any of: {', '.join(keys)}")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    unknown = set(args.models) - set(keys)
    if unknown:
        ap.error(f"unknown model key(s) {sorted(unknown)}; choose from {keys}")

    from huggingface_hub import snapshot_download

    for key in (keys if args.list else args.models):
        cfg = reg["text_to_image"].get(key) or reg["auxiliary"][key]
        if "hf_id" in cfg:
            patterns = t2i_patterns(cfg) if key in reg["text_to_image"] else ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
            if args.list:
                print(f"{key:26s} {cfg['hf_id']:48s} {remote_size(cfg['hf_id'], cfg['revision'], patterns):6.1f} GB  min_gpu={cfg.get('min_gpu','-')}")
                continue
            print(f"downloading {key}: {cfg['hf_id']}@{cfg['revision'][:10]}")
            path = snapshot_download(cfg["hf_id"], revision=cfg["revision"], allow_patterns=patterns)
            print(f"  -> {path}")
        elif "open_clip" in cfg:
            oc = cfg["open_clip"]
            if args.list:
                print(f"{key:26s} open_clip {oc['model']}/{oc['pretrained']:31s} {cfg['download_gb']:6.1f} GB")
                continue
            import open_clip

            print(f"downloading {key}: open_clip {oc['model']}/{oc['pretrained']}")
            open_clip.create_model_and_transforms(oc["model"], pretrained=oc["pretrained"])
        elif "url" in cfg:
            if args.list:
                print(f"{key:26s} {cfg['url'][:48]:48s} {cfg['bytes'] / 1e9:6.1f} GB")
                continue
            from guardrail_depth.metrics import SSCD

            SSCD("cpu")
            print("  sscd ok")


if __name__ == "__main__":
    main()
