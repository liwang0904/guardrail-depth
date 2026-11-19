#!/bin/bash
# Demo 3 downloads (network-bound, no GPU): run on the login node inside tmux, once, after demo2_download.sh.
#   bash slurm/demo3_download.sh
# Adds the two models outside Stable Diffusion (Playground v2.5, PixArt-alpha 512) and the CLIP ViT-L/14 the
# gate's output check uses. The generic near-miss manifest is committed (data/manifests/generic_prompts.csv).
set -euo pipefail
source slurm/env.sh
python scripts/download_models.py sd14 sd2base sdxl playground25 pixart512 sscd clip_eval clip_eval_small judge_vlm rewriter_reasoning
du -sh "$HF_HOME"
