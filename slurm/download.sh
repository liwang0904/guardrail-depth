#!/bin/bash
# Fetch data + model weights into course storage. Network-bound, not GPU-bound: run it on the
# login node (inside tmux/screen) or a CPU node:
#   bash slurm/download.sh            # data, ground truth, pilot models (~6 GB)
#   bash slurm/download.sh full       # + SDXL, Playground v2.5, judge VLM, reasoning LLM, ViT-g/14 (~70 GB)
set -euo pipefail
source slurm/env.sh
python scripts/prepare_data.py --gt-images sd1
python scripts/prepare_data.py --skip-raw --gt-images sd2
python scripts/download_models.py sd14 sd2base clip_eval_small sscd
if [ "${1:-}" = "full" ]; then
  python scripts/download_models.py sdxl playground25 judge_vlm rewriter_reasoning rewriter_llm clip_eval
fi
du -sh "$HF_HOME" "$GD_DATA_ROOT"
