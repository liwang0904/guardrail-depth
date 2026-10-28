#!/bin/bash
# Demo 2 downloads (network-bound, no GPU): run on the login node inside tmux, once.
#   bash slurm/demo2_download.sh
set -euo pipefail
source slurm/env.sh
python scripts/prepare_data.py                                         # source data, checksum-verified
python scripts/prepare_data.py --skip-raw --gt-images sd1 --gt-split test   # training images for SSCD
python scripts/prepare_data.py --skip-raw --gt-images sd2 --gt-split test
python scripts/download_models.py sd14 sd2base sdxl sscd clip_eval judge_vlm rewriter_reasoning
du -sh "$HF_HOME" "$GD_DATA_ROOT"
