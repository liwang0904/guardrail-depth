#!/bin/bash
# Score every Demo 3 run (SSCD / CLIP as in Demo 2), then build results/demo3/.
#   sbatch slurm/demo3_evaluate.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu2080|gpu4090|gpul40s"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J gd3-eval
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
python scripts/evaluate.py --clip clip_eval $(ls -d "$GD_OUTPUT_ROOT"/demo3/*/ | grep -v _dev_)
python scripts/demo3_report.py
