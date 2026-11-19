#!/bin/bash
# GPU smoke test of every Demo 3 component (screen, gated loop, PixArt, Playground, batched judge), ~10 min.
#   sbatch slurm/demo3_smoke.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpul40s|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -J gd3-smoke
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi -L
python scripts/check_env.py --strict
python scripts/demo3_smoke.py
