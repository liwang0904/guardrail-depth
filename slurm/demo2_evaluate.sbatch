#!/bin/bash
# Score every Demo 2 run (SSCD vs. training image, CLIP adherence / attribution, diversity), then build
# the tables and figures that get committed to results/demo2/.
#   sbatch slurm/demo2_evaluate.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu2080|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J gd-eval
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
python scripts/evaluate.py --clip clip_eval "$GD_OUTPUT_ROOT"/demo2/*/
python scripts/demo2_report.py
