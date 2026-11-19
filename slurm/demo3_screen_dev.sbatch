#!/bin/bash
# Dev-split evaluation of the concept screen (tuning only; the test split is never screened here):
# builds the concept cards, then screens every dev request without them (used) and with them (ablation).
#   sbatch slurm/demo3_screen_dev.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpul40s|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J gd3-screendev
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
CFG=configs/experiments/demo3.yaml
DEV=sd14_dev_copycat,sd14_dev_benign,sd14_dev_generic,sd14_dev_style
[ -f data/manifests/concept_cards.csv ] || python scripts/build_concept_cards.py
python scripts/run_screen.py --config $CFG --job $DEV --out "$GD_OUTPUT_ROOT/screens/dev_names_only.csv"
python scripts/run_screen.py --config $CFG --job $DEV --cards --out "$GD_OUTPUT_ROOT/screens/dev_cards.csv"
