#!/bin/bash
# Calibrate the gate on the dev split (memorization threshold per model, output-check threshold) and write
# configs/gate.yaml; every gated test job runs after this one.
#   sbatch slurm/demo3_calibrate.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu2080|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -J gd3-cal
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi -L
python scripts/run_generation.py --config configs/experiments/demo3.yaml --job sd14_dev_copycat,sd14_dev_benign,sd14_dev_generic
python scripts/calibrate_gate.py --models sd14,sd2base --check-runs \
    "$GD_OUTPUT_ROOT/demo3/sd14_dev_copycat" "$GD_OUTPUT_ROOT/demo3/sd14_dev_benign" "$GD_OUTPUT_ROOT/demo3/sd14_dev_generic"
cat configs/gate.yaml
