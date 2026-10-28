#!/bin/bash
# Model-level baselines on all 500+500 prompts: Wen et al. detection and NeMo neuron probe,
# then metrics for finished generation runs.
#   sbatch slurm/analysis.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu2080|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J gd-analysis
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
python scripts/run_detection.py --split all --n-images 4 --attacks direct,truncate_repeat,override
python scripts/run_neurons.py --split all --attacks direct,truncate_repeat,override
for run in "$GD_OUTPUT_ROOT"/track1_memorized/* "$GD_OUTPUT_ROOT"/track1_copycat/*; do
  [ -f "$run/records.jsonl" ] || ls "$run"/records.*.jsonl >/dev/null 2>&1 || continue
  python scripts/evaluate.py "$run" --clip clip_eval
done
