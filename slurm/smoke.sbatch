#!/bin/bash
# End-to-end check on the smallest GPU: environment, tests, and a 2-condition SD v1.4 run.
#   sbatch slurm/smoke.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C gpu2080
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J gd-smoke
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi -L
python scripts/check_env.py --strict
python -m pytest -q
python scripts/run_generation.py --name smoke --model sd14 --dataset memorized_sd1 --split dev --limit 2 \
    --attacks direct --defenses none,task_instruction --seeds 0 --probes noise,attention
python scripts/evaluate.py "$GD_OUTPUT_ROOT/adhoc/smoke"
