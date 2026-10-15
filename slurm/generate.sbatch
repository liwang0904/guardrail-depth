#!/bin/bash
# Run one experiment job as a SLURM array; each task takes every N-th item.
#   sbatch --array=0-15%2 slurm/generate.sbatch configs/experiments/track1_memorized.yaml sd14_memorized
#   sbatch --array=0-11%2 -C gpul40s --mem=48G slurm/generate.sbatch configs/experiments/track1_copycat.yaml playground_copycat
# `%2` keeps at most 2 of the class's 6 shared GPUs busy; tasks queue for the rest.
# SD v1.x/2.x fit the 11 GB 2080 Ti; SDXL-class models need -C gpu4090 or gpul40s (override on the command line).
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu2080|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -J gd-gen
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
source slurm/env.sh
CONFIG=$1
JOB=$2
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}
nvidia-smi -L
python scripts/run_generation.py --config "$CONFIG" --job "$JOB" --shard "$I/$N"
