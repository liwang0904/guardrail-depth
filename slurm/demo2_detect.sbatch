#!/bin/bash
# Wen et al. memorization detector (AUC, TPR and F1 at 1% FPR) on all 500 memorized + 500 control prompts
# for SD 1.4, and the 219 SD 2 memorized prompts, under every prompt-level guardrail.
#   sbatch slurm/demo2_detect.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu2080|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -J gd-detect
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
D=none,task_instruction,neg_inprompt,negbuf_rules,cot,rand_tokens
python scripts/run_detection.py --model sd14 --memorized memorized_sd1 --split all --n-images 4 \
    --defenses $D --attacks direct,truncate_repeat,override --out "$GD_OUTPUT_ROOT/detection/sd14_all"
python scripts/run_detection.py --model sd2base --memorized memorized_sd2 --split all --n-images 4 \
    --defenses $D --attacks direct --out "$GD_OUTPUT_ROOT/detection/sd2base_all"
