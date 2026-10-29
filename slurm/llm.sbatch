#!/bin/bash
# LLM/VLM stages on an L40S: D4 rewrites with reasoning traces (Qwen3-8B, thinking on), then the
# VLM judge (Qwen2.5-VL-7B) for He et al.'s DETECT / CONS on CopyCat generations.
#   sbatch slurm/llm.sbatch rewrite
#   sbatch slurm/llm.sbatch judge "$GD_OUTPUT_ROOT/track1_copycat/playground_copycat"
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C gpul40s
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -J gd-llm
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
case "$1" in
  rewrite)
    python scripts/run_rewriter.py --llm rewriter_reasoning --thinking --dataset copycat_characters --split test \
        --attacks direct,lure_desc60,lure_kw_laion5,perturb_phonetic,override
    python scripts/run_rewriter.py --llm rewriter_reasoning --thinking --dataset style_prompts --split test --attacks direct
    ;;
  judge)
    python scripts/run_judge.py "$2"
    ;;
esac
