#!/bin/bash
# Demo 2 LLM/VLM stages on a 24-48 GB GPU.
#   sbatch slurm/demo2_llm.sbatch rewrite              # D4 rewrites for every Demo 2 job (Qwen3-8B)
#   sbatch slurm/demo2_llm.sbatch judge sd14_copycat   # He et al. DETECT / CONS with Qwen2.5-VL-7B
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu4090|gpul40s"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -J gd-llm
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi -L
CFG=configs/experiments/demo2.yaml
RW="$GD_OUTPUT_ROOT/rewrites"
case "$1" in
  rewrite)
    # characters: reasoning on, so we can see whether the model recognized the character
    python scripts/run_rewriter.py --llm rewriter_reasoning --thinking --config $CFG --job sd14_copycat \
        --out "$RW/demo2_copycat_characters.csv"
    python scripts/run_rewriter.py --llm rewriter_reasoning --config $CFG --job sd14_memorized --out "$RW/demo2_memorized_sd1.csv"
    python scripts/run_rewriter.py --llm rewriter_reasoning --config $CFG --job sd14_controls --out "$RW/demo2_unmemorized_laion.csv"
    python scripts/run_rewriter.py --llm rewriter_reasoning --config $CFG --job sd2_memorized --out "$RW/demo2_memorized_sd2.csv"
    python scripts/run_rewriter.py --llm rewriter_reasoning --config $CFG --job sd14_style --out "$RW/demo2_style_prompts.csv"
    ;;
  judge)
    python scripts/run_judge.py "$GD_OUTPUT_ROOT/demo2/$2"
    ;;
esac
