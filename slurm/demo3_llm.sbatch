#!/bin/bash
# Demo 3 LLM/VLM stages on a 24-48 GB GPU.
#   sbatch slurm/demo3_llm.sbatch screen             # gate concept screen for every Demo 3 request + D4 rewrites
#                                                    # of the near-miss set (Qwen3-8B)
#   sbatch slurm/demo3_llm.sbatch judge <run>...     # He et al. DETECT / CONS (Qwen2.5-VL-7B) on runs under $GD_OUTPUT_ROOT
#   sbatch slurm/demo3_llm.sbatch onprompt [run...]  # on-prompt VQAScore (+ mismatch check) on every Demo 2 and Demo 3 run
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu4090|gpul40s"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -J gd3-llm
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi -L
CFG=configs/experiments/demo3.yaml
SC="$GD_OUTPUT_ROOT/screens"
RW="$GD_OUTPUT_ROOT/rewrites"
mode=$1; shift
case "$mode" in
  screen)
    python scripts/run_screen.py --config $CFG --job sd14_memorized --out "$SC/demo3_memorized_sd1.csv"
    python scripts/run_screen.py --config $CFG --job sd14_controls,sd14_dev_benign --out "$SC/demo3_unmemorized_laion.csv"
    python scripts/run_screen.py --config $CFG --job sd2_memorized --out "$SC/demo3_memorized_sd2.csv"
    python scripts/run_screen.py --config $CFG --job sd14_copycat,sd14_dev_copycat --out "$SC/demo3_copycat_characters.csv"
    python scripts/run_screen.py --config $CFG --job sd14_style --out "$SC/demo3_style_prompts.csv"
    python scripts/run_screen.py --config $CFG --job sd14_generic,sd14_dev_generic --out "$SC/demo3_generic_prompts.csv"
    # D4 baseline on the near-miss set (same rewriter and settings as the Demo 2 non-character rewrites)
    python scripts/run_rewriter.py --llm rewriter_reasoning --config $CFG --job sd14_generic --out "$RW/demo3_generic_prompts.csv"
    ;;
  judge)
    for run in "$@"; do python scripts/run_judge.py "$GD_OUTPUT_ROOT/$run" --batch-size 8; done
    ;;
  onprompt)   # all runs, or only the runs given (relative to $GD_OUTPUT_ROOT)
    if [ $# -gt 0 ]; then
      python scripts/run_onprompt.py --mismatch $(for r in "$@"; do echo "$GD_OUTPUT_ROOT/$r"; done)
    else
      python scripts/run_onprompt.py --mismatch "$GD_OUTPUT_ROOT"/demo2/*/ "$GD_OUTPUT_ROOT"/demo3/*/
    fi
    ;;
esac
