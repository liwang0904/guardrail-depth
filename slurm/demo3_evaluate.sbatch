#!/bin/bash
# Score every Demo 3 run (SSCD / CLIP as in Demo 2), then build results/demo3/.
#   sbatch slurm/demo3_evaluate.sbatch
# With GD_RERUN_DEMO2=1 (set by demo3_fresh_submit.sh) the re-rendered Demo 2 runs are scored too and their
# Demo 2 report goes to results/demo2_rerun/ (reproducibility check against the committed results/demo2/).
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu2080|gpu4090|gpul40s"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J gd3-eval
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
if [ -n "${GD_RERUN_DEMO2:-}" ]; then
  python scripts/evaluate.py --clip clip_eval "$GD_OUTPUT_ROOT"/demo2/*/
  python scripts/demo2_report.py --results results/demo2_rerun
fi
python scripts/evaluate.py --clip clip_eval $(ls -d "$GD_OUTPUT_ROOT"/demo3/*/ | grep -v _dev_)
python scripts/demo3_report.py
