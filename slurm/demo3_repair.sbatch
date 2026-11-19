#!/bin/bash
# One-off fix for runs generated before the output-check fix in generation._GateRunner.run: restore the G-pre
# ablation rows to their first render (scripts/repair_gate_pre.py, no rendering), re-score only what changed
# (evaluate.py on the runs with restored images; the judges skip images they already scored), rebuild the report.
#   sbatch slurm/demo3_repair.sbatch
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu4090|gpul40s"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J gd3-repair
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi -L
D3="$GD_OUTPUT_ROOT/demo3"
ALL=$(ls -d "$D3"/*/ | grep -v _dev_)
python scripts/repair_gate_pre.py $ALL | tee logs/repair_gate_pre.txt
# runs whose G-pre images changed: "<run>: N G-pre rows, M images back to the first render" with M > 0
CHANGED=$(awk -F: '/images back to the first render/ { split($2, a, ", "); split(a[2], b, " "); if (b[1] > 0) print $1 }' logs/repair_gate_pre.txt)
echo "re-scoring: $CHANGED"
for r in $CHANGED; do python scripts/evaluate.py --clip clip_eval "$D3/$r"; done
for r in $CHANGED; do
  case "$r" in *_copycat|*_generic) python scripts/run_judge.py "$D3/$r" --batch-size 8 ;; esac
done
[ -n "$CHANGED" ] && python scripts/run_onprompt.py --mismatch $(for r in $CHANGED; do echo "$D3/$r"; done)
rm -f results/demo3/fig_f1_*PixArt*.png
python scripts/demo3_report.py
