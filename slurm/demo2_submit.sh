#!/bin/bash
# Submit the whole Demo 2 grid as three lanes; each lane uses one GPU at a time, so we never hold more
# than 3 of the class's 6 GPUs. `afterany` = start when the previous job ends (even if it failed, so a
# single bad shard can't stall the lane; reruns skip finished images).
#   bash slurm/demo2_submit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CFG=configs/experiments/demo2.yaml
gen() {  # gen <dependency or ""> <array spec> <job> [extra sbatch args...]
  local dep=$1 arr=$2 job=$3; shift 3
  sbatch --parsable ${dep:+--dependency=afterany:$dep} --array="$arr" "$@" slurm/generate.sbatch $CFG "$job"
}

# Lane 3 (24 GB+ GPU): D4 rewrites first, then the SDXL and SD 1.4 character runs, then the VLM judge
RW=$(sbatch --parsable slurm/demo2_llm.sbatch rewrite)
X=$(gen "$RW" 0-1%1 sdxl_copycat -C "gpu4090|gpul40s" --mem=40G)
C=$(gen "$X" 0-1%1 sd14_copycat -C "gpu4090|gpul40s")
J1=$(sbatch --parsable --dependency=afterany:$C slurm/demo2_llm.sbatch judge sd14_copycat)
J2=$(sbatch --parsable --dependency=afterany:$J1 slurm/demo2_llm.sbatch judge sdxl_copycat)

# Lane 1: SD 1.4 exact copies (4 shards), the detector, then a pass that fills D4 once rewrites exist
M=$(gen "" 0-3%1 sd14_memorized)
DET=$(sbatch --parsable --dependency=afterany:$M slurm/demo2_detect.sbatch)
M2=$(gen "$DET:$RW" 0-3%1 sd14_memorized)

# Lane 2: SD 2 exact copies, style, benign controls, then the D4 fill pass for all three
S2=$(gen "" 0%1 sd2_memorized)
ST=$(gen "$S2" 0-1%1 sd14_style)
CT=$(gen "$ST" 0%1 sd14_controls)
F1=$(gen "$CT:$RW" 0%1 sd2_memorized)
F2=$(gen "$F1" 0-1%1 sd14_style)
F3=$(gen "$F2" 0%1 sd14_controls)

# Scoring + tables once every lane is done
EV=$(sbatch --parsable --dependency=afterany:$M2:$F3:$J2 slurm/demo2_evaluate.sbatch)
echo "submitted: rewrite $RW | lane1 $M $DET $M2 | lane2 $S2 $ST $CT $F1 $F2 $F3 | lane3 $X $C $J1 $J2 | eval $EV"
echo "watch with: squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.20R'"
