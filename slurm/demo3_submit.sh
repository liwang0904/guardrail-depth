#!/bin/bash
# Submit the whole Demo 3 grid as three lanes (<= 3 of the class's 6 GPUs at a time). `afterany` = start when
# the previous job ends even if it failed; reruns skip finished images, so a lane never stalls on one bad task.
#   bash slurm/demo3_submit.sh
# Prerequisites: the Demo 2 runs under $GD_OUTPUT_ROOT/demo2 (gate rows reuse their renders) and
# slurm/demo3_download.sh. Gated rows always run after the dev calibration (configs/gate.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CFG=configs/experiments/demo3.yaml
gen() {  # gen <dependency or ""> <array spec> <jobs> [extra sbatch args...]
  local dep=$1 arr=$2 job=$3; shift 3
  sbatch --parsable ${dep:+--dependency=afterany:$dep} --array="$arr" "$@" slurm/generate.sbatch $CFG "$job"
}
base() {  # like gen, but baselines only (gated rows are filled by a later pass)
  local dep=$1 arr=$2 job=$3; shift 3
  sbatch --parsable ${dep:+--dependency=afterany:$dep} --array="$arr" --export=ALL,GD_SKIP_GATED=1 "$@" \
      slurm/generate.sbatch $CFG "$job"
}
BIG=(-C "gpu4090|gpul40s" --mem=48G)

# Lane A (24 GB+): concept screen + near-miss rewrites, then Playground v2.5 and SDXL
S=$(sbatch --parsable slurm/demo3_llm.sbatch screen)
# Lane B (any GPU): dev renders + calibration, then the gate on the SD 1.4 / SD 2 Demo 2 grid and the near-miss set
CAL=$(sbatch --parsable slurm/demo3_calibrate.sbatch)
# Lane C (24 GB+): PixArt-alpha baselines while the screen and calibration run
PX0=$(base "" 0-1%1 pixart_copycat,pixart_generic,pixart_style "${BIG[@]}")

A1=$(gen "$S:$CAL" 0-1%1 playground_copycat "${BIG[@]}")
A2=$(gen "$A1" 0%1 playground_generic,playground_style "${BIG[@]}")
A3=$(gen "$A2" 0%1 sdxl_copycat,sdxl_generic "${BIG[@]}")
B1=$(gen "$S:$CAL" 0-1%1 sd14_memorized,sd14_controls,sd2_memorized)
B2=$(gen "$B1" 0-1%1 sd14_copycat,sd14_style,sd14_generic)
PX1=$(gen "$PX0:$S:$CAL" 0%1 pixart_copycat,pixart_generic,pixart_style "${BIG[@]}")

# Judges, on-prompt scoring and the report once every lane is done
J=$(sbatch --parsable --dependency=afterany:$A3:$B2:$PX1 slurm/demo3_llm.sbatch judge \
    demo3/sd14_copycat demo3/sdxl_copycat demo3/playground_copycat demo3/pixart_copycat \
    demo3/sd14_generic demo3/sdxl_generic demo3/playground_generic demo3/pixart_generic)
EV=$(sbatch --parsable --dependency=afterany:$J slurm/demo3_evaluate.sbatch)
OP=$(sbatch --parsable --dependency=afterany:$EV slurm/demo3_llm.sbatch onprompt)
RP=$(sbatch --parsable --dependency=afterany:$OP --wrap "source slurm/env.sh && python scripts/demo3_report.py" \
    -A csds447 -p markov_cpu -c 2 --mem=8G -t 00:30:00 -J gd3-report -o logs/%x-%j.out)
echo "submitted: screen $S | calibrate $CAL | lane A $A1 $A2 $A3 | lane B $B1 $B2 | lane C $PX0 $PX1 | judge $J eval $EV onprompt $OP report $RP"
echo "watch with: squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.20R'"
