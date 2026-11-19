#!/bin/bash
# Demo 2 baselines + Demo 3 from an empty $GD_OUTPUT_ROOT (an account that cannot read the Demo 2 images),
# sized for the class QOS cap of 2 GPUs per user.
#   bash slurm/demo3_fresh_submit.sh                  # submits everything below
#   RWC=<id> RWO=<id> bash slurm/demo3_fresh_submit.sh   # reuse rewrite jobs that are already queued/running
# Two lanes share the 2 GPUs:
#   big lane (4090/L40S): Demo 2 D4 rewrites, the gate's concept screen, PixArt-alpha and Playground v2.5,
#                         the VLM judges; it gets priority whenever a 24 GB+ GPU is free
#   SD lane  (2080 Ti):   dev calibration, the Demo 2 grid on SD 1.4 / SD 2 (configs/experiments/demo2.yaml,
#                         seed 0: GD_SEEDS=0), the gate rows and the near-miss set, scoring
# Short array tasks keep a GPU from being held for hours, so the big lane is not starved. The Demo 2
# re-run is also scored into results/demo2_rerun/ as a reproducibility check (seed 0 vs. the committed
# two-seed results/demo2/). `afterany` = start when the previous job ends even if it failed; reruns skip
# finished images. Nodes class21t (its L40S fails CUDA initialization) and class19t (one of its 2080 Tis
# returns NaN at random, in fp16 and fp32 alike) are excluded.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
C2=configs/experiments/demo2.yaml
C3=configs/experiments/demo3.yaml
EXC=(--exclude=class21t,class19t)
BIG=(-C "gpu4090|gpul40s" --mem=48G "${EXC[@]}")
SD=(-C gpu2080 --nice=100 "${EXC[@]}")
dep() { local d=""; for j in "$@"; do [ -n "$j" ] && d="${d:+$d:}$j"; done; echo "${d:+--dependency=afterany:$d}"; }
gen() {   # gen "<deps>" <array> <config> <jobs> [sbatch args...]
  local deps=$1 arr=$2 cfg=$3 job=$4; shift 4
  sbatch --parsable $(dep $deps) --array="$arr" "$@" slurm/generate.sbatch "$cfg" "$job"
}
d2() {    # Demo 2 grid, seed 0
  local deps=$1 arr=$2 job=$3; shift 3
  gen "$deps" "$arr" $C2 "$job" --export=ALL,GD_SEEDS=0 "$@"
}

# --- big lane
RWC=${RWC:-$(sbatch --parsable "${EXC[@]}" slurm/demo2_llm.sbatch rewrite-characters)}
RWO=${RWO:-$(sbatch --parsable "${EXC[@]}" slurm/demo2_llm.sbatch rewrite-others)}
SC=$(sbatch --parsable "${EXC[@]}" slurm/demo3_llm.sbatch screen)

# --- SD lane: calibration and the Demo 2 baselines (D4 rows wait for the rewrites, filled by F)
CAL=$(sbatch --parsable -C gpu2080 "${EXC[@]}" slurm/demo3_calibrate.sbatch)
M=$(d2 "" 0-2 sd14_memorized "${SD[@]}")
CC=$(d2 "" 0-1 sd14_copycat "${SD[@]}")
CT=$(d2 "" 0-1 sd14_controls,sd2_memorized "${SD[@]}")
ST=$(d2 "" 0-2 sd14_style "${SD[@]}")
F=$(d2 "$RWC $RWO $M $CC $CT $ST" 0-1 sd14_memorized,sd14_copycat,sd14_controls,sd2_memorized,sd14_style "${SD[@]}")
G1=$(gen "$SC $CAL $F" 0-1 $C3 sd14_memorized,sd14_controls,sd2_memorized,sd14_copycat,sd14_style,sd14_generic "${SD[@]}")

# --- big lane: models outside Stable Diffusion (gate rows need the screen and the calibration)
PX=$(gen "$SC $CAL $RWC" 0 $C3 pixart_copycat,pixart_generic "${BIG[@]}")
PG=$(gen "$PX" 0-1 $C3 playground_copycat,playground_generic "${BIG[@]}")

# --- judges, scoring, on-prompt, report
J=$(sbatch --parsable $(dep $G1 $PG) "${EXC[@]}" slurm/demo3_llm.sbatch judge \
    demo2/sd14_copycat demo3/sd14_copycat demo3/sd14_generic demo3/playground_copycat demo3/pixart_copycat \
    demo3/playground_generic demo3/pixart_generic)
EV=$(sbatch --parsable $(dep $G1 $PG) -C gpu2080 "${EXC[@]}" --export=ALL,GD_RERUN_DEMO2=1 slurm/demo3_evaluate.sbatch)
OP=$(sbatch --parsable $(dep $J) "${EXC[@]}" slurm/demo3_llm.sbatch onprompt)
RP=$(sbatch --parsable $(dep $EV $OP) -A csds447 -p markov_cpu -c 2 --mem=8G -t 00:30:00 -J gd3-report \
    -o logs/%x-%j.out --wrap "source slurm/env.sh && python scripts/demo2_report.py --results results/demo2_rerun && python scripts/demo3_report.py")
printf 'RWC=%s\nRWO=%s\nSC=%s\nCAL=%s\nM=%s\nCC=%s\nCT=%s\nST=%s\nF=%s\nG1=%s\nPX=%s\nPG=%s\nJ=%s\nEV=%s\nOP=%s\nRP=%s\n' \
    "$RWC" "$RWO" "$SC" "$CAL" "$M" "$CC" "$CT" "$ST" "$F" "$G1" "$PX" "$PG" "$J" "$EV" "$OP" "$RP" | tee logs/demo3_jobs.env
echo "watch with: squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.20R'"
