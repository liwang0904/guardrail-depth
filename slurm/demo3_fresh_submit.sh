#!/bin/bash
# Demo 2 baselines + Demo 3 from an empty $GD_OUTPUT_ROOT (an account that cannot read the Demo 2 images).
#   bash slurm/demo3_fresh_submit.sh renders   # phase 1: rewrites, calibration, every baseline render
#   bash slurm/demo3_fresh_submit.sh gate      # phase 2: concept screen, gate rows, judges, scoring, reports
# Phase 1 re-renders the Demo 2 grid with configs/experiments/demo2.yaml (same items, seeds, settings; the
# detector grid is skipped, Demo 3 does not use it) and the baselines of the new Demo 3 jobs. Phase 2 can be
# submitted while phase 1 runs (it waits on phase 1's job ids, saved in logs/demo3_phase1.env). The Demo 2
# re-run is scored into results/demo2_rerun/ as a reproducibility check; results/demo2/ stays as committed.
# About 6 GPUs at once at most: SD 1.x/2.x on the 2080 Tis, 1024 px models and LLM/VLM stages on 4090/L40S.
# `afterany` = start when the previous job ends even if it failed; reruns skip finished images.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
C2=configs/experiments/demo2.yaml
C3=configs/experiments/demo3.yaml
SMALL=(-C "gpu2080|gpu4090")
BIG=(-C "gpu4090|gpul40s" --mem=48G)
dep() { local d=""; for j in "$@"; do [ -n "$j" ] && d="${d:+$d:}$j"; done; echo "${d:+--dependency=afterany:$d}"; }
gen() {   # gen "<deps>" <array> <config> <jobs> [sbatch args...]
  local deps=$1 arr=$2 cfg=$3 job=$4; shift 4
  sbatch --parsable $(dep $deps) --array="$arr" "$@" slurm/generate.sbatch "$cfg" "$job"
}
base() {  # same, baselines only: the gate's rows are filled by a later pass of the same job
  local deps=$1 arr=$2 cfg=$3 job=$4; shift 4
  sbatch --parsable $(dep $deps) --array="$arr" --export=ALL,GD_SKIP_GATED=1 "$@" slurm/generate.sbatch "$cfg" "$job"
}

case "${1:-}" in
renders)
  RWC=$(sbatch --parsable slurm/demo2_llm.sbatch rewrite-characters)   # Demo 2 D4 rewrites, characters (reasoning)
  RWO=$(sbatch --parsable slurm/demo2_llm.sbatch rewrite-others)       # ... memorized, captions, styles
  CAL=$(sbatch --parsable slurm/demo3_calibrate.sbatch)                # dev renders -> configs/gate.yaml
  # Demo 2 baselines on SD 1.4 / SD 2; D4 rows wait for their rewrites and are filled by M2 / F2
  M=$(gen "" 0-3%2 $C2 sd14_memorized "${SMALL[@]}")
  CC=$(gen "$CAL" 0-1%1 $C2 sd14_copycat "${SMALL[@]}")
  ST=$(gen "$CC" 0-1%1 $C2 sd14_style "${SMALL[@]}")
  CT=$(gen "$M" 0-1%1 $C2 sd14_controls,sd2_memorized "${SMALL[@]}")
  M2=$(gen "$M $RWO" 0-1%1 $C2 sd14_memorized "${SMALL[@]}")
  F2=$(gen "$CT $ST $RWC $RWO" 0-1%1 $C2 sd14_copycat,sd14_style,sd14_controls,sd2_memorized "${SMALL[@]}")
  # 1024 px and DiT models start at once; their rows that need a Qwen rewrite are filled later (XF, gate pass)
  X=$(gen "" 0-1%1 $C2 sdxl_copycat "${BIG[@]}")
  PX=$(base "" 0-1%1 $C3 pixart_copycat,pixart_style "${BIG[@]}")
  PG=$(base "$PX" 0-1%1 $C3 playground_copycat,playground_style "${BIG[@]}")
  XF=$(gen "$X $RWC" 0%1 $C2 sdxl_copycat "${BIG[@]}")
  RW="$RWC:$RWO"
  printf 'RW=%s\nRWC=%s\nRWO=%s\nCAL=%s\nM=%s\nCC=%s\nST=%s\nCT=%s\nM2=%s\nF2=%s\nX=%s\nXF=%s\nPX=%s\nPG=%s\n' \
      "$RW" "$RWC" "$RWO" "$CAL" "$M" "$CC" "$ST" "$CT" "$M2" "$F2" "$X" "$XF" "$PX" "$PG" > logs/demo3_phase1.env
  cat logs/demo3_phase1.env
  ;;
gate)
  source logs/demo3_phase1.env
  SC=$(sbatch --parsable slurm/demo3_llm.sbatch screen)           # gate concept screen + near-miss rewrites
  # near-miss requests on every model (their D4 rows need the near-miss rewrites from SC)
  NG=$(base "$SC $PX $RWC" 0%1 $C3 pixart_generic "${BIG[@]}")
  NX=$(base "$SC $PG" 0%1 $C3 playground_generic,sdxl_generic "${BIG[@]}")
  # gate rows: screen + calibration + the renders they reuse
  G1=$(gen "$SC $CAL $M2 $F2" 0-1%1 $C3 sd14_memorized,sd14_controls,sd2_memorized,sd14_copycat,sd14_style,sd14_generic "${SMALL[@]}")
  G3=$(gen "$SC $CAL $XF $NG $NX" 0%1 $C3 sdxl_copycat,sdxl_generic,pixart_copycat,pixart_generic,pixart_style,playground_copycat,playground_generic,playground_style "${BIG[@]}")
  J=$(sbatch --parsable $(dep $F2 $XF $G1 $G3) slurm/demo3_llm.sbatch judge \
      demo2/sd14_copycat demo2/sdxl_copycat demo3/sd14_copycat demo3/sdxl_copycat demo3/playground_copycat \
      demo3/pixart_copycat demo3/sd14_generic demo3/sdxl_generic demo3/playground_generic demo3/pixart_generic)
  EV=$(sbatch --parsable $(dep $J) --export=ALL,GD_RERUN_DEMO2=1 slurm/demo3_evaluate.sbatch)
  OP=$(sbatch --parsable $(dep $EV) slurm/demo3_llm.sbatch onprompt)
  RP=$(sbatch --parsable $(dep $OP) -A csds447 -p markov_cpu -c 2 --mem=8G -t 00:30:00 -J gd3-report \
      -o logs/%x-%j.out --wrap "source slurm/env.sh && python scripts/demo3_report.py")
  printf 'SC=%s\nNG=%s\nNX=%s\nG1=%s\nG3=%s\nJ=%s\nEV=%s\nOP=%s\nRP=%s\n' "$SC" "$NG" "$NX" "$G1" "$G3" "$J" "$EV" "$OP" "$RP" \
      > logs/demo3_phase2.env
  cat logs/demo3_phase2.env
  ;;
*)
  echo "usage: bash slurm/demo3_fresh_submit.sh renders|gate"; exit 1 ;;
esac
echo "watch with: squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.20R'"
