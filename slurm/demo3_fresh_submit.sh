#!/bin/bash
# Demo 2 baselines + Demo 3 from an empty $GD_OUTPUT_ROOT (an account that cannot read the Demo 2 renders).
#   bash slurm/demo3_fresh_submit.sh
# Re-renders the Demo 2 grid with configs/experiments/demo2.yaml (same items, seeds, settings; skips the
# detector grid, which Demo 3 does not use), then runs the Demo 3 grid on top of it. The Demo 2 re-run is
# scored into results/demo2_rerun/ as a reproducibility check; the committed results/demo2/ stays as is.
# At most ~5 GPUs at once, SD 1.x/2.x on the 2080 Tis, the 1024 px models and LLM/VLM stages on 4090/L40S.
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

# --- LLM stages and calibration
RW=$(sbatch --parsable slurm/demo2_llm.sbatch rewrite)          # Demo 2 D4 rewrites (Qwen3-8B)
SC=$(sbatch --parsable slurm/demo3_llm.sbatch screen)           # gate concept screen + near-miss rewrites
CAL=$(sbatch --parsable slurm/demo3_calibrate.sbatch)           # dev renders -> configs/gate.yaml

# --- Demo 2 baselines on SD 1.4 / SD 2 (D4 rows for LLM-rewritten requests are filled after RW)
M=$(gen "" 0-3%2 $C2 sd14_memorized "${SMALL[@]}")
CC=$(gen "$CAL" 0-1%1 $C2 sd14_copycat "${SMALL[@]}")
ST=$(gen "$CC" 0-1%1 $C2 sd14_style "${SMALL[@]}")
CT=$(gen "$M" 0-1%1 $C2 sd14_controls,sd2_memorized "${SMALL[@]}")
M2=$(gen "$M $RW" 0-1%1 $C2 sd14_memorized "${SMALL[@]}")
F2=$(gen "$CT $ST $RW" 0-1%1 $C2 sd14_copycat,sd14_style,sd14_controls,sd2_memorized "${SMALL[@]}")

# --- 1024 px and DiT models (after the rewrites, so their D4 rows render in the same pass)
X=$(gen "$RW" 0-1%1 $C2 sdxl_copycat "${BIG[@]}")
PX=$(base "$SC $RW" 0-1%1 $C3 pixart_copycat,pixart_generic,pixart_style "${BIG[@]}")
PG=$(base "$X" 0-1%1 $C3 playground_copycat,playground_generic,playground_style,sdxl_generic "${BIG[@]}")

# --- Demo 3 gate rows (screen + calibration + the renders they reuse) and the near-miss set on SD 1.4
G1=$(gen "$SC $CAL $M2 $F2" 0-1%1 $C3 sd14_memorized,sd14_controls,sd2_memorized,sd14_copycat,sd14_style,sd14_generic "${SMALL[@]}")
G3=$(gen "$SC $CAL $X $PX $PG" 0%1 $C3 sdxl_copycat,sdxl_generic,pixart_copycat,pixart_generic,pixart_style,playground_copycat,playground_generic,playground_style "${BIG[@]}")

# --- judges, scoring, on-prompt, reports
J=$(sbatch --parsable $(dep $F2 $X $G1 $G3) slurm/demo3_llm.sbatch judge \
    demo2/sd14_copycat demo2/sdxl_copycat demo3/sd14_copycat demo3/sdxl_copycat demo3/playground_copycat \
    demo3/pixart_copycat demo3/sd14_generic demo3/sdxl_generic demo3/playground_generic demo3/pixart_generic)
EV=$(sbatch --parsable $(dep $J) --export=ALL,GD_RERUN_DEMO2=1 slurm/demo3_evaluate.sbatch)
OP=$(sbatch --parsable $(dep $EV) slurm/demo3_llm.sbatch onprompt)
RP=$(sbatch --parsable $(dep $OP) -A csds447 -p markov_cpu -c 2 --mem=8G -t 00:30:00 -J gd3-report \
    -o logs/%x-%j.out --wrap "source slurm/env.sh && python scripts/demo3_report.py")
echo "llm: rewrite $RW screen $SC | calibrate $CAL"
echo "demo2 baselines: $M $CC $ST $CT | fills $M2 $F2 | sdxl $X"
echo "demo3: pixart $PX playground+sdxl-generic $PG | gate $G1 $G3"
echo "judge $J | evaluate $EV | onprompt $OP | report $RP"
echo "watch with: squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.20R'"
