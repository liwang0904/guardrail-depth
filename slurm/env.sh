# Sourced by every Markov job: puts all large files in course storage, not the 25 GB home quota.
# Override TEAM_DIR if your team folder differs (see docs/MARKOV.md, "Team folder").
export TEAM_DIR="${TEAM_DIR:-/mnt/vstor/courses/csds447/${GD_TEAM:-$USER}}"
export ENV_DIR="${ENV_DIR:-$TEAM_DIR/envs/guardrail-depth}"
export HF_HOME="$TEAM_DIR/hf_cache"                 # model weights (~25 GB for the full model set)
export GD_DATA_ROOT="$TEAM_DIR/data"                # raw downloads + ground-truth images
export GD_OUTPUT_ROOT="$TEAM_DIR/outputs"           # generated images, probes, metrics
export PIP_CACHE_DIR="$TEAM_DIR/pip_cache"
export TORCH_HOME="$TEAM_DIR/torch_cache"
export XDG_CACHE_HOME="$TEAM_DIR/xdg_cache"         # open_clip / misc caches
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$GD_DATA_ROOT" "$GD_OUTPUT_ROOT" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME"
if [ -x "$ENV_DIR/bin/python" ]; then
  export PATH="$ENV_DIR/bin:$PATH"
fi
