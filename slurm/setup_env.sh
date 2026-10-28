#!/bin/bash
# One-time environment setup on Markov (run from the repo root, on a CPU node):
#   srun -A csds447 -p markov_cpu -c 4 --mem=16G -t 01:00:00 --pty bash slurm/setup_env.sh
# Creates a Python 3.11 env in course storage and installs the pinned requirements with
# CUDA 12.6 PyTorch wheels (they bundle their own CUDA runtime; no `module load CUDA` needed).
set -euo pipefail
source slurm/env.sh

if command -v conda >/dev/null 2>&1; then
  CONDA=conda
elif [ -x "$TEAM_DIR/miniforge3/bin/conda" ]; then
  CONDA="$TEAM_DIR/miniforge3/bin/conda"
else
  cat <<'EOF'
No conda found. Either load one (`module spider conda`, `module spider Miniforge`) or install
Miniforge into the team folder once:
  curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$TEAM_DIR/miniforge3"
then re-run this script.
EOF
  exit 1
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$CONDA" create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.11 pip
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
"$ENV_DIR/bin/python" -m pip install -r requirements.txt
"$ENV_DIR/bin/python" scripts/check_env.py
echo "env ready: $ENV_DIR  (jobs pick it up through slurm/env.sh)"
