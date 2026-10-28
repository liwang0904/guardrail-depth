# Running on the Markov teaching cluster

Condensed from the course's *Markov Teaching Cluster (csds447) Quick Start*; commands are the ones
our scripts use.

## Access and account

* OnDemand (no VPN): <https://ondemand-markov.case.edu> → Clusters → Markov Shell Access; or
  `ssh markov.case.edu` (on campus, or via the Case VPN).
* `sacctmgr -n show assoc user=$USER format=account` must list `csds447`; every `srun`/`sbatch`
  passes `-A csds447` explicitly (it may not be your default account).

## Storage

* Home is limited to **25 GB** and exceeding it blocks cluster access (`quotagrp csds447`).
* Everything large goes to course storage. `slurm/env.sh` uses
  `TEAM_DIR=/mnt/vstor/courses/csds447/$USER` (set `GD_TEAM=teamN` or `TEAM_DIR` to use a shared team
  folder) for the conda env, `HF_HOME`, data, outputs, pip/torch caches.
* Team folder (once, by one teammate; replace N and the partner's Case ID):

```bash
mkdir -m 700 /mnt/vstor/courses/csds447/teamN
setfacl -m u:partnerID:rwx,d:u:partnerID:rwx,d:u:$USER:rwx /mnt/vstor/courses/csds447/teamN
```

* Course spaces may be cleared two weeks after the semester; copy results out before then.

## GPUs (`-C` feature) and what we run on them

| feature | GPU | memory | our use |
|---|---|---|---|
| `gpu2080` | RTX 2080 Ti | 11 GB, fp16 only | SD 1.4 / SD 2-base generation, Wen detector, neuron probe, smoke test |
| `gpu4090` | RTX 4090 | 24 GB, bf16 | SDXL / Playground, Qwen2.5-VL-7B judge, Qwen2.5-7B rewriter |
| `gpul40s` | L40S | 48 GB (~45 usable) | Qwen3-8B with long reasoning traces; PixArt-α (T5-XXL) |
| `gpu2h100` | H100 MIG slice | ~12 GB | same as 2080 Ti; not a large-memory option |

Class limits: 6 GPUs and 128 cores in use at once across all csds447 students; prefer one GPU per job,
request only the time you need, and end interactive sessions (`exit`). `Pending (AssocGrpGRES)` means
the class allocation is busy. Our sbatch files list alternatives (`-C "gpu2080|gpu4090"`) so jobs
start on whichever type frees up first.

PyTorch wheels bundle CUDA (we install the CUDA 12.6 build); `module load CUDA/...` is only needed to
compile CUDA code, which we don't.

## Workflow

```bash
cd /mnt/vstor/courses/csds447/$USER/guardrail-depth && mkdir -p logs
srun -A csds447 -p markov_cpu -c 4 --mem=16G -t 01:00:00 --pty bash slurm/setup_env.sh   # once
bash slurm/download.sh                       # data + pilot weights (add `full` for everything)
sbatch slurm/smoke.sbatch                    # ~10 min on a 2080 Ti
sbatch --array=0-15%2 slurm/generate.sbatch configs/experiments/track1_memorized.yaml sd14_memorized
sbatch slurm/llm.sbatch rewrite              # D4 rewrites + reasoning traces (L40S)
sbatch --array=0-11%2 -C gpul40s --mem=48G slurm/generate.sbatch configs/experiments/track1_copycat.yaml playground_copycat
sbatch slurm/llm.sbatch judge "$GD_OUTPUT_ROOT/track1_copycat/playground_copycat"
sbatch slurm/analysis.sbatch                 # detector + neurons on 1000 prompts, metrics for finished runs

squeue -u $USER -o "%i %T %r %M %l"          # monitor
sacct -u $USER -X -o JobID,JobName,State,ExitCode,Elapsed,MaxRSS   # COMPLETED, 0:0
```

Interactive debugging on a GPU node (exit as soon as you are done):

```bash
srun -A csds447 -p markov_gpu -C gpu2080 --gres=gpu:1 -c 4 --mem=16G -t 01:00:00 --pty bash
source slurm/env.sh && python scripts/check_env.py
```

If compute nodes cannot reach the internet, run `slurm/download.sh` on the login node (inside
`tmux`) first; every later job reads weights from `$HF_HOME`. Missing VM ground truth (dead LAION
links) can be filled from Wen et al.'s Google Drive archive (link in their README) with `gdown`.
