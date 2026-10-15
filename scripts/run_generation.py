"""Generate images for (item x attack x defense x seed) with model-level probes attached.

    # all jobs of an experiment config
    python scripts/run_generation.py --config configs/experiments/pilot_local.yaml
    # one job, sharded for a SLURM array (task i of n)
    python scripts/run_generation.py --config configs/experiments/track1_memorized.yaml --job sd14_memorized --shard 0/8
    # ad hoc
    python scripts/run_generation.py --model sd14 --dataset memorized_sd1 --split dev --limit 4 \
        --attacks direct --defenses none,task_instruction,cot --seeds 0,1 --probes noise,attention
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.generation import Job, run_job  # noqa: E402
from guardrail_depth.models import load_t2i, pick_device  # noqa: E402
from guardrail_depth.paths import output_root  # noqa: E402


def csv(s: str | None, cast=str) -> list | None:
    return [cast(x) for x in s.split(",")] if s else None


def jobs_from_args(args) -> list[Job]:
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        defaults = cfg.get("defaults", {})
        jobs = [Job(**{**defaults, **j}) for j in cfg["jobs"]]
        if args.job:
            jobs = [j for j in jobs if j.name in args.job.split(",")]
            if not jobs:
                raise SystemExit(f"no job named {args.job} in {args.config}")
        for j in jobs:
            j.out = j.out or f"{Path(args.config).stem}/{j.name}"
        return jobs
    job = Job(name=args.name, model=args.model, dataset=args.dataset, split=args.split,
              attacks=csv(args.attacks), defenses=csv(args.defenses), seeds=csv(args.seeds, int),
              ids=csv(args.ids, int), limit=args.limit, probes=csv(args.probes) or [],
              x0_steps=csv(args.x0_steps, int) or [], steps=args.steps, guidance=args.guidance,
              rewrites=csv(args.rewrites) or [], out=f"adhoc/{args.name}")
    return [job]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--job", help="comma-separated job names from the config")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th item starting at i")
    ap.add_argument("--device")
    ap.add_argument("--name", default="adhoc")
    ap.add_argument("--model", default="sd14")
    ap.add_argument("--dataset", default="memorized_sd1")
    ap.add_argument("--split")
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--attacks", default="direct")
    ap.add_argument("--defenses", default="none")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--probes", default="noise")
    ap.add_argument("--x0-steps")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--rewrites")
    args = ap.parse_args()

    shard = tuple(int(x) for x in args.shard.split("/"))
    device = pick_device(args.device)
    jobs = jobs_from_args(args)
    pipe, loaded = None, None
    for job in jobs:
        if job.model != loaded:
            del pipe
            pipe, spec = load_t2i(job.model, device)
            loaded = job.model
            print(f"loaded {spec.hf_id}@{spec.revision[:10]} on {device} ({pipe.unet.dtype if hasattr(pipe, 'unet') else ''})")
        run_job(pipe, spec, job, output_root() / job.out, shard=shard)


if __name__ == "__main__":
    main()
