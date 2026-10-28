"""Experiment loop: items x attacks x defenses x seeds -> images + per-image records + probe arrays.

Output layout (one directory per job):
    <out>/images/<cond>_s<seed>.png      generated images
    <out>/probes/<cond>.npz              per-step probe arrays (float16)
    <out>/x0/<cond>_s<seed>_t<step>.jpg  predicted-x0 snapshots (optional)
    <out>/records.jsonl                  one line per image: prompts, token window, probe summaries
    <out>/job.json                       resolved job config + environment
Re-running a job skips images that already exist, so SLURM jobs can be resubmitted.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attacks import ATTACK_CODES, make_attack
from .data import load_manifest
from .defenses import DEFENSE_CODES, apply_defense, clip_token_window
from .probes import attention_summary, probes_attached


@dataclass
class Job:
    name: str
    model: str
    dataset: str
    attacks: list[str]
    defenses: list[str]
    seeds: list[int]
    split: str | None = None
    ids: list | None = None
    query: str | None = None
    limit: int | None = None
    probes: list[str] = field(default_factory=lambda: ["noise"])
    x0_steps: list[int] = field(default_factory=list)
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    rewrites: list[str] = field(default_factory=list)
    rewrite_attacks: list[str] | None = None   # read by scripts/run_rewriter.py only
    out: str | None = None


def select_items(job: Job, shard: tuple[int, int] = (0, 1)) -> list[dict]:
    df = load_manifest(job.dataset)
    if job.split:
        df = df[df["split"] == job.split]
    if job.query:
        df = df.query(job.query)
    if job.ids is not None:
        df = df.set_index("id").loc[job.ids].reset_index()
    if job.limit:
        df = df.head(job.limit)
    rows = df.to_dict("records")
    i, n = shard
    return rows[i::n]


def load_rewrites(paths: list[str]) -> dict:
    """(dataset, item_id, attack) -> {rewrite, rewriter, ...} from scripts/run_rewriter.py CSVs."""
    table = {}
    for p in paths:
        p = os.path.expandvars(p)
        if not os.path.exists(p):   # rewriter job not finished yet: D4 rows are skipped and filled on a rerun
            print(f"  rewrites file not found (D4 skipped for now): {p}")
            continue
        df = pd.read_csv(p, keep_default_na=False)
        for r in df.to_dict("records"):
            table[(r["dataset"], str(r["item_id"]), r["attack"])] = r
    return table


def _rewrite_for(job: Job, item: dict, attack: str, table: dict) -> dict | None:
    hit = table.get((job.dataset, str(item["id"]), attack))
    if hit:
        return {"rewrite": hit["rewrite"], "rewriter": hit.get("rewriter", "")}
    # He et al. released GPT-4 rewrites for the CopyCat name and 60-word prompts
    col = {"direct": "gpt4_rewrite_name", "lure_desc60": "gpt4_rewrite_desc60"}.get(attack)
    if job.dataset == "copycat_characters" and col and item.get(col):
        return {"rewrite": item[col], "rewriter": "gpt-4 (He et al. ICLR'25 release)"}
    return None


def _slug(x) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(x)).strip("-")


def _summaries(probe_out: dict, window: dict, b: int) -> dict:
    rec = {}
    if "noise" in probe_out and probe_out["noise"].size:
        nn = probe_out["noise"][:, b]
        rec.update(noise_first=float(nn[0]), noise_mean10=float(nn[:10].mean()), noise_mean=float(nn.mean()))
    if "attention" in probe_out and probe_out["attention"].size:
        att = probe_out["attention"]                      # (steps, layers, B, 77)
        n_tok = min(window.get("n_tokens", 75), 75)
        user = (window["user_tok_start"], window["user_tok_end"]) if "user_tok_start" in window else None
        for tag, mass in (("t0", att[0, :, b].mean(0)), ("avg", att[:, :, b].mean((0, 1)))):
            for k, v in attention_summary(mass, n_tok, user).items():
                rec[f"{k}_{tag}"] = float(v)
    return rec


def run_job(pipe, spec, job: Job, out_dir: Path, shard: tuple[int, int] = (0, 1), log=print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "probes", "x0"):
        (out_dir / sub).mkdir(exist_ok=True)
    steps = job.steps or spec.steps
    guidance = job.guidance or spec.guidance
    res = job.resolution or spec.resolution
    (out_dir / "job.json").write_text(json.dumps({
        "job": asdict(job), "model": asdict(spec), "steps": steps, "guidance": guidance, "resolution": res,
        "device": str(pipe.device), "dtype": str(pipe.unet.dtype if hasattr(pipe, "unet") else ""),
        "torch": torch.__version__, "platform": platform.platform(), "shard": shard,
    }, indent=1))
    records_path = out_dir / ("records.jsonl" if shard == (0, 1) else f"records.{shard[0]}of{shard[1]}.jsonl")
    tokenizer = getattr(pipe, "tokenizer", None)
    rewrites = load_rewrites(job.rewrites)
    has_unet = hasattr(pipe, "unet")
    items = select_items(job, shard)
    log(f"[{job.name}] {len(items)} items x {len(job.attacks)} attacks x {len(job.defenses)} defenses "
        f"x {len(job.seeds)} seeds on {spec.key} ({steps} steps, cfg {guidance}, {res}px)")
    skipped = set()
    n_done = 0
    t_start = time.time()
    (out_dir / "refused").mkdir(exist_ok=True)
    # D0 records by (item, attack, seed): a request the blocklist lets through reaches the model
    # unchanged, so its image is the D0 image (copied instead of regenerated)
    d0 = {}
    if records_path.exists():
        for line in open(records_path):
            r = json.loads(line)
            if r.get("defense") == "none":
                d0[(str(r["item_id"]), r["attack"], r["seed"])] = r
    with open(records_path, "a") as rec_f:
        for item in items:
            for attack in job.attacks:
                try:
                    adv = make_attack(attack, item["prompt"], item, seed=0, tokenizer=tokenizer)
                except KeyError as e:  # attack not defined for this dataset (e.g. lures on LAION captions)
                    if (job.dataset, attack) not in skipped:
                        log(f"  skip {attack} on {job.dataset}: {e}")
                        skipped.add((job.dataset, attack))
                    continue
                for defense in job.defenses:
                    meta = dict(item)
                    if defense.startswith("llm_rewrite"):
                        rw = _rewrite_for(job, item, attack, rewrites)
                        if rw is None:
                            if (attack, defense) not in skipped:
                                log(f"  skip {defense} x {attack}: no rewrite available")
                                skipped.add((attack, defense))
                            continue
                        meta.update(rw)
                    d = apply_defense(defense, adv["prompt"], meta, seed=0)
                    cond = f"{_slug(job.dataset)}-{_slug(item['id'])}_{ATTACK_CODES[attack]}_{DEFENSE_CODES[defense]}"
                    todo = [s for s in job.seeds if not (out_dir / "images" / f"{cond}_s{s}.png").exists()
                            and not (out_dir / "refused" / f"{cond}_s{s}.txt").exists()]
                    if not todo:
                        continue
                    base = {"dataset": job.dataset, "item_id": item["id"], "target": item.get("target", ""),
                            "attack": attack, "attack_code": ATTACK_CODES[attack], "defense": defense,
                            "defense_code": DEFENSE_CODES[defense], "model": spec.key,
                            "user_prompt": item["prompt"], "adv_prompt": adv["prompt"]}
                    if d.info.get("refused"):
                        for seed in todo:
                            rel = f"refused/{cond}_s{seed}.txt"
                            (out_dir / rel).write_text(d.info["blocked_term"])
                            rec_f.write(json.dumps({**base, "seed": seed, "image": rel, "refused": True,
                                                    "blocked_term": d.info["blocked_term"], "model_prompt": "",
                                                    "negative_prompt": ""}, default=_json_default) + "\n")
                        rec_f.flush()
                        continue
                    if defense == "blocklist" and all((str(item["id"]), attack, s) in d0 for s in todo):
                        for seed in todo:
                            src = d0[(str(item["id"]), attack, seed)]
                            rel = f"images/{cond}_s{seed}.png"
                            shutil.copyfile(out_dir / src["image"], out_dir / rel)
                            rec_f.write(json.dumps({**src, **base, "seed": seed, "image": rel, "refused": False,
                                                    "reused_from": src["image"]}, default=_json_default) + "\n")
                        rec_f.flush()
                        continue
                    window = clip_token_window(tokenizer, d.prompt, d.user_span) if tokenizer else {}
                    gens = [torch.Generator("cpu").manual_seed(s) for s in todo]
                    t0 = time.time()
                    with probes_attached(pipe, noise="noise" in job.probes and has_unet,
                                         attention="attention" in job.probes and has_unet,
                                         x0_steps=job.x0_steps if has_unet else None) as P:
                        images = pipe(prompt=d.prompt, negative_prompt=d.negative_prompt or None,
                                      num_images_per_prompt=len(todo), generator=gens, num_inference_steps=steps,
                                      guidance_scale=guidance, height=res, width=res).images
                        probe_out = {k: v.result() for k, v in P.items() if k in ("noise", "attention")}
                        x0 = P["x0"].decode(pipe) if "x0" in P else {}
                    dt = (time.time() - t0) / len(todo)
                    _check_finite(cond, images, probe_out)
                    if probe_out:
                        np.savez_compressed(out_dir / "probes" / f"{cond}.npz", seeds=np.array(todo),
                                            **{k: v.astype(np.float16) for k, v in probe_out.items()})
                    for b, (seed, img) in enumerate(zip(todo, images)):
                        rel = f"images/{cond}_s{seed}.png"
                        img.save(out_dir / rel)
                        x0_paths = []
                        for step, ims in x0.items():
                            p = f"x0/{cond}_s{seed}_t{step:02d}.jpg"
                            ims[b].resize((256, 256)).save(out_dir / p, quality=90)
                            x0_paths.append(p)
                        rec = {
                            **base, "seed": seed, "model_prompt": d.prompt, "refused": False,
                            "negative_prompt": d.negative_prompt, "image": rel, "x0": x0_paths,
                            "sec_per_image": round(dt, 2), **window, **_summaries(probe_out, window, b),
                        }
                        if defense == "none":
                            d0[(str(item["id"]), attack, seed)] = rec
                        rec_f.write(json.dumps(rec, default=_json_default) + "\n")
                    rec_f.flush()
                    n_done += len(todo)
                    if n_done % 20 < len(todo):
                        rate = n_done / (time.time() - t_start)
                        log(f"  {n_done} images ({rate * 60:.1f}/min) last: {cond}")
    log(f"[{job.name}] done: {n_done} new images -> {out_dir}")
    return records_path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _check_finite(cond: str, images, probe_out: dict) -> None:
    """Fail fast instead of logging black images (fp16 overflow / backend bugs give NaN latents)."""
    noise = probe_out.get("noise")
    if noise is not None and noise.size and not np.isfinite(noise).all():
        raise FloatingPointError(f"{cond}: non-finite noise predictions (fp16/attention backend problem)")
    if all(np.asarray(im).max() == 0 for im in images):
        raise FloatingPointError(f"{cond}: all generated images are black (NaN latents)")
