"""Experiment loop: items x attacks x defenses x seeds -> images + per-image records + probe arrays.

Output layout (one directory per job):
    <out>/images/<cond>_s<seed>.png      generated images
    <out>/probes/<cond>.npz              per-step probe arrays (float16)
    <out>/x0/<cond>_s<seed>_t<step>.jpg  predicted-x0 snapshots (optional)
    <out>/records.jsonl                  one line per image: prompts, token window, probe summaries
    <out>/job.json                       resolved job config + environment
Re-running a job skips images that already exist, so SLURM jobs can be resubmitted.

The Demo 3 gate (`gated`, `gated_pre`, see gate.py) decides per request which Demo 2 defense to apply.
When that defense hands the model a (prompt, negative prompt, seed) that an earlier run of the same
model and settings already rendered (`reuse:` in the job), the image is copied instead of regenerated:
the computation is identical, so the gate is compared with the baselines on the very same images.
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
from .paths import output_root
from .probes import attention_summary, probes_attached

GATED = ("gated", "gated_pre")


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
    screens: list[str] = field(default_factory=list)   # concept-screen CSVs (scripts/run_screen.py), for the gate
    reuse: list[str] = field(default_factory=list)     # run dirs (under GD_OUTPUT_ROOT) whose images may be reused
    gate: dict = field(default_factory=dict)           # thresholds, see configs/gate.yaml


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


def load_screens(paths: list[str]) -> dict:
    """(dataset, item_id, attack) -> concept-screen decision from scripts/run_screen.py CSVs."""
    table = {}
    for p in paths:
        p = os.path.expandvars(p)
        if not os.path.exists(p):
            print(f"  screen file not found (gate skipped for now): {p}")
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


def _negative(spec, negative: str):
    # SD/SDXL: None keeps the pipelines' own empty-negative behaviour (SDXL zeroes it), as in Demo 2;
    # PixArt tokenizes the negative prompt and needs a string.
    return negative if negative else ("" if getattr(spec, "family", "") == "pixart" else None)


def _render(pipe, spec, job: Job, d, seeds: list[int], steps: int, guidance: float, res: int):
    """One pipeline call for all `seeds` of a (prompt, negative prompt) pair, with the job's probes."""
    has_unet = hasattr(pipe, "unet")
    gens = [torch.Generator("cpu").manual_seed(s) for s in seeds]
    t0 = time.time()
    with probes_attached(pipe, noise="noise" in job.probes and has_unet,
                         attention="attention" in job.probes and has_unet,
                         x0_steps=job.x0_steps if has_unet else None) as P:
        images = pipe(prompt=d.prompt, negative_prompt=_negative(spec, d.negative_prompt),
                      num_images_per_prompt=len(seeds), generator=gens, num_inference_steps=steps,
                      guidance_scale=guidance, height=res, width=res).images
        probe_out = {k: v.result() for k, v in P.items() if k in ("noise", "attention")}
        x0 = P["x0"].decode(pipe) if "x0" in P else {}
    return images, probe_out, x0, (time.time() - t0) / len(seeds)


def _token_window(tokenizer, spec, d) -> dict:
    if tokenizer is None:
        return {}
    return clip_token_window(tokenizer, d.prompt, d.user_span, max_len=120 if getattr(spec, "family", "") == "pixart" else 77)


class ImageCache:
    """(model prompt, negative prompt, seed) -> an image some run of the same model and settings
    already rendered, with that image's record. Indexes this run plus the job's `reuse:` dirs."""

    def __init__(self, out_dir: Path, job: Job, spec, steps: int, guidance: float, res: int, log=print):
        self.table: dict[tuple, tuple[Path, dict]] = {}
        dirs = [out_dir] + [output_root() / r for r in job.reuse]
        for d in dirs:
            if not d.exists():
                log(f"  reuse dir not found: {d}")
                continue
            if d != out_dir and (d / "job.json").exists():
                j = json.loads((d / "job.json").read_text())
                same = (j.get("model", {}).get("key") == spec.key and j.get("steps") == steps
                        and float(j.get("guidance")) == float(guidance) and j.get("resolution") == res)
                if not same:
                    log(f"  not reusing {d}: different model or settings")
                    continue
            n = 0
            for f in sorted(d.glob("records*.jsonl")):
                for line in open(f):
                    r = json.loads(line)
                    if r.get("refused") or not r.get("image") or "model_prompt" not in r:
                        continue
                    if (d / r["image"]).exists():
                        self.table[(r["model_prompt"], r.get("negative_prompt", "") or "", int(r["seed"]))] = (d, r)
                        n += 1
            log(f"  image cache: {n} images from {d}")

    def get(self, prompt: str, negative: str, seed: int):
        return self.table.get((prompt, negative or "", int(seed)))

    def add(self, run_dir: Path, rec: dict):
        self.table[(rec["model_prompt"], rec.get("negative_prompt", "") or "", int(rec["seed"]))] = (run_dir, rec)


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
    items = select_items(job, shard)
    log(f"[{job.name}] {len(items)} items x {len(job.attacks)} attacks x {len(job.defenses)} defenses "
        f"x {len(job.seeds)} seeds on {spec.key} ({steps} steps, cfg {guidance}, {res}px)")
    skipped = set()
    n_done = 0
    t_start = time.time()
    (out_dir / "refused").mkdir(exist_ok=True)
    gate = _GateRunner(pipe, spec, job, out_dir, steps, guidance, res, rewrites, log) \
        if any(d in GATED for d in job.defenses) else None
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
                    cond = f"{_slug(job.dataset)}-{_slug(item['id'])}_{ATTACK_CODES[attack]}_{DEFENSE_CODES[defense]}"
                    todo = [s for s in job.seeds if not (out_dir / "images" / f"{cond}_s{s}.png").exists()
                            and not (out_dir / "refused" / f"{cond}_s{s}.txt").exists()]
                    if not todo:
                        continue
                    base = {"dataset": job.dataset, "item_id": item["id"], "target": item.get("target", ""),
                            "attack": attack, "attack_code": ATTACK_CODES[attack], "defense": defense,
                            "defense_code": DEFENSE_CODES[defense], "model": spec.key,
                            "user_prompt": item["prompt"], "adv_prompt": adv["prompt"]}
                    if defense in GATED:
                        n_done += gate.run(item, attack, adv["prompt"], defense, cond, todo, base, rec_f, skipped)
                        continue
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
                    window = _token_window(tokenizer, spec, d)
                    images, probe_out, x0, dt = _render(pipe, spec, job, d, todo, steps, guidance, res)
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
                        if gate is not None:
                            gate.cache.add(out_dir, rec)
                        rec_f.write(json.dumps(rec, default=_json_default) + "\n")
                    rec_f.flush()
                    n_done += len(todo)
                    if n_done % 20 < len(todo):
                        rate = n_done / (time.time() - t_start)
                        log(f"  {n_done} images ({rate * 60:.1f}/min) last: {cond}")
    if gate is not None:
        log(f"[{job.name}] gate: {gate.stats}")
    log(f"[{job.name}] done: {n_done} new images -> {out_dir}")
    return records_path


# Probe summaries copied from a reused image's record (they describe that same computation).
_PROBE_KEYS = re.compile(r"^(noise_|attn_|n_tokens|n_truncated|fits|user_tok_|suffix_tokens_)")


class _GateRunner:
    """Runs the Demo 3 gate inside a generation job: screens, plan, render-or-reuse, output check."""

    def __init__(self, pipe, spec, job: Job, out_dir: Path, steps: int, guidance: float, res: int, rewrites: dict, log):
        from . import gate as G

        self.G, self.pipe, self.spec, self.job, self.out_dir = G, pipe, spec, job, out_dir
        self.steps, self.guidance, self.res, self.rewrites, self.log = steps, guidance, res, rewrites, log
        self.screens = load_screens(job.screens)
        self.cache = ImageCache(out_dir, job, spec, steps, guidance, res, log)
        self.tokenizer = getattr(pipe, "tokenizer", None)
        cfg = {**G.load_gate_config(), **(job.gate or {})}
        self.cfg = cfg
        # the exact-copy screen applies to the SD 1.x / 2.x models the memorized prompts were found on
        self.tau = cfg.get("tau_mem", {}).get(spec.key) if isinstance(cfg.get("tau_mem"), dict) else cfg.get("tau_mem")
        self.mem_on = self.tau is not None and getattr(spec, "family", "") in ("sd1", "sd2") and hasattr(pipe, "unet")
        self.mem_cache: dict[str, float] = {}
        self.check = None
        if "gated" in job.defenses:
            from .metrics import CLIPScorer

            clip = CLIPScorer(cfg.get("check_clip", "ViT-L-14"), cfg.get("check_pretrained", "openai"), pipe.device)
            self.check = G.OutputCheck(clip, float(cfg.get("check_threshold", 0.5)))
        self.stats = {"requests": 0, "concept": 0, "memorization": 0, "output_regen": 0, "rendered": 0, "reused": 0}
        log(f"  gate: {len(self.screens)} screen decisions, memorization screen "
            f"{'tau=%.2f' % self.tau if self.mem_on else 'off'}, output check "
            f"{'threshold=%.2f' % self.check.threshold if self.check else 'off'}")

    def mem_fn(self, prompts: list[str]) -> np.ndarray:
        todo = [p for p in dict.fromkeys(prompts) if p not in self.mem_cache]
        if todo:
            s = self.G.memorization_scores(self.pipe, todo, n_images=int(self.cfg.get("mem_n_images", 4)),
                                           steps=self.steps, batch_size=4 if self.pipe.device.type == "cuda" else 1)
            self.mem_cache.update(zip(todo, map(float, s)))
        return np.array([self.mem_cache[p] for p in prompts])

    def produce(self, d, seeds: list[int], cond: str, tag: str) -> dict:
        """seed -> (PIL image, record fields): copied from the cache when this exact computation exists."""
        out, missing = {}, []
        for s in seeds:
            hit = self.cache.get(d.prompt, d.negative_prompt, s)
            if hit is None:
                missing.append(s)
                continue
            src_dir, src = hit
            from PIL import Image

            fields = {k: v for k, v in src.items() if _PROBE_KEYS.match(k)}
            fields.update(reused_from=str((src_dir / src["image"]).relative_to(output_root()))
                          if src_dir.is_relative_to(output_root()) else str(src_dir / src["image"]),
                          sec_per_image=src.get("sec_per_image"), rendered=False)
            out[s] = (Image.open(src_dir / src["image"]).convert("RGB"), fields)
            self.stats["reused"] += 1
        if missing:
            images, probe_out, _, dt = _render(self.pipe, self.spec, self.job, d, missing, self.steps, self.guidance, self.res)
            _check_finite(cond, images, probe_out)
            window = _token_window(self.tokenizer, self.spec, d)
            for b, (s, img) in enumerate(zip(missing, images)):
                rel = f"images/{cond}{tag}_s{s}.png"
                img.save(self.out_dir / rel)
                rec = {"model_prompt": d.prompt, "negative_prompt": d.negative_prompt, "seed": s, "image": rel,
                       "sec_per_image": round(dt, 2), **window, **_summaries(probe_out, window, b)}
                self.cache.add(self.out_dir, rec)
                out[s] = (img, {k: v for k, v in rec.items() if _PROBE_KEYS.match(k)} | {"sec_per_image": round(dt, 2),
                                                                                          "rendered": True})
                self.stats["rendered"] += 1
        return out

    def run(self, item: dict, attack: str, request: str, defense: str, cond: str, todo: list[int], base: dict,
            rec_f, skipped: set) -> int:
        key = (self.job.dataset, str(item["id"]), attack)
        screen = self.screens.get(key)
        if screen is None:
            if ("screen", attack) not in skipped:
                self.log(f"  skip {defense} x {attack}: no concept-screen decision (run scripts/run_screen.py)")
                skipped.add(("screen", attack))
            return 0
        rw = _rewrite_for(self.job, item, attack, self.rewrites)
        rewrite = rw["rewrite"] if rw else None
        t0 = time.time()
        plan = self.G.plan(request, {"match": screen.get("match", ""), "kind": screen.get("kind", "")}, rewrite,
                           self.mem_fn if self.mem_on else None, self.tau if self.mem_on else None)
        mem_seconds = time.time() - t0
        self.stats["requests"] += 1
        self.stats["concept"] += int("concept" in plan.reason)
        self.stats["memorization"] += int("memorization" in plan.reason)
        first = self.produce(plan.defended, todo, cond, "_pre" if defense == "gated" else "")
        gate_info = {"gate_applied": plan.applied, "gate_target": plan.target, "gate_reason": plan.reason,
                     "screen_stage": screen.get("stage", ""), "screen_reply": screen.get("reply", ""),
                     "screen_seconds": float(screen.get("screen_seconds", 0) or 0), "mem_seconds": round(mem_seconds, 3),
                     "rewrite_available": bool(rewrite), **{k: v for k, v in plan.info.items() if k != "mem_rung_scores"},
                     "mem_rung_scores": json.dumps(plan.info.get("mem_rung_scores", {}))}
        for seed in todo:
            img, fields = first[seed]
            final_plan, final_img, final_fields = plan, img, fields
            rec = {**base, **gate_info, "seed": seed, "refused": False, "n_generations": 1}
            if self.check is not None:
                c = self.check([img])[0]
                rec.update({f"pre_{k}": v for k, v in c.items()})
                # the output check covers what the screens missed: a request already mitigated for a
                # detected character keeps that mitigation (CLIP can misname the mitigated image)
                if c["check_flag"] and not self.G.screened_character(plan):
                    final_plan = self.G.replan_for_output(request, c["check_character"], rewrite, plan)
                    again = self.produce(final_plan.defended, [seed], cond, "")
                    final_img, final_fields = again[seed]
                    rec["n_generations"] = 2
                    rec["pre_image"] = fields.get("reused_from") or f"images/{cond}_pre_s{seed}.png"
                    self.stats["output_regen"] += 1
                    c2 = self.check([final_img])[0]
                    rec.update(c2)
                else:
                    rec.update(c)
            rel = f"images/{cond}_s{seed}.png"
            final_img.save(self.out_dir / rel)
            rec.update(final_fields)
            rec.update({"image": rel, "model_prompt": final_plan.defended.prompt,
                        "negative_prompt": final_plan.defended.negative_prompt,
                        "gate_final_applied": final_plan.applied, "gate_final_target": final_plan.target,
                        "gate_final_reason": final_plan.reason})
            rec_f.write(json.dumps(rec, default=_json_default) + "\n")
        rec_f.flush()
        return len(todo)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
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
