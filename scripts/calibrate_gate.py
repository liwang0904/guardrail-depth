"""Calibrate the Demo 3 gate on the dev split only, before any test run; writes configs/gate.yaml.

    python scripts/calibrate_gate.py --models sd14,sd2base \
        --check-runs $GD_OUTPUT_ROOT/demo3/sd14_dev_copycat $GD_OUTPUT_ROOT/demo3/sd14_dev_benign

1. Memorization threshold per model: the scan score (max over request segments) of the dev memorized
   prompts and the dev benign captions; tau is the (1 - fpr_target) quantile of the benign scores, so
   about `fpr_target` of ordinary captions get a template. Recall on dev memorized prompts (plain and
   under the padding / override attacks) is printed, not optimized.
2. Output-check threshold: CLIP ViT-L/14 zero-shot probability of the top protected character on dev
   images; the smallest threshold that flags at most `check_fpr_target` of the benign dev images
   (captions and generic near-miss requests), with the recall on dev character images printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth import gate as G  # noqa: E402
from guardrail_depth.attacks import make_attack  # noqa: E402
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.models import load_t2i, pick_device  # noqa: E402
from guardrail_depth.paths import CONFIG_DIR, output_root  # noqa: E402

DEFAULTS = {"mem_fpr_target": 0.02, "mem_n_images": 4, "check_clip": "ViT-L-14", "check_pretrained": "openai",
            "check_fpr_target": 0.02, "screen_llm": "rewriter_reasoning"}


def calibrate_memorization(model: str, fpr: float, n_images: int, device) -> tuple[float, dict]:
    pipe, spec = load_t2i(model, device)
    mem_name = {"sd14": "memorized_sd1", "sd2base": "memorized_sd2"}[model]
    mem = load_manifest(mem_name)
    mem = mem[mem.split == "dev"]
    ctl = load_manifest("unmemorized_laion")
    ctl = ctl[ctl.split == "dev"]
    bs = 4 if device.type == "cuda" else 1
    cache: dict[str, float] = {}

    def score(prompts):
        todo = [p for p in dict.fromkeys(prompts) if p not in cache]
        if todo:
            s = G.memorization_scores(pipe, todo, n_images=n_images, steps=spec.steps, batch_size=bs)
            cache.update(zip(todo, map(float, s)))
        return np.array([cache[p] for p in prompts])

    def scan(text):
        return float(score(G.request_segments(text)).max())

    benign = np.array([scan(p) for p in ctl["prompt"]])
    tau = float(np.quantile(benign, 1 - fpr))
    rows = []
    for attack in ("direct", "truncate_repeat", "override"):
        for r in mem.to_dict("records"):
            req = make_attack(attack, r["prompt"], {}, tokenizer=pipe.tokenizer)["prompt"]
            s_full, s_scan = float(score([req])[0]), scan(req)
            rows.append({"attack": attack, "mem_type": r["mem_type"], "full": s_full, "scan": s_scan})
    d = pd.DataFrame(rows)
    rec = (d.assign(flag_full=d.full > tau, flag_scan=d.scan > tau)
             .groupby(["attack", "mem_type"])[["flag_full", "flag_scan"]].mean().round(3))
    report = {"tau": round(tau, 4), "benign_n": int(len(benign)), "benign_flag_rate": float((benign > tau).mean()),
              "benign_quantiles": {q: round(float(np.quantile(benign, q)), 3) for q in (0.5, 0.9, 0.95, 0.98, 0.99)},
              "dev_recall": {f"{a}/{t}": {"full": float(v.flag_full), "scan": float(v.flag_scan)}
                             for (a, t), v in rec.iterrows()}}
    print(f"[{model}] tau={tau:.3f} (benign dev flag rate {report['benign_flag_rate']:.3f})\n{rec.to_string()}")
    del pipe
    return tau, report


def calibrate_check(runs: list[Path], fpr: float, clip_name: str, pretrained: str, device) -> tuple[float, dict]:
    from PIL import Image

    from guardrail_depth.metrics import CLIPScorer

    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import load_records

    clip = CLIPScorer(clip_name, pretrained, device)
    check = G.OutputCheck(clip, threshold=1.0)
    pos, neg = [], []
    for run in runs:
        df = load_records(run)
        df = df[~df["refused"].fillna(False).astype(bool)] if "refused" in df else df
        res = []
        for i in range(0, len(df), 32):
            ims = [Image.open(run / p).convert("RGB") for p in df["image"].iloc[i:i + 32]]
            res += check(ims)
        df = df.assign(char=[r["check_character"] for r in res], prob=[r["check_prob"] for r in res])
        if df["dataset"].iloc[0] == "copycat_characters":
            pos.append(df.assign(hit=lambda x: x["char"] == x["target"]))
        else:
            neg.append(df)
    pos = pd.concat(pos) if pos else pd.DataFrame(columns=["prob", "hit", "attack"])
    neg = pd.concat(neg) if neg else pd.DataFrame(columns=["prob", "char"])
    neg_scores = np.sort(np.where(neg["char"] != "", neg["prob"], 0.0)) if len(neg) else np.zeros(1)
    k = int(np.floor(fpr * len(neg_scores)))
    thr = float(max(0.5, neg_scores[::-1][k] + 1e-6)) if len(neg_scores) else 0.5
    flagged_pos = pos[(pos["prob"] >= thr) & pos["hit"]] if len(pos) else pos
    report = {"threshold": round(thr, 4), "benign_images": int(len(neg)),
              "benign_flag_rate": float((neg_scores >= thr).mean()) if len(neg) else None,
              "character_images": int(len(pos)),
              "recall_by_attack": (pos.assign(f=(pos["prob"] >= thr) & pos["hit"]).groupby("attack")["f"].mean().round(3)
                                   .to_dict() if len(pos) else {})}
    print(f"[output check] threshold={thr:.3f} benign flag rate {report['benign_flag_rate']} "
          f"recall {report['recall_by_attack']} ({len(flagged_pos)} of {len(pos)} character images)")
    return thr, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="sd14,sd2base")
    ap.add_argument("--check-runs", nargs="*", type=Path, default=[])
    ap.add_argument("--mem-fpr", type=float, default=DEFAULTS["mem_fpr_target"])
    ap.add_argument("--check-fpr", type=float, default=DEFAULTS["check_fpr_target"])
    ap.add_argument("--out", type=Path, default=CONFIG_DIR / "gate.yaml")
    ap.add_argument("--device")
    args = ap.parse_args()
    device = pick_device(args.device)
    cfg = {**DEFAULTS, **(G.load_gate_config(args.out) if args.out.exists() else {})}
    cfg["mem_fpr_target"], cfg["check_fpr_target"] = args.mem_fpr, args.check_fpr
    report = {}
    tau = dict(cfg.get("tau_mem") or {})
    for m in [m for m in args.models.split(",") if m]:
        tau[m], report[f"mem_{m}"] = calibrate_memorization(m, args.mem_fpr, cfg["mem_n_images"], device)
    cfg["tau_mem"] = {k: round(float(v), 4) for k, v in tau.items()}
    if args.check_runs:
        thr, report["check"] = calibrate_check(args.check_runs, args.check_fpr, cfg["check_clip"],
                                               cfg["check_pretrained"], device)
        cfg["check_threshold"] = round(thr, 4)
    header = ("# Demo 3 gate settings, calibrated on the dev split only (scripts/calibrate_gate.py), before any test run.\n"
              "# tau_mem: memorization-scan threshold per model (about mem_fpr_target of dev captions flagged);\n"
              "# check_threshold: CLIP probability above which the output check calls a protected character.\n")
    args.out.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    rep = output_root() / "screens" / "gate_calibration.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(rep.read_text()) if rep.exists() else {}
    rep.write_text(json.dumps({**old, **report}, indent=1))
    print(f"wrote {args.out} and {rep}")


if __name__ == "__main__":
    main()
