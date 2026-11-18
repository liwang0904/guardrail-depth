"""Concept screen of the Demo 3 gate: one decision per (item, attack) of a job, written to a CSV that
the generation job reads (like the D4 rewrites).

    python scripts/run_screen.py --config configs/experiments/demo3.yaml --job sd14_copycat_gate \
        --out $GD_OUTPUT_ROOT/screens/demo3_copycat_characters.csv
    python scripts/run_screen.py --dataset copycat_characters --split dev --attacks direct,lure_desc60 --llm rewriter_reasoning_small

Stage 1 is an exact protected-name check (after Unicode confusable folding); stage 2 asks the LLM
(Qwen3-8B, thinking off, greedy) to map the request onto the closed protected list. The screen never
sees the attack texts at design time: its worked examples use dev-split characters only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.attacks import make_attack  # noqa: E402
from guardrail_depth.data import items as manifest_items  # noqa: E402
from guardrail_depth.gate import ConceptScreen  # noqa: E402
from guardrail_depth.models import pick_device, registry  # noqa: E402
from guardrail_depth.paths import output_root  # noqa: E402


def job_requests(config: str, job_name: str) -> tuple[str, list[dict], list[str]]:
    import yaml

    exp = yaml.safe_load(open(config))
    job = {**exp.get("defaults", {}), **next(j for j in exp["jobs"] if j["name"] == job_name)}
    ids = job.get("ids")
    rows = manifest_items(job["dataset"], None if ids else job.get("split"), ids)
    if job.get("limit"):
        rows = rows[: job["limit"]]
    return job["dataset"], rows, job["attacks"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="rewriter_reasoning", help="key under `auxiliary` in configs/models.yaml")
    ap.add_argument("--config")
    ap.add_argument("--job", help="comma-separated job names in --config (their requests are screened together)")
    ap.add_argument("--dataset", default="copycat_characters")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ids")
    ap.add_argument("--attacks", default="direct")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cards", action="store_true", help="also list LLM-written appearance cards (dev ablation; off)")
    ap.add_argument("--device")
    args = ap.parse_args()

    batches = []
    if args.config:
        for name in args.job.split(","):
            batches.append(job_requests(args.config, name))
    else:
        ids = [int(x) if x.isdigit() else x for x in args.ids.split(",")] if args.ids else None
        batches.append((args.dataset, manifest_items(args.dataset, None if ids else args.split, ids), args.attacks.split(",")))

    cfg = registry()["auxiliary"][args.llm]
    screen = ConceptScreen(cfg["hf_id"], cfg["revision"], pick_device(args.device), use_cards=args.cards)
    tok = None
    rows, done = [], set()
    t0 = time.time()
    for dataset, items, attacks in batches:
        for it in items:
            for attack in attacks:
                if (dataset, str(it["id"]), attack) in done:
                    continue
                done.add((dataset, str(it["id"]), attack))
                if attack.startswith("truncate_"):
                    # padding attacks depend on the T2I model's CLIP tokenizer
                    if tok is None:
                        from transformers import CLIPTokenizer
                        sd = registry()["text_to_image"]["sd14"]
                        tok = CLIPTokenizer.from_pretrained(sd["hf_id"], subfolder="tokenizer", revision=sd["revision"])
                try:
                    request = make_attack(attack, it["prompt"], it, seed=0, tokenizer=tok)["prompt"]
                except KeyError:
                    continue
                r = screen(request)
                rows.append({"dataset": dataset, "item_id": it["id"], "attack": attack, "target": it.get("target", ""),
                             "request": request, **r})
                if len(rows) % 50 == 0:
                    print(f"  {len(rows)} screened ({len(rows) / (time.time() - t0):.1f}/s)", flush=True)
    df = pd.DataFrame(rows)
    out = args.out or output_root() / "screens" / f"screen_{batches[0][0]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    flagged = df[df["match"] != ""]
    print(f"wrote {len(df)} decisions -> {out}; flagged {len(flagged)} "
          f"({flagged['stage'].value_counts().to_dict()}), {df['screen_seconds'].mean():.2f} s/request")
    print(screen_report(df).to_string(index=False))


def screen_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and attack: share of protected requests matched to their own target (recall) and share of
    unprotected requests matched to anything (false matches: captions, near-miss requests, public-domain styles)."""
    from guardrail_depth.gate import protected_concepts

    protected = set(protected_concepts()["name"])
    d = df.assign(target=df["target"].fillna("").astype(str), match=df["match"].fillna("").astype(str))
    d["protected"] = d["target"].isin(protected)
    rows = []
    for (ds, atk), x in d.groupby(["dataset", "attack"]):
        p, n = x[x["protected"]], x[~x["protected"]]
        rows.append({"dataset": ds, "attack": atk, "n_protected": len(p),
                     "recall": (p["match"] == p["target"]).mean() if len(p) else float("nan"),
                     "wrong_concept": ((p["match"] != "") & (p["match"] != p["target"])).mean() if len(p) else float("nan"),
                     "n_unprotected": len(n), "false_match": (n["match"] != "").mean() if len(n) else float("nan")})
    return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    main()
