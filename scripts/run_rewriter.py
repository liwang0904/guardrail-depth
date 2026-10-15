"""D4 prompt rewriting with an open LLM, logging reasoning traces, entropy and identity leakage.

    python scripts/run_rewriter.py --llm rewriter_reasoning_small --thinking \
        --dataset copycat_characters --split dev --attacks direct,lure_desc60

Output CSV columns feed both the generation runner (`rewrites:` in an experiment config) and the
reasoning-pattern analysis: did the model recognize the protected character in its reasoning,
does the final rewrite still leak the name or the identifying keywords, and is the model's
token-level entropy different when it leaks?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.attacks import make_attack  # noqa: E402
from guardrail_depth.data import items as manifest_items  # noqa: E402
from guardrail_depth.models import pick_device, registry  # noqa: E402
from guardrail_depth.paths import output_root  # noqa: E402
from guardrail_depth.rewriter import LLMRewriter, leakage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="rewriter_reasoning", help="key under `auxiliary` in configs/models.yaml")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--dataset", default="copycat_characters")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ids")
    ap.add_argument("--attacks", default="direct,lure_desc60")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = registry()["auxiliary"][args.llm]
    rw = LLMRewriter(cfg["hf_id"], cfg["revision"], pick_device(args.device), thinking=args.thinking,
                     max_new_tokens=args.max_new_tokens)
    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    rows = []
    for it in manifest_items(args.dataset, None if ids else args.split, ids):
        keywords = [k for col in ("kw_lm5", "kw_laion5") for k in str(it.get(col, "")).split(",")]
        for attack in args.attacks.split(","):
            request = make_attack(attack, it["prompt"], it)["prompt"]
            r = rw.rewrite(request, seed=args.seed)
            final_leak = leakage(r["rewrite"], it.get("target", ""), keywords)
            reason_leak = leakage(r["reasoning"], it.get("target", ""), keywords)
            rows.append({"dataset": args.dataset, "item_id": it["id"], "target": it.get("target", ""),
                         "attack": attack, "request": request, **r,
                         "rewrite_name_leak": final_leak["name_leak"], "rewrite_kw_hits": final_leak["kw_hits"],
                         "rewrite_kw_list": final_leak["kw_hit_list"], "kw_total": final_leak["kw_total"],
                         "reasoning_names_target": reason_leak["name_leak"]})
            print(f"[{it.get('target', it['id'])} / {attack}] name_leak={final_leak['name_leak']} "
                  f"kw={final_leak['kw_hits']}/{final_leak['kw_total']} recognized={reason_leak['name_leak']} "
                  f"H_final={r['entropy_final_mean']:.2f} -> {r['rewrite'][:90]}", flush=True)
    out = args.out or output_root() / "rewrites" / f"{args.llm}{'_think' if args.thinking else ''}_{args.dataset}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rewrites -> {out}")


if __name__ == "__main__":
    main()
