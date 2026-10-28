"""How much identity do LLM rewrites keep? (no GPU needed)

For every CopyCat character: the name prompt rewritten by GPT-4 under the DALL-E policy (He et al.'s
released D4 rewrite), the 60-word lure rewritten the same way, and the lure itself, checked for the
character name and for its identifying keywords (union of the LM-ranked and LAION co-occurrence lists).

    python scripts/analyze_rewrite_leakage.py   # -> outputs/analysis/gpt4_rewrite_leakage.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.paths import output_root  # noqa: E402
from guardrail_depth.rewriter import leakage  # noqa: E402


def main() -> None:
    rows = []
    for r in load_manifest("copycat_characters").to_dict("records"):
        kws = [k for col in ("kw_lm5", "kw_laion5") for k in r[col].split(",")]
        for col, source in (("gpt4_rewrite_name", "GPT-4 rewrite of name"), ("gpt4_rewrite_desc60", "GPT-4 rewrite of lure"),
                            ("desc60", "60-word lure")):
            if r[col]:
                rows.append({"target": r["target"], "source": source, **leakage(r[col], r["target"], kws)})
    df = pd.DataFrame(rows)
    out = output_root() / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "gpt4_rewrite_leakage.csv", index=False)
    summ = df.groupby("source", sort=False).agg(n=("target", "size"), name_leak=("name_leak", "sum"),
                                                kw_hits=("kw_hits", "mean"), kw_total=("kw_total", "mean"),
                                                any_kw=("kw_hits", lambda x: (x > 0).mean()))
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
