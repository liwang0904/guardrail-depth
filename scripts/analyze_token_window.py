"""How much of each guardrail actually reaches the text encoder? (no GPU needed)

SD 1.x/2.x encode at most 75 prompt tokens; everything after is silently dropped. For every
dataset x attack x template defense we report how often the defended prompt is truncated and
what fraction of the guardrail text that follows the user's request survives.

    python scripts/analyze_token_window.py   # -> outputs/analysis/token_window.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.attacks import ATTACK_CODES, make_attack  # noqa: E402
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.defenses import DEFENSE_CODES, apply_defense, clip_token_window  # noqa: E402
from guardrail_depth.paths import output_root  # noqa: E402

GRID = {
    "memorized_sd1": ["direct", "override", "truncate_repeat"],
    "copycat_characters": ["direct", "lure_desc60", "lure_kw_laion5", "override", "truncate_neutral"],
    "style_prompts": ["direct"],
}
TEMPLATES = ["reissinger_base", "task_instruction", "neg_inprompt", "cot"]


def main() -> None:
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
                                        revision="133a221b8aa7292a167afc5127cb63fb5005638b")
    rows = []
    for ds, attacks in GRID.items():
        items = load_manifest(ds).to_dict("records")
        for atk in attacks:
            for dfn in TEMPLATES:
                ws = []
                for it in items:
                    d = apply_defense(dfn, make_attack(atk, it["prompt"], it, tokenizer=tok)["prompt"], it)
                    ws.append(clip_token_window(tok, d.prompt, d.user_span))
                w = pd.DataFrame(ws)
                tail = w.suffix_tokens_total > 0
                frac = np.where(tail, w.suffix_tokens_seen / w.suffix_tokens_total.clip(lower=1), np.nan)
                rows.append({"dataset": ds, "attack": ATTACK_CODES[atk], "defense": DEFENSE_CODES[dfn], "n": len(w),
                             "median_tokens": float(w.n_tokens.median()),
                             "pct_truncated": round(100 * float((w.n_truncated > 0).mean()), 1),
                             "pct_tail_fully_lost": round(100 * float((tail & (w.suffix_tokens_seen == 0)).mean()), 1),
                             "mean_frac_tail_visible": round(float(np.nanmean(frac)), 3)})
    # LLM rewrites from the CopyCat release (D4) have no template tail; report their own length
    cc = load_manifest("copycat_characters")
    for col, atk in (("gpt4_rewrite_name", "direct"), ("gpt4_rewrite_desc60", "lure_desc60")):
        n = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in cc[col] if t]
        rows.append({"dataset": "copycat_characters", "attack": ATTACK_CODES[atk], "defense": "D4 (GPT-4 release)",
                     "n": len(n), "median_tokens": float(np.median(n)),
                     "pct_truncated": round(100 * float(np.mean([x > 75 for x in n])), 1),
                     "pct_tail_fully_lost": np.nan, "mean_frac_tail_visible": np.nan})
    out = output_root() / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "token_window.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
