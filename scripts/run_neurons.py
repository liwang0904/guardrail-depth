"""NeMo-style neuron probe (Hintersdorf et al., NeurIPS'24) under guardrails and attacks.

Memorized prompts drive a few cross-attention value neurons far outside their usual range.
We ask whether a guardrail silences those neurons, and whether an attack re-activates them.

Calibration matters: NeMo's reference statistics come from short natural LAION prompts, while a
guardrail template dominates the 77-token average and collapses activation variance, which hides
outliers whether or not memorization is still triggered (std of z drops from 1.0 to ~0.5 for any
long template, including the random-word control). By default (`--stats matched`) each
(attack, defense) condition is therefore scored against statistics of *non-memorized* prompts
passed through the same attack and defense (unmemorized test split), so z measures "unusual
for this template", not "unusual for raw LAION text".

    python scripts/run_neurons.py --model sd14 --split dev --attacks direct,truncate_repeat,override
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.attacks import ATTACK_CODES, make_attack  # noqa: E402
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.defenses import DEFENSE_CODES, apply_defense  # noqa: E402
from guardrail_depth.models import load_t2i, pick_device  # noqa: E402
from guardrail_depth.paths import output_root, raw_dir  # noqa: E402
from guardrail_depth.probes import (jaccard, load_nemo_stats, neuron_activations,  # noqa: E402
                                    neuron_outliers, neuron_reference_stats)


def zscores(acts, stats) -> np.ndarray:
    """(n_prompts, total_neurons) z-scores, layers concatenated."""
    return np.concatenate([(a - m) / s for a, (m, s) in zip(acts, stats)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sd14")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--theta", type=float, default=5.0, help="z-score threshold (NeMo initial theta)")
    ap.add_argument("--stats", default="matched", choices=["matched", "nemo"])
    ap.add_argument("--defenses", default="none,reissinger_base,task_instruction,neg_inprompt,cot,rand_tokens")
    ap.add_argument("--attacks", default="direct")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device")
    args = ap.parse_args()
    out = args.out or output_root() / "neurons" / f"{args.model}_{args.split}_{args.stats}"
    out.mkdir(parents=True, exist_ok=True)

    pipe, spec = load_t2i(args.model, pick_device(args.device))
    tok = pipe.tokenizer
    mem, ctl = load_manifest("memorized_sd1"), load_manifest("unmemorized_laion")
    ref_prompts = ctl[ctl.split == "test"]["prompt"].tolist()        # calibration set, disjoint from eval
    if args.split != "all":
        mem, ctl = mem[mem.split == args.split], ctl[ctl.split == args.split]
    items = ([dict(r, label=1) for r in mem.to_dict("records")]
             + [dict(r, label=0, mem_type="control") for r in ctl.to_dict("records")])
    nemo_stats = (load_nemo_stats(raw_dir() / "nemo/statistics/statistics_additional_laion_prompts_v1_4.pt")
                  if args.stats == "nemo" else None)

    def transform(attack, defense, prompts):
        return [apply_defense(defense, make_attack(attack, p, {}, tokenizer=tok)["prompt"], {}).prompt for p in prompts]

    base_sets, base_idx, rows = {}, {}, []
    for attack in args.attacks.split(","):
        for defense in args.defenses.split(","):
            stats = nemo_stats or neuron_reference_stats(
                neuron_activations(pipe, transform(attack, defense, ref_prompts)))
            acts = neuron_activations(pipe, transform(attack, defense, [it["prompt"] for it in items]))
            res = neuron_outliers(acts, stats, args.theta)
            z = zscores(acts, stats)
            offsets = np.cumsum([0] + [a.shape[1] for a in acts])
            for k, it in enumerate(items):
                key = (it["label"], it["id"])
                if attack == "direct" and defense == "none":
                    base_sets[key] = res["neurons"][k]
                    base_idx[key] = [offsets[l] + i for l, i in res["neurons"][k]]
                idx = base_idx.get(key, [])
                rows.append({"attack": ATTACK_CODES[attack], "defense": DEFENSE_CODES[defense], "id": it["id"],
                             "label": it["label"], "mem_type": it["mem_type"], "n_ood": int(res["n_ood"][k]),
                             "max_z": float(res["max_z"][k]),
                             # how strongly the neurons that fired for the undefended prompt still fire
                             "retained_z": float(z[k, idx].mean()) if idx else np.nan,
                             "jaccard_vs_D0": jaccard(res["neurons"][k], base_sets.get(key, set()))})
            print(f"  {ATTACK_CODES[attack]:12s} {DEFENSE_CODES[defense]:5s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "neurons.csv", index=False)
    summ = (df.groupby(["attack", "defense", "mem_type"], sort=False)
              .agg(n=("id", "size"), n_ood=("n_ood", "mean"), frac_any_ood=("n_ood", lambda x: float(np.mean(x > 0))),
                   max_z=("max_z", "mean"), retained_z=("retained_z", "mean"), jaccard_vs_D0=("jaccard_vs_D0", "mean"))
              .round(3).reset_index())
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
