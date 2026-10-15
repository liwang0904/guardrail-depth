"""Wen et al. (ICLR'24) memorization detector, applied to what each guardrail actually feeds the model.

For every (attack, defense) the first-step text-conditional noise magnitude is computed for the
memorized prompts and the non-memorized LAION controls; AUC / TPR@1%FPR say whether the
memorization trigger survives the guardrail. A prompt-level defense that "works" should push the
memorized scores down to the control level; if the scores stay high, the copy is still being
triggered and the guardrail only perturbs the surface text.

    python scripts/run_detection.py --model sd14 --split dev --n-images 4 \
        --defenses none,reissinger_base,task_instruction,neg_inprompt,negbuf_rules,cot,rand_tokens \
        --attacks direct,truncate_repeat
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.attacks import ATTACK_CODES, make_attack  # noqa: E402
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.defenses import DEFENSE_CODES, apply_defense  # noqa: E402
from guardrail_depth.models import load_t2i, pick_device  # noqa: E402
from guardrail_depth.paths import output_root  # noqa: E402
from guardrail_depth.probes import wen_detection_scores  # noqa: E402


def tpr_at_fpr(y, s, fpr_max=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sd14")
    ap.add_argument("--memorized", default="memorized_sd1")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--defenses", default="none,reissinger_base,task_instruction,neg_inprompt,negbuf_rules,cot,rand_tokens")
    ap.add_argument("--attacks", default="direct")
    ap.add_argument("--batch-size", type=int, help="prompts per UNet call (default: 4 on CUDA, 1 on MPS/CPU)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device")
    args = ap.parse_args()
    out = args.out or output_root() / "detection" / f"{args.model}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    pipe, spec = load_t2i(args.model, device)
    # MPS has no memory-efficient attention kernel: 64x64 self-attention over many latents needs GBs
    bs = args.batch_size or (4 if device.type == "cuda" else 1)
    mem = load_manifest(args.memorized)
    ctl = load_manifest("unmemorized_laion")
    if args.split != "all":
        mem, ctl = mem[mem.split == args.split], ctl[ctl.split == args.split]
    items = ([dict(r, label=1, source=args.memorized) for r in mem.to_dict("records")]
             + [dict(r, label=0, mem_type="control", source="unmemorized_laion") for r in ctl.to_dict("records")])
    print(f"{len(mem)} memorized + {len(ctl)} control prompts, {args.n_images} noise seeds, model {spec.key}")

    rows = []
    for attack in args.attacks.split(","):
        for defense in args.defenses.split(","):
            t0 = time.time()
            adv = [make_attack(attack, it["prompt"], {}, tokenizer=pipe.tokenizer)["prompt"] for it in items]
            dd = [apply_defense(defense, a, {}) for a in adv]
            prompts = [d.prompt for d in dd]
            scores = wen_detection_scores(pipe, prompts, n_images=args.n_images, num_steps=spec.steps, batch_size=bs)
            negs = [d.negative_prompt for d in dd]
            guide = (wen_detection_scores(pipe, prompts, n_images=args.n_images, num_steps=spec.steps,
                                          negative_prompts=negs, batch_size=bs) if any(negs) else scores)
            for it, s, g in zip(items, scores, guide):
                rows.append({"attack": ATTACK_CODES[attack], "defense": DEFENSE_CODES[defense], "id": it["id"],
                             "source": it["source"], "label": it["label"], "mem_type": it["mem_type"],
                             "score": float(s.mean()), "score_1img": float(s[0]), "guidance_norm": float(g.mean())})
            print(f"  {ATTACK_CODES[attack]:12s} {DEFENSE_CODES[defense]:8s} {time.time() - t0:5.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)

    summary = []
    for (atk, dfn), g in df.groupby(["attack", "defense"], sort=False):
        base = df[(df.attack == "A0") & (df.defense == "D0")].set_index(["source", "id"])["score"]
        rel = g.set_index(["source", "id"])["score"] / base.reindex(g.set_index(["source", "id"]).index).values
        summary.append({
            "attack": atk, "defense": dfn,
            "auc_n": roc_auc_score(g.label, g.score), "auc_1img": roc_auc_score(g.label, g.score_1img),
            "tpr@1%fpr": tpr_at_fpr(g.label, g.score),
            "score_VM": g[g.mem_type == "VM"].score.mean(), "score_TM": g[g.mem_type == "TM"].score.mean(),
            "score_ctrl": g[g.label == 0].score.mean(),
            "rel_to_D0_VM": float(np.nanmean(rel[g.set_index(["source", "id"]).mem_type.values == "VM"])),
            "rel_to_D0_ctrl": float(np.nanmean(rel[g.set_index(["source", "id"]).label.values == 0])),
        })
    summ = pd.DataFrame(summary).round(3)
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
