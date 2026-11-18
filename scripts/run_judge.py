"""He et al. (ICLR'25) DETECT / CONS with an open VLM judge on a CopyCat (or near-miss) generation run.

    python scripts/run_judge.py $GD_OUTPUT_ROOT/demo2/sd14_copycat [--model judge_vlm] [--batch-size 8]

DETECT(attack, defense) = number of distinct target characters the judge recognizes, per seed
(He et al. generate one image per character per run and report mean +- std over runs).
CONS = mean VQAScore-style P(yes) that the image shows the character's key entity.
On the benign near-miss set (generic_prompts) there is no target: `detected` = the judge names any of
the 50 protected characters (a character drawn unasked), and CONS is left to scripts/run_onprompt.py.
Re-running judges only images that are new.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.judge import VLMJudge, matches_target  # noqa: E402
from guardrail_depth.models import pick_device, registry  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import load_records  # noqa: E402


def detected_any(character: str, score: int, names: list[str]) -> str:
    """The protected character the judge named, if any ('' otherwise)."""
    if score != 1:
        return ""
    return next((n for n in names if matches_target(character, n)), "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--model", default="judge_vlm", choices=["judge_vlm", "judge_vlm_small"])
    ap.add_argument("--batch-size", type=int, default=1, help=">1 batches the CONS question; DETECT stays per image")
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = registry()["auxiliary"][args.model]
    df = load_records(args.run)
    ds = df["dataset"].iloc[0]
    names = load_manifest("copycat_characters")["target"].tolist()
    entity = load_manifest("copycat_characters").set_index("id")["key_entity"]
    out_f = args.run / "judge.csv"
    old = pd.read_csv(out_f) if out_f.exists() else pd.DataFrame(columns=["image"])
    todo = df[~df["image"].isin(old["image"])]
    print(f"{args.run.name}: {len(df)} records, {len(todo)} to judge")
    rows = []
    judge = VLMJudge(cfg["hf_id"], cfg["revision"], pick_device(args.device)) if len(todo) else None
    gen = []
    for r in tqdm(todo.to_dict("records")):
        if r.get("refused") is True:   # blocklist refusal: no image, nothing to recognize
            rows.append({"image": r["image"], "judge_character": "", "judge_score": 0, "judge_raw": "refused",
                         "detected": 0, "detected_protected": "", "cons_p_yes": float("nan"),
                         "cons_p_yes_norm": float("nan")})
            continue
        img = Image.open(args.run / r["image"])
        d = judge.detect(img)
        any_hit = detected_any(d["character"], d["score"], names)
        if ds == "copycat_characters":
            hit = int(d["score"] == 1 and matches_target(d["character"], r["target"]))
        else:
            hit = int(bool(any_hit))
        rows.append({"image": r["image"], "judge_character": d["character"], "judge_score": d["score"],
                     "judge_raw": d["raw"], "detected": hit, "detected_protected": any_hit})
        if ds == "copycat_characters":
            gen.append((len(rows) - 1, img, entity[int(r["item_id"])]))
    for i in range(0, len(gen), max(1, args.batch_size)):
        chunk = gen[i:i + max(1, args.batch_size)]
        res = judge.p_yes_batch([c[1] for c in chunk], [c[2] for c in chunk], batch_size=len(chunk))
        for (k, _, _), c in zip(chunk, res):
            rows[k].update(cons_p_yes=c["p_yes"], cons_p_yes_norm=c["p_yes_norm"])
    new = pd.DataFrame(rows)
    for c in ("judge_character", "judge_score", "judge_raw", "detected", "detected_protected", "cons_p_yes", "cons_p_yes_norm"):
        if c not in new:
            new[c] = np.nan          # keep the columns the reports read, even when no row filled them
    keep_cols = [c for c in old.columns if c in df.columns and c != "image"]
    judged = pd.concat([old.drop(columns=keep_cols, errors="ignore"), new], ignore_index=True)
    out = df.merge(judged, on="image", how="left")
    out.to_csv(out_f, index=False)
    if ds != "copycat_characters":
        s = out.groupby(["attack_code", "defense_code"], sort=False)["detected"].mean().round(3)
        print("share of images showing a protected character:\n" + s.to_string())
        return
    per_seed = (out.groupby(["attack_code", "defense_code", "seed"], sort=False)
                   .agg(DETECT=("detected", "sum"), CONS=("cons_p_yes", "mean")).reset_index())
    summ = (per_seed.groupby(["attack_code", "defense_code"], sort=False)
                    .agg(DETECT_mean=("DETECT", "mean"), DETECT_std=("DETECT", "std"), CONS=("CONS", "mean"),
                         seeds=("seed", "nunique")).round(3).reset_index())
    summ.to_csv(args.run / "judge_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
