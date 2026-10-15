"""He et al. (ICLR'25) DETECT / CONS with an open VLM judge on a CopyCat generation run.

    python scripts/run_judge.py outputs/track1_copycat/playground_copycat [--model judge_vlm]

DETECT(attack, defense) = number of distinct target characters the judge recognizes, per seed
(He et al. generate one image per character per run and report mean +- std over runs).
CONS = mean VQAScore-style P(yes) that the image shows the character's key entity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--model", default="judge_vlm", choices=["judge_vlm", "judge_vlm_small"])
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = registry()["auxiliary"][args.model]
    judge = VLMJudge(cfg["hf_id"], cfg["revision"], pick_device(args.device))
    df = load_records(args.run)
    entity = load_manifest("copycat_characters").set_index("id")["key_entity"]
    rows = []
    for r in tqdm(df.to_dict("records")):
        img = Image.open(args.run / r["image"])
        d = judge.detect(img)
        c = judge.p_yes(img, entity[r["item_id"]])
        rows.append({"image": r["image"], "judge_character": d["character"], "judge_score": d["score"],
                     "judge_raw": d["raw"], "detected": int(d["score"] == 1 and matches_target(d["character"], r["target"])),
                     "cons_p_yes": c["p_yes"], "cons_p_yes_norm": c["p_yes_norm"]})
    out = df.merge(pd.DataFrame(rows), on="image")
    out.to_csv(args.run / "judge.csv", index=False)
    per_seed = (out.groupby(["attack_code", "defense_code", "seed"], sort=False)
                   .agg(DETECT=("detected", "sum"), CONS=("cons_p_yes", "mean")).reset_index())
    summ = (per_seed.groupby(["attack_code", "defense_code"], sort=False)
                    .agg(DETECT_mean=("DETECT", "mean"), DETECT_std=("DETECT", "std"), CONS=("CONS", "mean"),
                         seeds=("seed", "nunique")).round(3).reset_index())
    summ.to_csv(args.run / "judge_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
