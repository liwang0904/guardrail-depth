"""Side-by-side example panels from the Demo 2 runs, for presenting (not committed: they show
copyrighted characters and training images, which stay inside the team; see README section 7).

    python scripts/demo2_examples.py            # writes $GD_OUTPUT_ROOT/examples/*.png

Each panel is one request under several guardrails, labelled with the score the evaluation gave it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.paths import gt_image_dir, output_root  # noqa: E402

RUNS = output_root() / "demo2"

# (title, run, item_id, [(attack, defense, caption), ...], show training image)
PANELS = [
    ("rug_exact_copy", "sd14_memorized", 147,
     [("A0", "D0", "no guardrail"), ("A0", "D2b", "negative prompt"), ("A0", "D1", "task instruction"),
      ("A0", "C1", "random words"), ("A0", "D4", "LLM rewrite")], True),
    ("rug_padding", "sd14_memorized", 147,
     [("A3-trunc-rep", "D0", "padding, no guardrail"), ("A3-trunc-rep", "D1", "padding + task instruction")], True),
    ("carpet_strong_trigger", "sd14_memorized", 146,
     [("A0", "D0", "no guardrail"), ("A3-trunc-rep", "D0", "padding, no guardrail"),
      ("A3-override", "D0", "override, no guardrail"), ("A0", "D1", "task instruction")], True),
    ("pikachu", "sd14_copycat", 34,
     [("A0", "D0", "\"Pikachu\", none"), ("A0", "D1", "\"Pikachu\", task instr."),
      ("A1-kwLAION", "V0", "keywords, blocklist"), ("A1-kwLAION", "D2b-kw", "keywords, neg. prompt"),
      ("A0", "D4", "\"Pikachu\", LLM rewrite"), ("A1-desc", "D0", "60-word lure, none")], False),
    ("van_gogh_style", "sd14_style", "Vincent van Gogh|0",
     [("A0", "D0", "no guardrail"), ("A0", "D2b-name", "name as negative prompt")], False),
    ("karla_ortiz_style", "sd14_style", "Karla Ortiz|0",
     [("A0", "D0", "no guardrail"), ("A0", "D2b-name", "name as negative prompt")], False),
]


def load(run: str) -> pd.DataFrame:
    d = pd.read_csv(RUNS / run / "metrics.csv", low_memory=False)
    if (RUNS / run / "judge.csv").exists():
        j = pd.read_csv(RUNS / run / "judge.csv", usecols=["image", "detected", "judge_character"])
        d = d.merge(j, on="image", how="left")
    d["item_id"] = d["item_id"].astype(str)
    return d


def score_label(r: pd.Series) -> str:
    if r.get("refused") in (1, 1.0, True, "True"):
        return "refused"
    if pd.notna(r.get("sscd_gt")):
        return f"SSCD {r['sscd_gt']:.2f}" + (" COPY" if r["sscd_gt"] > 0.5 else "")
    if "judge_character" in r and isinstance(r.get("judge_character"), str):
        return f"judge: {r['judge_character']}"
    if "detected" in r and pd.notna(r.get("detected")):
        return "judge: target" if r["detected"] == 1 else "judge: not target"
    if "artist_top1" in r:
        return f"CLIP: {r['artist_top1']}"
    return ""


def panel(title, run, item, cells, with_gt, out: Path) -> Path | None:
    import matplotlib.pyplot as plt

    df = load(run)
    rows = df[(df["item_id"] == str(item)) & (df["seed"] == 0)]
    tiles = []
    if with_gt:
        gt = gt_image_dir() / ("sd1" if run.startswith("sd14") else "sd2") / f"{int(item):04d}.jpg"
        if gt.exists():
            tiles.append((Image.open(gt).convert("RGB"), "TRAINING IMAGE", "the original"))
    for attack, defense, caption in cells:
        r = rows[(rows["attack_code"] == attack) & (rows["defense_code"] == defense)]
        if r.empty:
            print(f"  {title}: no record for {attack}/{defense}")
            continue
        r = r.iloc[0]
        path = RUNS / run / r["image"]
        img = Image.open(path).convert("RGB") if path.suffix == ".png" and path.exists() else Image.new("RGB", (512, 512), "white")
        tiles.append((img, caption, score_label(r)))
    if not tiles:
        return None
    fig, axes = plt.subplots(1, len(tiles), figsize=(2.6 * len(tiles), 3.4), facecolor="white")
    axes = [axes] if len(tiles) == 1 else axes
    for ax, (img, cap, lab) in zip(axes, tiles):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{cap}\n{lab}", fontsize=9)
    prompt = rows["user_prompt"].iloc[0] if len(rows) else ""
    fig.suptitle(f"{title}  |  request: {str(prompt)[:90]}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = out / f"{title}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main() -> None:
    out = output_root() / "examples"
    out.mkdir(parents=True, exist_ok=True)
    for spec in PANELS:
        p = panel(*spec, out=out)
        if p:
            print("wrote", p)
    print(f"\nview them: OnDemand -> Files -> {out}  (or scp -r the folder to your laptop)")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
