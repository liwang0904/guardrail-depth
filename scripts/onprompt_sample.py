"""Hand-labelled check of the on-prompt judge: sample images, label them in a browser, measure agreement.

    python scripts/onprompt_sample.py --n 120            # writes $GD_OUTPUT_ROOT/labeling/onprompt_label.html + key.csv
    # open the HTML (one self-contained file), answer each card, press "Download labels"
    python scripts/onprompt_sample.py --merge ~/Downloads/onprompt_labels.csv   # -> results/demo3/onprompt_labels.csv

The page shows the image and the text the judge was asked about, never the judge's answer. The sample is
stratified over runs and guardrails and balanced between images the judge called on- and off-prompt, so
both kinds of error are visible. The labels CSV (no images) is committed; demo3_report.py reports accuracy
and Cohen's kappa. Images stay out of git (they can show training images and characters).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.paths import output_root  # noqa: E402

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>On-prompt labels</title>
<style>body{font:14px system-ui;margin:16px;background:#fafaf8;color:#111}.card{display:flex;gap:16px;margin:10px 0;
padding:10px;background:#fff;border:1px solid #ddd;border-radius:8px}.card img{width:256px;height:256px;object-fit:cover}
.q{font-size:15px;margin-bottom:8px}button{margin-right:6px;padding:6px 14px}.yes.sel{background:#1baf7a;color:#fff}
.no.sel{background:#d03b3b;color:#fff}#bar{position:sticky;top:0;background:#fafaf8;padding:8px 0}</style></head><body>
<div id="bar"><b>Does the image show the text?</b> Judge the requested content only (a different-looking character
or a style change is fine). <span id="n"></span> <button onclick="dl()">Download labels</button></div><div id="c"></div>
<script>const D=__DATA__;const L={};const c=document.getElementById('c');
function upd(){document.getElementById('n').textContent=Object.keys(L).length+' / '+D.length+' labelled';}
D.forEach(d=>{const e=document.createElement('div');e.className='card';e.innerHTML=`<img src="data:image/jpeg;base64,${d.img}">
<div><div class="q">Does this image show <b>"${d.text}"</b>?</div><button class="yes">yes</button><button class="no">no</button>
<div style="color:#888;margin-top:8px">#${d.sid}</div></div>`;const [y,n]=e.querySelectorAll('button');
y.onclick=()=>{L[d.sid]=1;y.classList.add('sel');n.classList.remove('sel');upd()};
n.onclick=()=>{L[d.sid]=0;n.classList.add('sel');y.classList.remove('sel');upd()};c.appendChild(e);});upd();
function dl(){const rows=['sid,human_on_prompt'].concat(D.map(d=>d.sid+','+(d.sid in L?L[d.sid]:'')));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{type:'text/csv'}));
a.download='onprompt_labels.csv';a.click();}</script></body></html>"""


def sample(outputs: Path, n: int, seed: int = 347) -> pd.DataFrame:
    frames = []
    for stage in ("demo2", "demo3"):
        for run in sorted((outputs / stage).glob("*/")):
            f = run / "onprompt.csv"
            if not f.exists() or "_dev_" in run.name:
                continue
            o = pd.read_csv(f)
            o = o[o.get("refused", 0) == 0].dropna(subset=["p_yes_norm"])
            frames.append(o.assign(run=f"{stage}/{run.name}", judge_yes=(o["p_yes_norm"] > 0.5).astype(int)))
    df = pd.concat(frames, ignore_index=True)
    df["guardrail"] = df["image"].str.extract(r"_([A-Za-z0-9+-]+)_s\d+\.png$")[0]
    per = max(1, n // (df["run"].nunique() * 2))
    pick = (df.groupby(["run", "judge_yes"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per), random_state=seed)))
    pick = pick.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    pick["sid"] = range(len(pick))
    return pick


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path(os.environ.get("GD_OUTPUT_ROOT", ROOT / "outputs")))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--merge", type=Path, help="labels CSV downloaded from the page")
    args = ap.parse_args()
    lab_dir = args.outputs / "labeling"
    if args.merge:
        key = pd.read_csv(lab_dir / "key.csv")
        lab = pd.read_csv(args.merge)
        out = key.merge(lab, on="sid", how="left")
        dest = ROOT / "results" / "demo3" / "onprompt_labels.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out[["sid", "run", "guardrail", "image", "onprompt_text", "p_yes_norm", "human_on_prompt"]].to_csv(dest, index=False)
        done = out["human_on_prompt"].notna().sum()
        print(f"{done} of {len(out)} labelled -> {dest}")
        return
    pick = sample(args.outputs, args.n)
    data = []
    for r in pick.to_dict("records"):
        im = Image.open(args.outputs / r["run"] / r["image"]).convert("RGB").resize((256, 256))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        data.append({"sid": r["sid"], "text": str(r["onprompt_text"]).replace('"', "'"), "img": base64.b64encode(buf.getvalue()).decode()})
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "onprompt_label.html").write_text(HTML.replace("__DATA__", json.dumps(data)))
    pick[["sid", "run", "guardrail", "image", "onprompt_text", "p_yes_norm"]].to_csv(lab_dir / "key.csv", index=False)
    print(f"{len(pick)} images -> {lab_dir / 'onprompt_label.html'} (key: {lab_dir / 'key.csv'})")


if __name__ == "__main__":
    main()
