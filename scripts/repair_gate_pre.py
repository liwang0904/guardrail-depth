"""Repair the G-pre rows written before the output-check fix in generation._GateRunner.run.

Before the fix, a job that ran both `gated` (G) and `gated_pre` (G-pre, the ablation without the output
check) output-checked the G-pre rows too, so every G-pre image was G's final image. The correct G-pre image
is the first render of each request, which each record already names in `pre_image` (G's own first render,
or the baseline render the gate reused). This script restores it without rendering anything:

* G-pre rows the check had re-rendered (n_generations == 2): image <- the first render, and its prompt,
  negative prompt and probe summaries <- that render's record (for G's own first renders, which have no
  record of their own: the negative prompt minus the part the output check appended, probes unknown);
* every G-pre row: final plan <- first plan, n_generations <- 1, output-check fields removed;
* judge.csv / onprompt.csv: rows of the changed images are dropped, so run_judge.py and run_onprompt.py
  score exactly those images again (evaluate.py rebuilds metrics.csv from scratch).

Originals are copied to <run>/backup_gate_pre/ first.

    python scripts/repair_gate_pre.py $GD_OUTPUT_ROOT/demo3/*/ [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth import gate as G  # noqa: E402
from guardrail_depth.generation import _PROBE_KEYS  # noqa: E402
from guardrail_depth.paths import output_root  # noqa: E402

CHECK_PREFIXES = ("check_", "pre_check_")


def source_index(run: Path, root: Path) -> dict[str, dict]:
    """'<dir relative to the output root>/<image>' -> record, for this run and the dirs it reused."""
    dirs = [run]
    jf = run / "job.json"
    if jf.exists():
        dirs += [root / r for r in json.loads(jf.read_text()).get("job", {}).get("reuse", []) or []]
    out = {}
    for d in dirs:
        if not d.exists():
            continue
        rel = d.resolve().relative_to(root.resolve())
        for f in sorted(d.glob("records*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r.get("image") and not r.get("refused"):
                    out[f"{rel}/{r['image']}"] = r
    return out


def first_negative(rec: dict) -> str | None:
    """Negative prompt of the first plan when the output check appended the seen character's to it."""
    final, target = rec.get("negative_prompt") or "", str(rec.get("gate_final_target") or "")
    if "+" not in target:
        return None
    extra = G.he_negative_prompt("name_emb5_laion5", G.concept_meta(target.rsplit("+", 1)[1]))
    if final.endswith(", " + extra):
        return final[: -len(", " + extra)]
    return "" if final == extra else None


def repair_run(run: Path, root: Path, dry: bool) -> tuple[int, int, set[str]]:
    shards = sorted(run.glob("records*.jsonl"))
    index = source_index(run, root)
    n_pre, problems, changed = 0, 0, set()
    new_lines: dict[Path, list[str]] = {}
    copies: list[tuple[Path, Path]] = []
    for f in shards:
        lines = []
        for line in open(f):
            rec = json.loads(line)
            if rec.get("defense") != "gated_pre":
                lines.append(line.rstrip("\n"))
                continue
            n_pre += 1
            if int(rec.get("n_generations") or 1) == 2:
                pre = rec.get("pre_image", "")
                src = root / pre if (root / pre).exists() else run / pre
                if not pre or not src.exists():
                    print(f"  ! {run.name}: first render not found for {rec['image']} ({pre!r})")
                    problems += 1
                    lines.append(line.rstrip("\n"))
                    continue
                copies.append((src, run / rec["image"]))
                srec = index.get(pre)
                for k in [k for k in rec if _PROBE_KEYS.match(k)]:
                    rec[k] = None
                if srec is not None:
                    rec["model_prompt"], rec["negative_prompt"] = srec["model_prompt"], srec.get("negative_prompt", "")
                    rec.update({k: v for k, v in srec.items() if _PROBE_KEYS.match(k)})
                    rec["sec_per_image"] = srec.get("sec_per_image", rec.get("sec_per_image"))
                else:   # G's own first render: same prompt, negative without the appended part
                    neg = first_negative(rec)
                    if neg is None:
                        print(f"  ! {run.name}: cannot recover the first negative prompt of {rec['image']}")
                        problems += 1
                    else:
                        rec["negative_prompt"] = neg
                rec.update(reused_from=pre, rendered=False)
                changed.add(rec["image"])
            for k in [k for k in rec if k.startswith(CHECK_PREFIXES) or k == "pre_image"]:
                del rec[k]
            rec.update(gate_final_applied=rec.get("gate_applied"), gate_final_target=rec.get("gate_target"),
                       gate_final_reason=rec.get("gate_reason"), n_generations=1, gate_pre_repaired=True)
            lines.append(json.dumps(rec))
        new_lines[f] = lines
    print(f"{run.name}: {n_pre} G-pre rows, {len(changed)} images back to the first render"
          + (f", {problems} problems" if problems else ""))
    if dry or n_pre == 0:
        return n_pre, problems, changed
    bak = run / "backup_gate_pre"
    bak.mkdir(exist_ok=True)
    for f in shards + [run / "judge.csv", run / "onprompt.csv", run / "metrics.csv", run / "summary.csv"]:
        if f.exists() and not (bak / f.name).exists():
            shutil.copy2(f, bak / f.name)
    for src, dst in copies:
        if not (bak / "images" / dst.name).exists():
            (bak / "images").mkdir(exist_ok=True)
            shutil.copy2(dst, bak / "images" / dst.name)
        shutil.copyfile(src, dst)
    for f, lines in new_lines.items():
        tmp = f.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(f)
    for name in ("judge.csv", "onprompt.csv"):
        p = run / name
        if p.exists() and changed:
            df = pd.read_csv(p, low_memory=False)
            df[~df["image"].isin(changed)].to_csv(p, index=False)
    return n_pre, problems, changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = output_root()
    total = [0, 0, 0]
    for run in args.runs:
        n, p, c = repair_run(run, root, args.dry_run)
        total = [total[0] + n, total[1] + p, total[2] + len(c)]
    print(f"total: {total[0]} G-pre rows, {total[2]} images restored, {total[1]} problems"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    if total[1]:
        sys.exit(1)


if __name__ == "__main__":
    main()
