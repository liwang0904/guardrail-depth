"""Demo 2 tables and figures: every baseline on every dataset and model, side by side.

    python scripts/demo2_report.py [--outputs $GD_OUTPUT_ROOT] [--results results/demo2]

Reads the scored runs (`evaluate.py` -> metrics.csv, `run_judge.py` -> judge.csv), the detector
summaries and the rewriter CSVs, and writes small committed files to results/demo2/:

  main_<run>.csv / .md        protection metric, guardrail x attack
  comparison.csv / .md        one row per (run, guardrail): protection under the plain request and the
                              worst attack, usefulness, refusal rate, cost on benign prompts
  subgroups.csv / .md         protection by subgroup (memorization type, duplicate count, character
                              origin and rights holder, artist group) + the gap between groups
  errors.md                   items the best prompt-level guardrail still fails on
  detection.csv / .md         Wen et al. detector: AUC, TPR / precision / F1 at 1% FPR
  rewriter.csv / .md          D4 leakage: name kept in the rewrite, character recognized in reasoning
  headline.json               the numbers quoted in the demo
  fig_*.png                   heatmaps, protection vs. usefulness, artist-group bars
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAMES = {"D0": "none", "V0": "blocklist (vanilla)", "D1": "task instruction", "D2a": "negation in prompt",
         "D2b": "negative prompt", "D2b-kw": "negative prompt (name+kw)", "D2b-name": "negative prompt (name)",
         "D3": "chain-of-thought", "D4": "LLM rewrite", "D4+D2b": "LLM rewrite + neg. prompt", "C1": "random words"}
ATTACKS = {"A0": "plain request", "A1-desc": "60-word lure", "A1-kwLAION": "keyword lure", "A2-phon": "misspelling",
           "A3-trunc-rep": "padding", "A3-override": "override"}
ORDER = list(NAMES)
# which number means "it copied" for each dataset, and which means "still useful"
PROTECT = {"memorized_sd1": ("copy", "copy rate"), "memorized_sd2": ("copy", "copy rate"),
           "copycat_characters": ("detected", "character detected"), "style_prompts": ("attributed", "style attributed")}
USEFUL = {"memorized_sd1": "clip_t", "memorized_sd2": "clip_t", "unmemorized_laion": "clip_t",
          "copycat_characters": "clip_cons", "style_prompts": "clip_t"}
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def load_run(run: Path) -> pd.DataFrame | None:
    f = run / "metrics.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, low_memory=False)
    if (run / "judge.csv").exists():
        j = pd.read_csv(run / "judge.csv", usecols=["image", "detected", "cons_p_yes"])
        df = df.merge(j, on="image", how="left")
    df["run"] = run.name
    df["refused"] = pd.to_numeric(df.get("refused", 0), errors="coerce").fillna(0)
    ds = df["dataset"].iloc[0]
    if ds == "copycat_characters" and "detected" not in df:   # judge not run yet: CLIP zero-shot proxy
        df["detected"] = df["clip_detect"]
    if ds == "style_prompts":
        df["attributed"] = np.where(df["refused"] == 1, 0.0, (df.get("artist_top1") == df["target"]).astype(float))
    if ds == "copycat_characters" and "cons_p_yes" in df:
        df["clip_cons"] = df["cons_p_yes"].fillna(df.get("clip_cons"))
    for c in ("copy", "detected", "attributed", "clip_t", "clip_cons"):
        if c in df:
            df[c] = pd.to_numeric(df[c].map(lambda v: {"True": 1, "False": 0}.get(v, v)).astype("object"),
                                  errors="coerce")
    return df


def ordered(idx) -> list:
    return sorted(idx, key=lambda c: ORDER.index(c) if c in ORDER else 99)


def pct(x) -> str:
    return "" if pd.isna(x) else f"{100 * x:.0f}%"


def md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main_tables(runs: dict, out: Path) -> dict:
    texts = {}
    for name, df in runs.items():
        ds = df["dataset"].iloc[0]
        if ds not in PROTECT:
            continue
        col, label = PROTECT[ds]
        t = df.pivot_table(index="defense_code", columns="attack_code", values=col, aggfunc="mean")
        t = t.loc[ordered(t.index), [a for a in ATTACKS if a in t.columns]]
        t.to_csv(out / f"main_{name}.csv")
        show = t.apply(lambda s: s.map(pct)).rename(columns=ATTACKS)
        show.insert(0, "guardrail", [f"{c} {NAMES.get(c, '')}" for c in show.index])
        texts[name] = f"### {name}: {label} (lower = better protection)\n\n" + md(show)
        (out / f"main_{name}.md").write_text(texts[name] + "\n")
    return texts


def _benign(code: str) -> str:
    return "D2b" if code.startswith("D2b") else code


def comparison(runs: dict) -> pd.DataFrame:
    ctrl = runs.get("sd14_controls")
    ctrl_use = ctrl.groupby("defense_code")["clip_t"].mean() if ctrl is not None else pd.Series(dtype=float)
    ctrl_ref = ctrl.groupby("defense_code")["refused"].mean() if ctrl is not None else pd.Series(dtype=float)
    rows = []
    for name, df in runs.items():
        ds = df["dataset"].iloc[0]
        if ds not in PROTECT:
            continue
        col = PROTECT[ds][0]
        use = USEFUL[ds]
        by = df.groupby(["defense_code", "attack_code"])[col].mean().unstack()
        for d in ordered(by.index):
            g = df[df.defense_code == d]
            gen = g[g.refused == 0]
            rows.append({
                "run": name, "model": g["model"].iloc[0], "dataset": ds, "guardrail": d, "name": NAMES.get(d, d),
                "protect_metric": PROTECT[ds][1],
                "plain_request": by.loc[d].get("A0", np.nan), "worst_attack": by.loc[d].max(),
                "worst_attack_name": ATTACKS.get(by.loc[d].idxmax(), "") if by.loc[d].notna().any() else "",
                "mean_over_attacks": by.loc[d].mean(),
                "usefulness": gen[use].mean() if use in gen else np.nan,
                "refusal_rate": g["refused"].mean(),
                # benign captions have no named target, so every D2b variant runs as the D2b rule list there
                "benign_usefulness": ctrl_use.get(_benign(d), np.nan), "benign_refusal": ctrl_ref.get(_benign(d), np.nan),
                "n_images": int((g.refused == 0).sum()),
            })
    c = pd.DataFrame(rows)
    if len(c) and ctrl_use.size and "D0" in ctrl_use:
        c["benign_usefulness_cost"] = ctrl_use["D0"] - c["benign_usefulness"]
    return c


def subgroups(runs: dict) -> pd.DataFrame:
    rows = []

    def add(name, df, col, by, label):
        d = df[df.attack_code == "A0"] if label != "attack-avg" else df
        t = d.groupby(["defense_code", by], observed=True)[col].agg(["mean", "count"]).reset_index()
        for r in t.itertuples():
            rows.append({"run": name, "subgroup_by": by, "subgroup": r[2], "guardrail": r.defense_code,
                         "rate": r.mean, "n": r.count})

    for name, df in runs.items():
        ds = df["dataset"].iloc[0]
        if ds.startswith("memorized"):
            if "mem_type" in df:
                add(name, df, "copy", "mem_type", "A0")
            if "numdups" in df and df["numdups"].notna().any():
                df = df.copy()
                df["duplicates"] = pd.qcut(df["numdups"].rank(method="first"), 3, labels=["low", "mid", "high"])
                add(name, df, "copy", "duplicates", "A0")
        elif ds == "copycat_characters":
            add(name, df, "detected", "origin_region", "attack-avg")
            top = df["rights_holder"].value_counts().index[:4]
            add(name, df[df.rights_holder.isin(top)], "detected", "rights_holder", "attack-avg")
        elif ds == "style_prompts":
            add(name, df, "attributed", "group", "A0")
    return pd.DataFrame(rows)


def group_gaps(sub: pd.DataFrame) -> pd.DataFrame:
    """Largest minus smallest subgroup rate for each guardrail: how unequal the protection is."""
    if sub.empty:
        return sub
    g = sub[sub.n >= 5].groupby(["run", "subgroup_by", "guardrail"])["rate"]
    out = pd.DataFrame({"max_rate": g.max(), "min_rate": g.min(),
                        "most_exposed": sub.loc[g.idxmax()].set_index(["run", "subgroup_by", "guardrail"])["subgroup"],
                        "least_exposed": sub.loc[g.idxmin()].set_index(["run", "subgroup_by", "guardrail"])["subgroup"]})
    out["gap"] = out["max_rate"] - out["min_rate"]
    return out.reset_index()


def best_prompt_guardrail(comp: pd.DataFrame, run: str) -> str | None:
    c = comp[(comp.run == run) & ~comp.guardrail.isin(["D0", "V0", "C1"])]
    return None if c.empty else c.sort_values(["mean_over_attacks", "refusal_rate"]).iloc[0]["guardrail"]


def errors(runs: dict, comp: pd.DataFrame) -> str:
    parts = []
    for name, df in runs.items():
        ds = df["dataset"].iloc[0]
        if ds not in PROTECT:
            continue
        best = best_prompt_guardrail(comp, name)
        if best is None:
            continue
        col = PROTECT[ds][0]
        g = df[(df.defense_code == best) & (df[col] == 1)]
        if g.empty:
            parts.append(f"### {name}: best prompt-level guardrail {best} ({NAMES.get(best)}) has no failures\n")
            continue
        keys = ["item_id", "attack_code"]
        extra = [c for c in ("mem_type", "numdups", "origin_region", "rights_holder", "group", "target") if c in g]
        agg = g.groupby(keys).agg(failures=(col, "size"), **{c: (c, "first") for c in extra},
                                  prompt=("adv_prompt", "first")).reset_index()
        if "sscd_gt" in g:
            agg["max_sscd"] = g.groupby(keys)["sscd_gt"].max().values
        agg["prompt"] = agg["prompt"].astype(str).str.slice(0, 70)
        agg = agg.sort_values("failures", ascending=False).head(15)
        n_items = df[df.defense_code == best]["item_id"].nunique()
        parts.append(f"### {name}: {g['item_id'].nunique()} of {n_items} items still fail under {best} "
                     f"({NAMES.get(best)})\n\n" + md(agg) + "\n")
    return "\n".join(parts)


def detection(outputs: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((outputs / "detection").glob("*_all/summary.csv")):
        d = pd.read_csv(f)
        d.insert(0, "model", f.parent.name.replace("_all", ""))
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rewriter(outputs: Path) -> pd.DataFrame:
    f = outputs / "rewrites" / "demo2_copycat_characters.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    cols = {"rewrite_name_leak": "name kept in rewrite", "reasoning_names_target": "character named in reasoning",
            "entropy_final_mean": "mean token entropy"}
    t = d.groupby("attack")[[c for c in cols if c in d]].mean().rename(columns=cols)
    if "rewrite_kw_hits" in d and "kw_total" in d:
        t["keyword share kept"] = (d["rewrite_kw_hits"] / d["kw_total"].replace(0, np.nan)).groupby(d["attack"]).mean()
    return t.reset_index()


def bootstrap_ci(runs: dict, n_boot: int = 2000, seed: int = 347) -> pd.DataFrame:
    """95% intervals for the protection metric, resampling items (prompts / characters / style prompts),
    the unit that was sampled; seeds and attacks of one item move together."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, df in runs.items():
        ds = df["dataset"].iloc[0]
        if ds not in PROTECT:
            continue
        col = PROTECT[ds][0]
        d = df.dropna(subset=[col])
        for (g, a), x in d.groupby(["defense_code", "attack_code"]):
            t = x.groupby("item_id")[col].agg(["sum", "count"])
            s, c = t["sum"].to_numpy(float), t["count"].to_numpy(float)
            idx = rng.integers(0, len(t), (n_boot, len(t)))
            b = s[idx].sum(1) / c[idx].sum(1)
            rows.append({"run": name, "guardrail": g, "attack": a, "rate": s.sum() / c.sum(),
                         "ci_low": np.quantile(b, 0.025), "ci_high": np.quantile(b, 0.975),
                         "n_items": len(t), "n_images": int(c.sum())})
    out = pd.DataFrame(rows)
    out["_g"] = out["guardrail"].map(lambda c: ORDER.index(c) if c in ORDER else 99)
    out["_a"] = out["attack"].map(lambda a: list(ATTACKS).index(a) if a in ATTACKS else 99)
    return out.sort_values(["run", "_a", "_g"]).drop(columns=["_g", "_a"]).reset_index(drop=True)


def coverage(runs: dict) -> pd.DataFrame:
    """How many sampled exact-copy prompts still have their training image (SSCD needs it)."""
    rows = []
    for name, df in runs.items():
        if not df["dataset"].iloc[0].startswith("memorized"):
            continue
        has = df.dropna(subset=["copy"])["item_id"].unique()
        for t, g in df.groupby("mem_type"):
            rows.append({"run": name, "mem_type": t, "items": g["item_id"].nunique(),
                         "with_training_image": int(np.isin(g["item_id"].unique(), has).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def fig_heatmaps(runs: dict, out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    made = []
    for name, df in runs.items():
        ds = df["dataset"].iloc[0]
        if ds not in PROTECT:
            continue
        col, label = PROTECT[ds]
        t = df.pivot_table(index="defense_code", columns="attack_code", values=col, aggfunc="mean")
        t = t.loc[ordered(t.index), [a for a in ATTACKS if a in t.columns]]
        fig, ax = plt.subplots(figsize=(max(6.4, 2.6 + 1.25 * t.shape[1]), 1.1 + 0.38 * t.shape[0]), facecolor=SURFACE)
        ax.imshow(t.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if v > 0.55 else INK)
        ax.set_xticks(range(t.shape[1]), [ATTACKS[a] for a in t.columns], rotation=20, ha="right")
        ax.set_yticks(range(t.shape[0]), [f"{c}  {NAMES.get(c, '')}" for c in t.index])
        ax.tick_params(length=0, colors=INK2, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.suptitle(f"{name}: {label} (%), lower = better", x=0.02, ha="left", fontsize=11, color=INK)
        p = out / f"fig_heatmap_{name}.png"
        fig.savefig(p, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        made.append(p.name)
    return made


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart so no two are closer than `gap` (keeps their order)."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return list(out)


def fig_tradeoff(comp: pd.DataFrame, out: Path) -> str | None:
    """Protection under the worst attack vs. usefulness on benign prompts, SD 1.4 exact copies."""
    import matplotlib.pyplot as plt

    c = comp[(comp.run == "sd14_memorized")].dropna(subset=["worst_attack", "benign_usefulness"]).copy()
    if c.empty:
        return None
    c["label"] = c["guardrail"] + " " + c["name"]
    c = (c.groupby(["benign_usefulness", "worst_attack"], sort=False)
           .agg(label=("label", " = ".join), guardrail=("guardrail", "first")).reset_index())
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.scatter(c["benign_usefulness"], 100 * c["worst_attack"], s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ys = _spread(list(100 * c["worst_attack"]), gap=0.045 * max(1.0, 100 * c["worst_attack"].max()))
    dx = 0.02 * max(1e-3, c["benign_usefulness"].max() - c["benign_usefulness"].min())
    for r, y in zip(c.itertuples(), ys):
        ax.annotate(r.label, xy=(r.benign_usefulness, 100 * r.worst_attack),
                    xytext=(r.benign_usefulness + dx, y), textcoords="data", fontsize=8.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6, shrinkA=0, shrinkB=4))
    ax.margins(x=0.25, y=0.1)
    ax.set_xlabel("usefulness on benign prompts (CLIP adherence, higher = better)", color=INK2, fontsize=9)
    ax.set_ylabel("copy rate under the worst attack (%)", color=INK2, fontsize=9)
    ax.set_title("SD 1.4 exact copies: protection vs. usefulness, one dot per guardrail", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    p = out / "fig_tradeoff_sd14_memorized.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p.name


def fig_artist_groups(sub: pd.DataFrame, out: Path) -> str | None:
    import matplotlib.pyplot as plt

    s = sub[(sub.run == "sd14_style") & (sub.subgroup_by == "group")]
    if s.empty:
        return None
    groups = [g for g in ("living", "deceased_copyright", "public_domain") if g in set(s.subgroup)]
    labels = {"living": "living", "deceased_copyright": "deceased, in copyright", "public_domain": "public domain"}
    guards = ordered(s.guardrail.unique())
    x = np.arange(len(guards))
    w = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(2 + 1.15 * len(guards), 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for k, g in enumerate(groups):
        v = [100 * s[(s.guardrail == d) & (s.subgroup == g)]["rate"].mean() for d in guards]
        bars = ax.bar(x + (k - (len(groups) - 1) / 2) * w, v, w - 0.03, color=SERIES[k], label=labels[g])
        for b, val in zip(bars, v):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{val:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f"{d}\n" + textwrap.fill(NAMES.get(d, ""), 13) for d in guards], fontsize=8)
    ax.set_ylabel("prompts attributed to the requested artist (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Style protection by artist group (SD 1.4, plain request)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    p = out / "fig_artist_groups.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p.name


# ----------------------------------------------------------------------------- headline + main
def headline(comp: pd.DataFrame, gaps: pd.DataFrame, det: pd.DataFrame, rw: pd.DataFrame) -> dict:
    h = {}
    for run in comp["run"].unique():
        c = comp[comp.run == run].set_index("guardrail")
        best = best_prompt_guardrail(comp, run)
        h[run] = {
            "no_guardrail_plain": c.loc["D0", "plain_request"] if "D0" in c.index else None,
            "best_prompt_guardrail": best,
            "best_plain": c.loc[best, "plain_request"] if best else None,
            "best_worst_attack": c.loc[best, "worst_attack"] if best else None,
            "best_worst_attack_name": c.loc[best, "worst_attack_name"] if best else None,
            "vanilla_plain": c.loc["V0", "plain_request"] if "V0" in c.index else None,
            "vanilla_worst_attack": c.loc["V0", "worst_attack"] if "V0" in c.index else None,
            "vanilla_refusal": c.loc["V0", "refusal_rate"] if "V0" in c.index else None,
            "random_words_plain": c.loc["C1", "plain_request"] if "C1" in c.index else None,
        }
    if not gaps.empty:
        h["largest_subgroup_gaps"] = gaps.sort_values("gap", ascending=False).head(5).round(3).to_dict("records")
    if not det.empty:
        h["detector"] = det[(det.attack == "A0") & (det.defense == "D0")].round(3).to_dict("records")
    if not rw.empty:
        h["rewriter"] = rw.round(3).to_dict("records")
    return json.loads(pd.Series(h).to_json())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path(os.environ.get("GD_OUTPUT_ROOT", ROOT / "outputs")))
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "demo2")
    args = ap.parse_args()
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    runs = {p.name: df for p in sorted((args.outputs / "demo2").glob("*/")) if (df := load_run(p)) is not None}
    if not runs:
        raise SystemExit(f"no scored runs under {args.outputs / 'demo2'} (run scripts/evaluate.py first)")

    mains = main_tables(runs, out)
    comp = comparison(runs)
    comp.round(4).to_csv(out / "comparison.csv", index=False)
    sub = subgroups(runs)
    sub.round(4).to_csv(out / "subgroups.csv", index=False)
    gaps = group_gaps(sub)
    gaps.round(4).to_csv(out / "subgroup_gaps.csv", index=False)
    det = detection(args.outputs)
    if not det.empty:
        det.round(4).to_csv(out / "detection.csv", index=False)
    rw = rewriter(args.outputs)
    if not rw.empty:
        rw.round(4).to_csv(out / "rewriter.csv", index=False)
    ci = bootstrap_ci(runs)
    ci.round(4).to_csv(out / "ci.csv", index=False)
    cov = coverage(runs)
    if not cov.empty:
        cov.to_csv(out / "coverage.csv", index=False)
    (out / "errors.md").write_text("# Error analysis: what the best prompt-level guardrail still lets through\n\n"
                                   + errors(runs, comp))
    figs = fig_heatmaps(runs, out) + [f for f in (fig_tradeoff(comp, out), fig_artist_groups(sub, out)) if f]
    (out / "headline.json").write_text(json.dumps(headline(comp, gaps, det, rw), indent=1))

    view = comp.copy()
    for c in ("plain_request", "worst_attack", "mean_over_attacks", "refusal_rate", "benign_refusal"):
        view[c] = view[c].map(pct)
    view = view[["run", "guardrail", "name", "plain_request", "worst_attack", "worst_attack_name", "refusal_rate",
                 "usefulness", "benign_usefulness", "benign_refusal"]]
    report = ["# Demo 2 results", "",
              f"{sum(len(d) for d in runs.values())} scored images/refusals across {len(runs)} runs "
              f"({', '.join(runs)}). Lower protection numbers are better; usefulness is CLIP adherence.", "",
              "## All baselines side by side", "", md(view), "", "## Per run: guardrail x attack", ""]
    report += [t + "\n" for t in mains.values()]
    if not gaps.empty:
        g = gaps.copy()
        for c in ("max_rate", "min_rate", "gap"):
            g[c] = g[c].map(pct)
        report += ["## Subgroups: how unequal is the protection?", "", md(g), ""]
    if not det.empty:
        report += ["## Memorization detector (Wen et al.)", "", md(det), ""]
    if not rw.empty:
        report += ["## LLM rewriter (D4) on characters", "", md(rw), ""]
    plain = ci[ci.attack == "A0"].copy()
    plain["95% CI"] = [f"{100 * r.rate:.1f}% [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]" for r in plain.itertuples()]
    report += ["## Uncertainty: plain request, 95% bootstrap intervals over items", "",
               md(plain[["run", "guardrail", "95% CI", "n_items", "n_images"]]), "",
               "All attacks: [ci.csv](ci.csv). Intervals resample items (prompts, characters, style prompts).", ""]
    if not cov.empty:
        report += ["## Coverage: exact-copy prompts with a training image still online", "", md(cov), "",
                   "Copy rates are computed only on prompts whose training image could be downloaded.", ""]
    report += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + ["", "Error analysis: [errors.md](errors.md)"]
    (out / "README.md").write_text("\n".join(report) + "\n")
    print(f"wrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    main()
