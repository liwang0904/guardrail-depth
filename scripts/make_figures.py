"""Milestone figures (static PDF/PNG for the report). Each figure is skipped if its inputs are missing.

    python scripts/make_figures.py [--out outputs/figures]

Colors: reference categorical palette slots 1-4 in fixed order (validated: adjacent CVD dE >= 9.1);
aqua/yellow are below 3:1 contrast on the surface, so every figure carries a legend and its numbers
are also tabulated in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.paths import gt_image_dir, output_root  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # blue, orange, aqua, yellow
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 7.5, "savefig.dpi": 220,
})


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ----------------------------------------------------------------------------- 1. token window
def fig_token_window(out: Path) -> None:
    f = output_root() / "analysis/token_window.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    panels = [("memorized_sd1", "Memorized captions (n = 500)", ["A0", "A3-override", "A3-trunc-rep"]),
              ("copycat_characters", "CopyCat characters (n = 50)", ["A0", "A1-desc", "A3-trunc-neu"])]
    names = {"A0": "direct request", "A1-desc": "60-word description lure", "A3-override": "override suffix",
             "A3-trunc-rep": "truncation (repeat)", "A3-trunc-neu": "truncation (filler)"}
    defenses = ["D1", "D2a", "D3"]
    labels = {"D1": "D1 task\ninstruction", "D2a": "D2a in-prompt\nnegation", "D3": "D3 chain-of-\nthought"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, (ds, title, attacks) in zip(axes, panels):
        x = np.arange(len(defenses))
        w = 0.24
        for k, atk in enumerate(attacks):
            sub = df[(df.dataset == ds) & (df.attack == atk)].set_index("defense").reindex(defenses)
            vals = 100 * sub.mean_frac_tail_visible.values
            ax.bar(x + (k - 1) * w, vals, w, color=SLOTS[k], edgecolor=SURFACE, linewidth=1.5, label=names[atk])
            for xi, v in zip(x + (k - 1) * w, vals):
                if v < 2:  # an empty bar would read as missing data: say it is zero
                    ax.text(xi, 1.5, "0%", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x, [labels[d] for d in defenses])
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_ylabel("guardrail text after the request\nthat reaches the encoder (%)")
    fig.tight_layout()
    save(fig, out, "fig_token_window")


# ----------------------------------------------------------------------------- 2. Wen detector
def fig_detection(out: Path) -> None:
    f = output_root() / "detection/sd14_dev/scores.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    t = output_root() / "detection/sd14_dev_trunc/scores.csv"
    if t.exists():
        df = pd.concat([df, pd.read_csv(t)])
    order = [("A0", "D0"), ("A0", "D0r"), ("A0", "D1"), ("A0", "D2a"), ("A0", "D2b"), ("A0", "D3"), ("A0", "C1"),
             ("A3-trunc-rep", "D0"), ("A3-trunc-rep", "D1"), ("A3-trunc-rep", "D3")]
    order = [o for o in order if ((df.attack == o[0]) & (df.defense == o[1])).any()]
    groups = [("VM", "verbatim memorized", SLOTS[0]), ("TM", "template memorized", SLOTS[1]),
              ("control", "not memorized (control)", SLOTS[2])]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for k, (g, label, color) in enumerate(groups):
        means, los, his = [], [], []
        for atk, dfn in order:
            v = df[(df.attack == atk) & (df.defense == dfn) & (df.mem_type == g)]["score"].values
            boot = [np.mean(np.random.default_rng(i).choice(v, len(v))) for i in range(500)]
            means.append(v.mean()); los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
        x = np.arange(len(order)) + (k - 1) * 0.22
        ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - means], fmt="o", color=color,
                    ms=5, mec=SURFACE, mew=1.2, elinewidth=1.2, capsize=0, label=label)
    short = {"D0": "none", "D0r": "wrapper", "D1": "task instr.", "D2a": "neg. text", "D2b": "neg. buffer",
             "D3": "CoT", "C1": "random words"}
    ax.set_xticks(np.arange(len(order)),
                  [f"{d}\n{short[d]}" if a == "A0" else f"{d} + trunc.\n{short[d]}" for a, d in order])
    ax.set_ylabel("first-step text-conditional\nnoise magnitude (Wen et al.)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig_detection")


# ----------------------------------------------------------------------------- 3. image grids
def _grid(run: Path, rows: list, cols: list, out: Path, name: str, gt_dir: Path | None, score_col: str | None,
          row_label, col_titles: list[str], cell: float = 0.95) -> None:
    rec = pd.read_csv(run / "metrics.csv") if (run / "metrics.csv").exists() else None
    recs = [json.loads(l) for f in sorted(run.glob("records*.jsonl")) for l in open(f)]
    if not recs:
        return
    ncol = len(cols) + (1 if gt_dir else 0)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(cell * ncol, cell * len(rows) + 0.3))
    axes = np.atleast_2d(axes)
    for i, item in enumerate(rows):
        c0 = 0
        if gt_dir:
            p = gt_dir / f"{int(item):04d}.jpg"
            ax = axes[i, 0]
            if p.exists():
                ax.imshow(Image.open(p).convert("RGB").resize((256, 256)))
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
            ax.set_ylabel(row_label(item), fontsize=6.5, rotation=0, ha="right", va="center", color=INK2)
            c0 = 1
        for j, (atk, dfn) in enumerate(cols):
            ax = axes[i, c0 + j]
            hit = [r for r in recs if str(r["item_id"]) == str(item) and r["attack_code"] == atk
                   and r["defense_code"] == dfn and r["seed"] == 0]
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
            if not hit:
                continue
            ax.imshow(Image.open(run / hit[0]["image"]).convert("RGB").resize((256, 256)))
            if rec is not None and score_col and score_col in rec:
                v = rec.loc[rec.image == hit[0]["image"], score_col]
                if len(v) and pd.notna(v.iloc[0]):
                    ax.set_xlabel(f"{v.iloc[0]:.2f}", fontsize=6.5, color=INK2, labelpad=1)
            if not gt_dir and j == 0:
                ax.set_ylabel(row_label(item), fontsize=6.5, rotation=0, ha="right", va="center", color=INK2)
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=6.5, color=INK, fontweight="bold")
    fig.subplots_adjust(wspace=0.04, hspace=0.18)
    save(fig, out, name)


def fig_memorized_grid(out: Path) -> None:
    run = output_root() / "pilot_local/memorized"
    if not run.exists():
        return
    from guardrail_depth.data import load_manifest

    m = load_manifest("memorized_sd1").set_index("id")
    # #485 (a memorized meme of a real actor) is scored in the tables but not shown by default
    ids = [202, 274, 112, 118, 154]
    cols = [("A0", d) for d in ["D0", "D1", "D2a", "D2b", "D3", "C1"]]
    _grid(run, ids, cols, out, "fig_memorized_grid", gt_image_dir() / "sd1", "sscd_gt",
          lambda i: f"{m.loc[i, 'mem_type']} #{i}", ["training image", "D0 none", "D1 task instr.", "D2a neg. text",
                                                      "D2b neg. buffer", "D3 CoT", "C1 random words"])


def fig_copycat_grid(out: Path) -> None:
    run = output_root() / "pilot_local/copycat"
    if not run.exists():
        return
    from guardrail_depth.data import load_manifest

    c = load_manifest("copycat_characters").set_index("id")
    ids = [23, 16, 41, 28, 47]
    cols = [("A0", "D0"), ("A0", "D1"), ("A0", "D2b-kw"), ("A0", "D4+D2b"),
            ("A1-desc", "D0"), ("A1-desc", "D1"), ("A1-desc", "D4+D2b"), ("A1-kwLAION", "D0"), ("A1-kwLAION", "D2b-kw")]
    titles = ["name\nD0", "name\nD1", "name\nD2b-kw", "name\nD4+D2b", "desc. lure\nD0", "desc. lure\nD1",
              "desc. lure\nD4+D2b", "keywords\nD0", "keywords\nD2b-kw"]
    _grid(run, ids, cols, out, "fig_copycat_grid", None, "p_target", lambda i: c.loc[i, "target"], titles)


# ----------------------------------------------------------------------------- 4. x0 lock-in
def fig_x0(out: Path) -> None:
    f = output_root() / "pilot_local/memorized/metrics.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    cols = sorted(c for c in df if c.startswith("sscd_x0_t"))
    if not cols:
        return
    steps = [int(c.split("_t")[1]) for c in cols]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, mt in zip(axes, ["VM", "TM"]):
        for k, d in enumerate(["D0", "D1", "D3", "C1"]):
            sub = df[(df.mem_type == mt) & (df.defense_code == d)][cols].mean()
            ax.plot(steps, sub.values, "-o", color=SLOTS[k], lw=2, ms=4.5, mec=SURFACE, mew=1.2,
                    label={"D0": "D0 none", "D1": "D1 task instruction", "D3": "D3 chain-of-thought",
                           "C1": "C1 random words"}[d])
        ax.axhline(0.5, color=MUTED, lw=0.8)
        if mt == "VM":
            ax.text(steps[0], 0.51, "copy threshold (SSCD 0.5)", ha="left", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_title({"VM": "Verbatim memorized prompts", "TM": "Template memorized prompts"}[mt], loc="left")
        ax.set_xlabel("denoising step (of 50)")
    axes[0].set_ylabel("SSCD(predicted x0, training image)")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_x0_lockin")


# ----------------------------------------------------------------------------- 5. attention shares
def fig_attention(out: Path) -> None:
    runs = [("Memorized captions", output_root() / "pilot_local/memorized"),
            ("CopyCat: description lure", output_root() / "pilot_local/copycat")]
    def has_rows(run):
        if not (run / "records.jsonl").exists():
            return False
        df = pd.DataFrame([json.loads(l) for l in open(run / "records.jsonl")])
        return "copycat" not in str(run) or (df.attack_code == "A1-desc").any()

    runs = [(t, r) for t, r in runs if has_rows(r)]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3.5 * len(runs), 2.3), sharex=True)
    axes = np.atleast_1d(axes)
    parts = [("attn_bos_t0", "start token"), ("attn_user_t0", "user's request"),
             ("attn_guardrail_t0", "guardrail text"), ("attn_summary_t0", "end/padding")]
    for ax, (title, run) in zip(axes, runs):
        df = pd.DataFrame([json.loads(l) for l in open(run / "records.jsonl")])
        if "copycat" in str(run):
            df = df[df.attack_code == "A1-desc"]
        # rewrites (D4) have no separate user/guardrail span, so they are left out of this split
        order = [d for d in ["D0", "D0r", "D1", "D2a", "D2b", "D2b-kw", "D3", "C1"]
                 if d in set(df.defense_code) and df[df.defense_code == d].attn_user_t0.notna().any()]
        means = df.groupby("defense_code")[[p for p, _ in parts]].mean().reindex(order)
        left = np.zeros(len(order))
        for k, (p, label) in enumerate(parts):
            ax.barh(np.arange(len(order)), means[p].values, left=left, color=SLOTS[k], height=0.6,
                    edgecolor=SURFACE, linewidth=1.5, label=label)
            left += means[p].values
        ax.set_yticks(np.arange(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("share of cross-attention at step 1")
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, out, "fig_attention")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=output_root() / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_token_window, fig_detection, fig_memorized_grid, fig_copycat_grid, fig_x0, fig_attention):
        fn(args.out)


if __name__ == "__main__":
    main()
