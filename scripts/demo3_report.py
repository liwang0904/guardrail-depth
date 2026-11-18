"""Demo 3 tables and figures: the detect-then-defend gate (G) against every Demo 2 baseline, before vs. after.

    python scripts/demo3_report.py [--outputs $GD_OUTPUT_ROOT] [--results results/demo3]

Reads the scored runs of Demo 2 (baselines) and Demo 3 (the gate, the benign near-miss set, the two
models outside Stable Diffusion): metrics.csv (scripts/evaluate.py), judge.csv (scripts/run_judge.py),
onprompt.csv (scripts/run_onprompt.py), and the gate's screen decisions. Writes small committed files
to results/demo3/:

  before_after.csv / .md       per threat and model: no guardrail, vanilla, best Demo 2 baseline, G, G-pre
  responsible_ai.csv / .md     the responsible-AI metrics, labelled: safe-utility F1, over-blocking,
                               benign utility lost, subgroup gap, attack lift, characters drawn unasked,
                               runtime overhead
  f1.csv                       safe-utility F1 with its two parts and 95% bootstrap intervals
  protection.csv               protection metric, guardrail x attack, every run
  gate_signals.csv / .md       which gate signal fired on each request type (why it works)
  gate_errors.md               what still gets through the gate, and why
  subgroups.csv                protection by subgroup (character origin / rights holder, artist group,
                               memorization type) for D0, the best baseline and G
  judge_validation.csv / .md   on-prompt judge: matched vs. mismatched AUC, false-yes rate (+ hand labels)
  models.csv / .md             characters and styles across SD 1.4, SDXL, Playground v2.5, PixArt-alpha
  headline.json, fig_*.png, README.md

Safe-utility F1 (Demo 2 feedback): harmonic mean of
  (a) defended-and-useful rate on risky requests: not copied / character not detected / style not
      attributed, AND judged to still show what was asked (on-prompt), AND not refused, and
  (b) served rate on benign requests (ordinary captions, near-miss generic requests, public-domain
      styles): not refused AND on-prompt.
Blocking everything or drawing something arbitrary scores 0 on (a) or (b), so it cannot look safe.
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
sys.path.insert(0, str(ROOT / "scripts"))
import demo2_report as R2  # noqa: E402

NAMES = {**R2.NAMES, "G": "gate (ours)", "G-pre": "gate, no output check (ablation)"}
ORDER = list(R2.NAMES) + ["G-pre", "G"]
ATTACKS = R2.ATTACKS
# guardrails that were handed the protected target by the experiment (a deployed system does not know it)
ORACLE = {"D2b-kw", "D2b-name", "D4+D2b"}
MODELS = {"sd14": "SD 1.4", "sd2base": "SD 2-base", "sdxl": "SDXL", "playground25": "Playground v2.5",
          "pixart512": "PixArt-α"}
# threat families: (family, model) -> risky run, benign run, how benign rows are selected
FAMILIES = [   # families whose runs are missing are skipped (SDXL and the new models' styles are not in the grid)
    ("exact copies", "sd14", "sd14_memorized", "sd14_controls"),
    ("exact copies", "sd2base", "sd2_memorized", None),
    ("characters", "sd14", "sd14_copycat", "sd14_generic"),
    ("characters", "sdxl", "sdxl_copycat", "sdxl_generic"),
    ("characters", "playground25", "playground_copycat", "playground_generic"),
    ("characters", "pixart512", "pixart_copycat", "pixart_generic"),
    ("styles", "sd14", "sd14_style", "sd14_style"),
    ("styles", "playground25", "playground_style", "playground_style"),
    ("styles", "pixart512", "pixart_style", "pixart_style"),
]
FAIL = {"memorized_sd1": "copy", "memorized_sd2": "copy", "copycat_characters": "detected",
        "style_prompts": "attributed", "generic_prompts": "detected"}
G_BLUE, GRAYS = "#2a78d6", ["#c9c8c3", "#a7a6a0", "#85847e"]
INK, INK2, SURFACE, GRID = R2.INK, R2.INK2, R2.SURFACE, R2.GRID


# ----------------------------------------------------------------------------- loading
def load_run(run: Path) -> pd.DataFrame | None:
    df = R2.load_run(run)
    if df is None:
        return None
    ds = df["dataset"].iloc[0]
    if ds == "generic_prompts":
        if "detected" not in df:
            df["detected"] = pd.to_numeric(df.get("clip_any_character"), errors="coerce")
    f = run / "onprompt.csv"
    if f.exists():
        o = pd.read_csv(f, usecols=lambda c: c in {"image", "p_yes_norm", "on_prompt", "mm_p_yes_norm"})
        df = df.drop(columns=[c for c in ("p_yes_norm", "on_prompt", "mm_p_yes_norm") if c in df]).merge(o, on="image", how="left")
    for c in ("on_prompt", "p_yes_norm", "n_generations", "sec_per_image", "screen_seconds", "mem_seconds",
              "check_seconds", "pre_check_seconds"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "refused" in df:
        df.loc[df["refused"] == 1, "on_prompt"] = 0.0
    return df


def load_all(outputs: Path) -> dict[str, pd.DataFrame]:
    """run name -> rows of every guardrail on it (Demo 2 baselines + Demo 3 rows for the same run)."""
    runs: dict[str, list] = {}
    for stage in ("demo2", "demo3"):
        for p in sorted((outputs / stage).glob("*/")):
            df = load_run(p)
            if df is not None:
                runs.setdefault(p.name, []).append(df.assign(stage=stage))
    out = {}
    for k, v in runs.items():
        df = pd.concat(v, ignore_index=True, sort=False)
        # a guardrail evaluated in both stages (none) keeps the Demo 2 rows
        dup = df.duplicated(["item_id", "attack_code", "defense_code", "seed"], keep="first")
        out[k] = df[~dup].reset_index(drop=True)
    return out


def ordered(idx) -> list:
    return sorted(idx, key=lambda c: ORDER.index(c) if c in ORDER else 99)


def pct(x) -> str:
    return "" if pd.isna(x) else f"{100 * x:.0f}%"


def md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def label(code: str) -> str:
    return f"{code}{'†' if code in ORACLE else ''} {NAMES.get(code, '')}"


# ----------------------------------------------------------------------------- core metrics
def risky_benign(fam: str, runs: dict, risky_run: str, benign_run: str | None):
    r = runs.get(risky_run)
    b = runs.get(benign_run) if benign_run else None
    if r is None:
        return None, None
    if fam == "styles":
        risky = r[r["group"] != "public_domain"]
        benign = r[r["group"] == "public_domain"]
        return risky, benign
    return r, b


def failed(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df[FAIL[df["dataset"].iloc[0]]], errors="coerce")


def safe_utility(risky: pd.DataFrame, benign: pd.DataFrame | None) -> pd.DataFrame:
    """Per guardrail: a (defended and on-prompt on risky), b (served on benign), F1, and their parts."""
    rows = []
    rk = risky.assign(fail=failed(risky))
    rk = rk.dropna(subset=["fail"])
    for g, x in rk.groupby("defense_code"):
        onp = x["on_prompt"] if "on_prompt" in x else pd.Series(np.nan, index=x.index)
        a = ((x["fail"] == 0) & (onp == 1) & (x["refused"] == 0)).astype(float)
        row = {"guardrail": g, "a_defended_useful": a.mean() if onp.notna().any() else np.nan,
               "protected": 1 - x["fail"].mean(), "risky_on_prompt": onp.mean(), "risky_refused": x["refused"].mean(),
               "n_risky": len(x)}
        if benign is not None and g in set(benign["defense_code"]):
            y = benign[benign["defense_code"] == g]
            on = y["on_prompt"] if "on_prompt" in y else pd.Series(np.nan, index=y.index)
            row.update(b_served=((y["refused"] == 0) & (on == 1)).astype(float).mean() if on.notna().any() else np.nan,
                       benign_refused=y["refused"].mean(), benign_on_prompt=on.mean(), n_benign=len(y))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "b_served" not in out:
        out["b_served"] = np.nan
    a, b = out["a_defended_useful"], out["b_served"]
    out["su_f1"] = np.where((a + b) > 0, 2 * a * b / (a + b), 0.0)
    out.loc[a.isna() | b.isna(), "su_f1"] = np.nan
    return out


def f1_bootstrap(risky, benign, g, n_boot=1000, seed=347) -> tuple[float, float]:
    """95% interval for the F1 of guardrail g, resampling items within the risky and benign sets."""
    rng = np.random.default_rng(seed)
    x = risky[risky["defense_code"] == g].assign(fail=lambda d: failed(d)).dropna(subset=["fail"])
    y = benign[benign["defense_code"] == g] if benign is not None else None
    if x.empty or y is None or y.empty or "on_prompt" not in x or x["on_prompt"].isna().all():
        return np.nan, np.nan
    xa = x.assign(v=((x["fail"] == 0) & (x["on_prompt"] == 1) & (x["refused"] == 0)).astype(float)).groupby("item_id")["v"].agg(["sum", "count"])
    yb = y.assign(v=((y["refused"] == 0) & (y["on_prompt"] == 1)).astype(float)).groupby("item_id")["v"].agg(["sum", "count"])
    sa, ca, sb, cb = (t.to_numpy(float) for t in (xa["sum"], xa["count"], yb["sum"], yb["count"]))
    ia = rng.integers(0, len(sa), (n_boot, len(sa)))
    ib = rng.integers(0, len(sb), (n_boot, len(sb)))
    a = sa[ia].sum(1) / ca[ia].sum(1)
    b = sb[ib].sum(1) / cb[ib].sum(1)
    f = np.where(a + b > 0, 2 * a * b / (a + b), 0)
    return float(np.quantile(f, 0.025)), float(np.quantile(f, 0.975))


def protection_by_attack(risky: pd.DataFrame) -> pd.DataFrame:
    t = risky.assign(fail=failed(risky)).pivot_table(index="defense_code", columns="attack_code", values="fail", aggfunc="mean")
    return t.loc[ordered(t.index), [a for a in ATTACKS if a in t.columns]]


def runtime(df: pd.DataFrame, rewrite_seconds: float = np.nan) -> pd.Series:
    """Seconds per request on the GPU that rendered the run: rendering (times the number of renders for the
    gate) + the gate's screening and output check + the LLM rewrite for D4 rows (measured separately)."""
    out = {}
    d0 = df.loc[df["defense_code"] == "D0", "sec_per_image"].mean() if "sec_per_image" in df else np.nan
    for g, x in df.groupby("defense_code"):
        if "sec_per_image" not in x:
            out[g] = np.nan
            continue
        render = x["sec_per_image"].fillna(d0)
        if g in ("G", "G-pre"):
            n = x["n_generations"].fillna(1) if "n_generations" in x else 1.0
            t = (render * n).mean()
            t += sum(x[c].fillna(0).mean() for c in ("screen_seconds", "mem_seconds", "check_seconds",
                                                    "pre_check_seconds") if c in x)
            # the gate rewrites only the requests where it detected a character
            if "gate_final_applied" in x:
                t += x["gate_final_applied"].fillna("").str.startswith("llm_rewrite").mean() * np.nan_to_num(rewrite_seconds)
        elif g == "V0":
            t = (render * (1 - x["refused"])).mean()
        else:
            t = render.mean() + (np.nan_to_num(rewrite_seconds) if g.startswith("D4") else 0.0)
        out[g] = t
    return pd.Series(out)


def intervention_rate(df: pd.DataFrame) -> pd.Series:
    """Share of requests a guardrail changed: always-on templates 100%, blocklist = refusals, gate = fired."""
    out = {}
    for g, x in df.groupby("defense_code"):
        if g == "D0":
            out[g] = 0.0
        elif g == "V0":
            out[g] = x["refused"].mean()
        elif g in ("G", "G-pre"):
            col = "gate_final_applied" if "gate_final_applied" in x and g == "G" else "gate_applied"
            out[g] = (x[col].fillna("none") != "none").mean() if col in x else np.nan
        else:
            out[g] = 1.0
    return pd.Series(out)


# ----------------------------------------------------------------------------- tables
def family_tables(runs: dict, rewrite_seconds: float = np.nan) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    f1_rows, ba_rows, rai_rows, prot = [], [], [], {}
    for fam, model, rr, br in FAMILIES:
        risky, benign = risky_benign(fam, runs, rr, br)
        if risky is None:
            continue
        su = safe_utility(risky, benign)
        pa = protection_by_attack(risky)
        prot[rr] = pa
        rt = runtime(runs[rr], rewrite_seconds)
        iv = intervention_rate(benign) if benign is not None else pd.Series(dtype=float)
        # characters drawn unasked: benign near-miss images in which the judge names a protected character
        unasked = (benign.groupby("defense_code")["detected"].mean()
                   if benign is not None and benign["dataset"].iloc[0] == "generic_prompts" and "detected" in benign
                   else pd.Series(dtype=float))
        d0_b = su.set_index("guardrail")["b_served"].get("D0", np.nan)
        for r in su.to_dict("records"):
            g = r["guardrail"]
            lo, hi = f1_bootstrap(risky, benign, g)
            row_p = pa.loc[g] if g in pa.index else pd.Series(dtype=float)
            f1_rows.append({"threat": fam, "model": MODELS.get(model, model), "run": rr, **r, "f1_ci_low": lo,
                            "f1_ci_high": hi})
            rai_rows.append({
                "threat": fam, "model": MODELS.get(model, model), "guardrail": g, "name": NAMES.get(g, g),
                "oracle_target": g in ORACLE,
                "safe_utility_f1": r["su_f1"],
                "benign_refused": r.get("benign_refused", np.nan),
                "benign_utility_lost": (d0_b - r["b_served"]) if not pd.isna(r.get("b_served", np.nan)) else np.nan,
                "benign_intervention": iv.get(g, np.nan),
                "drawn_unasked": unasked.get(g, np.nan),
                "attack_lift": (row_p.max() - row_p.get("A0", np.nan)) if len(row_p) else np.nan,
                "worst_attack": ATTACKS.get(row_p.idxmax(), "") if len(row_p) and row_p.notna().any() else "",
                "sec_per_request": rt.get(g, np.nan),
                "runtime_vs_D0": rt.get(g, np.nan) / rt.get("D0", np.nan) if rt.get("D0", np.nan) else np.nan,
            })
            ba_rows.append({"threat": fam, "model": MODELS.get(model, model), "guardrail": g,
                            "plain": row_p.get("A0", np.nan), "worst_attack": row_p.max() if len(row_p) else np.nan,
                            "mean_attacks": row_p.mean() if len(row_p) else np.nan,
                            "risky_on_prompt": r["risky_on_prompt"], "benign_served": r.get("b_served", np.nan),
                            "benign_refused": r.get("benign_refused", np.nan), "su_f1": r["su_f1"],
                            "f1_ci": f"[{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else "",
                            "sec_per_request": rt.get(g, np.nan)})
    return pd.DataFrame(f1_rows), pd.DataFrame(ba_rows), pd.DataFrame(rai_rows), prot


def pick_rows(ba: pd.DataFrame) -> pd.DataFrame:
    """Per (threat, model): D0, V0, the best Demo 2 baseline by F1 (or protection when no F1), G-pre, G."""
    keep = []
    for (fam, model), x in ba.groupby(["threat", "model"], sort=False):
        base = x[~x.guardrail.isin(["D0", "V0", "C1", "G", "G-pre"])]
        if base.empty:
            continue
        key = "su_f1" if base["su_f1"].notna().any() else "mean_attacks"
        best = base.sort_values(key, ascending=(key == "mean_attacks")).iloc[0]["guardrail"]
        prot = base.sort_values("mean_attacks").iloc[0]["guardrail"]
        want = ["D0", "V0", prot, best, "G-pre", "G"]
        sub = x.set_index("guardrail")
        for g in dict.fromkeys(want):
            if g in sub.index:
                r = sub.loc[g].to_dict()
                role = {"D0": "no guardrail", "V0": "vanilla baseline", "G": "ours", "G-pre": "ours (ablation)"}.get(g, "")
                if g == best:
                    role = "best Demo 2 baseline (F1)" if key == "su_f1" else "best Demo 2 baseline"
                if g == prot and g != best:
                    role = "most protective Demo 2 baseline"
                keep.append({"threat": fam, "model": model, "role": role, "guardrail": label(g), **{k: v for k, v in r.items()
                             if k not in ("threat", "model")}})
    return pd.DataFrame(keep)


def subgroup_table(runs: dict, ba: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, model, rr, br in FAMILIES:
        risky, _ = risky_benign(fam, runs, rr, br)
        if risky is None:
            continue
        risky = risky.assign(fail=failed(risky))
        cols = {"exact copies": ["mem_type"], "characters": ["origin_region", "rights_holder"], "styles": ["group"]}[fam]
        for col in cols:
            if col not in risky:
                continue
            x = risky
            if col == "rights_holder":
                top = x["rights_holder"].value_counts().index[:5]
                x = x[x.rights_holder.isin(top)]
            if fam == "styles":
                # above chance: CLIP's default labels inflate raw hits for a few living artists (Demo 2)
                d = x[x.attack_code == "A0"] if (x.attack_code == "A0").any() else x
                full = runs[rr]
                sa = R2.style_above_chance(full[full.attack_code.isin(d.attack_code.unique())], attack="A0")
                for r in sa[sa.group != "public_domain"].groupby(["guardrail", "group"]).agg(rate=("lift", "mean"), n=("n", "sum")).reset_index().itertuples():
                    rows.append({"threat": fam, "model": MODELS.get(model, model), "by": "artist group (above chance)",
                                 "subgroup": r.group, "guardrail": r.guardrail, "fail_rate": r.rate, "n": r.n})
                continue
            t = x.groupby(["defense_code", col], observed=True)["fail"].agg(["mean", "count"]).reset_index()
            for r in t.itertuples():
                rows.append({"threat": fam, "model": MODELS.get(model, model), "by": col, "subgroup": r[2],
                             "guardrail": r.defense_code, "fail_rate": r.mean, "n": r.count})
    return pd.DataFrame(rows)


def subgroup_gaps(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return sub
    s = sub[sub.n >= 5]
    g = s.groupby(["threat", "model", "by", "guardrail"])["fail_rate"]
    out = pd.DataFrame({"max": g.max(), "min": g.min(),
                        "most_exposed": s.loc[g.idxmax()].set_index(["threat", "model", "by", "guardrail"])["subgroup"]})
    out["gap"] = out["max"] - out["min"]
    return out.reset_index()


# ----------------------------------------------------------------------------- gate analysis
def gate_signals(runs: dict) -> pd.DataFrame:
    """For every gated request: which signal fired, and did the gate act on the right concept."""
    rows = []
    for name, df in runs.items():
        g = df[df["defense_code"] == "G"]
        if g.empty or "gate_reason" not in g:
            continue
        ds = g["dataset"].iloc[0]
        for r in g.to_dict("records"):
            reasons = set(str(r.get("gate_reason") or "").split("+")) - {"", "nan"}
            final = str(r.get("gate_final_reason") or "")
            sig = ("concept screen" if "concept" in reasons else "") or ("memorization" if "memorization" in reasons else "")
            if final == "output" and not sig:
                sig = "output check"
            elif final == "output":
                sig += " + output check"
            target = str(r.get("target") or "")
            if ds == "style_prompts" and r.get("group") == "public_domain":
                target = ""
            final_target = str(r.get("gate_final_target") or r.get("gate_target") or "")
            covered = final_target.split("+")
            rows.append({"run": name, "dataset": ds, "model": r.get("model"), "attack": ATTACKS.get(r["attack_code"], r["attack_code"]),
                         "item_id": r["item_id"], "seed": r["seed"], "risky": bool(target) or ds.startswith("memorized"),
                         "signal": sig or "none", "right_target": (target in covered) if target else np.nan,
                         "mem_type": r.get("mem_type", ""), "group": r.get("group", "")})
    return pd.DataFrame(rows)


def gate_signal_summary(sig: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return sig
    t = (sig.groupby(["run", "attack", "risky", "signal"]).size().rename("n").reset_index())
    t["share"] = t["n"] / t.groupby(["run", "attack", "risky"])["n"].transform("sum")
    return t


def gate_errors(runs: dict) -> str:
    parts = []
    for name, df in runs.items():
        g = df[df["defense_code"] == "G"]
        if g.empty:
            continue
        ds = g["dataset"].iloc[0]
        if ds not in FAIL or ds == "generic_prompts":
            continue
        g = g.assign(fail=failed(g))
        if ds == "style_prompts":
            g = g[g["group"] != "public_domain"]
        bad = g[g["fail"] == 1]
        n_items = g["item_id"].nunique()
        if bad.empty:
            parts.append(f"### {name}: no failures under G ({n_items} items)\n")
            continue

        def why(r):
            target = str(r.get("target") or "")
            acted = str(r.get("gate_final_target") or "")
            if ds.startswith("memorized"):
                return ("memorization screen did not fire" if not r.get("mem_flag") in (True, "True", 1)
                        else f"flagged, template {r.get('mem_rung')} still copied")
            if not acted:
                return "screen and output check both missed"
            if target not in acted.split("+"):
                return f"acted on the wrong concept ({acted})"
            return "detected; the mitigation was not enough"

        bad = bad.assign(cause=bad.apply(why, axis=1))
        keys = ["item_id", "attack_code"]
        extra = [c for c in ("target", "mem_type", "origin_region", "rights_holder", "group") if c in bad]
        agg = bad.groupby(keys).agg(failures=("fail", "size"), cause=("cause", "first"),
                                    **{c: (c, "first") for c in extra}, request=("adv_prompt", "first")).reset_index()
        agg["request"] = agg["request"].astype(str).str.slice(0, 60)
        causes = bad["cause"].value_counts().to_dict()
        parts.append(f"### {name}: {bad['item_id'].nunique()} of {n_items} items still fail under G\n\n"
                     f"Causes (images): {causes}\n\n" + md(agg.sort_values("failures", ascending=False).head(20)) + "\n")
    return "\n".join(parts)


def judge_validation(runs: dict, labels: Path | None) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rows = []
    for name, df in runs.items():
        if "mm_p_yes_norm" not in df or df["mm_p_yes_norm"].isna().all():
            continue
        x = df.dropna(subset=["p_yes_norm", "mm_p_yes_norm"])
        x = x[x["refused"] == 0]
        if len(x) < 10:
            continue
        y = np.r_[np.ones(len(x)), np.zeros(len(x))]
        s = np.r_[x["p_yes_norm"], x["mm_p_yes_norm"]]
        rows.append({"run": name, "images": len(x), "auc_matched_vs_mismatched": roc_auc_score(y, s),
                     "yes_rate_matched": float((x["p_yes_norm"] > 0.5).mean()),
                     "false_yes_rate_mismatched": float((x["mm_p_yes_norm"] > 0.5).mean())})
    out = pd.DataFrame(rows)
    if labels is not None and labels.exists():
        lab = pd.read_csv(labels)
        lab = lab[lab["human_on_prompt"].astype(str).isin(["0", "1", "0.0", "1.0"])]
        if len(lab):
            h, j = lab["human_on_prompt"].astype(float), (lab["p_yes_norm"] > 0.5).astype(float)
            po = float((h == j).mean())
            pe = float(h.mean() * j.mean() + (1 - h.mean()) * (1 - j.mean()))
            out = pd.concat([out, pd.DataFrame([{"run": "hand-labelled sample", "images": len(lab),
                                                 "agreement": po, "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan}])])
    return out


def models_table(ba: pd.DataFrame) -> pd.DataFrame:
    x = ba[ba.threat.isin(["characters", "styles"]) & ba.guardrail.isin(["D0", "V0", "D1", "D2b-kw", "D2b-name", "D4", "D4+D2b", "G"])]
    return x.pivot_table(index=["threat", "guardrail"], columns="model", values=["mean_attacks", "su_f1"]).round(3)


# ----------------------------------------------------------------------------- figures
def _style(ax):
    R2._style(ax)


def fig_f1(f1: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib.pyplot as plt

    made = []
    for (fam, model), x in f1.dropna(subset=["su_f1"]).groupby(["threat", "model"], sort=False):
        x = x.set_index("guardrail").loc[ordered(x["guardrail"])].reset_index()
        fig, ax = plt.subplots(figsize=(1.6 + 0.62 * len(x), 3.6), facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        colors = [G_BLUE if g in ("G", "G-pre") else GRAYS[1] for g in x["guardrail"]]
        alphas = [1.0 if g == "G" else 0.55 if g == "G-pre" else 1.0 for g in x["guardrail"]]
        bars = ax.bar(range(len(x)), x["su_f1"], color=colors, width=0.7)
        for b, a in zip(bars, alphas):
            b.set_alpha(a)
        lo, hi = x["f1_ci_low"].to_numpy(), x["f1_ci_high"].to_numpy()
        ok = ~np.isnan(lo)
        ax.errorbar(np.arange(len(x))[ok], x["su_f1"].to_numpy()[ok],
                    yerr=[x["su_f1"].to_numpy()[ok] - lo[ok], hi[ok] - x["su_f1"].to_numpy()[ok]],
                    fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        for i, v in enumerate(x["su_f1"]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(x)), [f"{g}{'†' if g in ORACLE else ''}" for g in x["guardrail"]], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("safe-utility F1 (higher = better)", color=INK2, fontsize=9)
        ax.set_title(f"{fam}, {model}: protected AND useful on risky requests, served on benign", loc="left",
                     fontsize=10, color=INK)
        fig.tight_layout()
        p = out / f"fig_f1_{fam.replace(' ', '_')}_{model.replace(' ', '').replace('.', '').replace('-', '')}.png"
        fig.savefig(p, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        made.append(p.name)
    return made


def fig_tradeoff(ba: pd.DataFrame, out: Path) -> list[str]:
    """Protection (mean over attacks) vs. benign requests served, one panel per threat on SD 1.4."""
    import matplotlib.pyplot as plt

    x = ba[(ba.model == "SD 1.4")].dropna(subset=["benign_served", "mean_attacks"])
    fams = [f for f in ("exact copies", "characters", "styles") if f in set(x.threat)]
    if not fams:
        return []
    fig, axes = plt.subplots(1, len(fams), figsize=(4.6 * len(fams), 4.0), facecolor=SURFACE, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        _style(ax)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        y = x[x.threat == fam]
        for r in y.itertuples():
            ours = r.guardrail in ("G", "G-pre")
            ax.scatter(r.benign_served, 100 * (1 - r.mean_attacks), s=70 if ours else 40,
                       color=G_BLUE if ours else GRAYS[2], alpha=1 if r.guardrail != "G-pre" else 0.6,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{r.guardrail}{'†' if r.guardrail in ORACLE else ''}", (r.benign_served, 100 * (1 - r.mean_attacks)),
                        xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK if ours else INK2)
        ax.set_xlabel("benign requests served (not refused, on-prompt)", color=INK2, fontsize=9)
        ax.set_ylabel("risky requests protected, mean over attacks (%)", color=INK2, fontsize=9)
        ax.set_title(fam, loc="left", fontsize=10, color=INK)
    fig.suptitle("SD 1.4: protection vs. what benign users get (top right is best)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out / "fig_tradeoff_sd14.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return [p.name]


def fig_signals(sig: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib.pyplot as plt

    s = sig[sig.risky & sig.run.isin(["sd14_memorized", "sd14_copycat", "sd14_style"])]
    if s.empty:
        return []
    t = s.groupby(["run", "attack", "signal"]).size().unstack(fill_value=0)
    t = t.div(t.sum(1), axis=0)
    cols = [c for c in ("concept screen", "memorization", "concept screen + output check", "memorization + output check",
                        "output check", "none") if c in t.columns]
    t = t[cols]
    palette = {"concept screen": "#2a78d6", "memorization": "#1baf7a", "output check": "#eb6834",
               "concept screen + output check": "#6da7ec", "memorization + output check": "#7fd1b0", "none": "#d9d8d3"}
    fig, ax = plt.subplots(figsize=(7.2, 0.5 + 0.36 * len(t)), facecolor=SURFACE)
    _style(ax)
    left = np.zeros(len(t))
    for c in cols:
        ax.barh(range(len(t)), t[c], left=left, color=palette[c], label=c, height=0.7)
        left += t[c].to_numpy()
    ax.set_yticks(range(len(t)), [f"{r.replace('sd14_', '')}: {a}" for r, a in t.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of risky requests (SD 1.4, test split)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Which gate signal fired, by request type", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    p = out / "fig_gate_signals.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return [p.name]


def fig_models(ba: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib.pyplot as plt

    x = ba[(ba.threat == "characters") & ba.guardrail.isin(["D0", "V0", "D4", "D4+D2b", "G"])]
    if x.empty:
        return []
    models = [m for m in ("SD 1.4", "SDXL", "Playground v2.5", "PixArt-α") if m in set(x.model)]
    guards = [g for g in ("D0", "V0", "D4", "D4+D2b", "G") if g in set(x.guardrail)]
    w = 0.8 / len(guards)
    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(models), 3.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    shades = {"D0": "#d9d8d3", "V0": "#b5b4ae", "D4": "#919089", "D4+D2b": "#6c6b65", "G": G_BLUE}
    for k, g in enumerate(guards):
        v = [100 * x[(x.model == m) & (x.guardrail == g)]["mean_attacks"].mean() for m in models]
        bars = ax.bar(np.arange(len(models)) + (k - (len(guards) - 1) / 2) * w, v, w - 0.02, color=shades[g],
                      label=f"{g}{'†' if g in ORACLE else ''} {NAMES.get(g, '')}")
        for b, val in zip(bars, v):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(models)), models, fontsize=9)
    ax.set_ylabel("characters detected, mean over 4 attacks (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("Characters across model families (lower = better)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    p = out / "fig_models_characters.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return [p.name]


def fig_gate_flow(out: Path) -> list[str]:
    """How a request moves through the gate (no data; for the slides and the README)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.6, 4.6), facecolor=SURFACE)
    ax.set_xlim(-2, 116)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=SURFACE, ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9.5, color=INK, weight="bold")
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=7.8, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1, label="", color=INK2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.0, label, ha="center", fontsize=7.5, color=color)

    blue, green, orange, gray = "#2a78d6", "#1baf7a", "#eb6834", "#85847e"
    box(0, 17, 14, 12, "request", "what the user\ntyped (maybe a lure,\nmisspelling, padding)", gray)
    box(19, 26, 24, 18, "concept screen", "protected name after\nUnicode folding, else\nQwen3-8B over the closed\nlist of 50 characters +\n15 in-copyright artists", blue)
    box(19, 2, 24, 18, "memorization screen", "SD 1.x/2.x only: first-step\ntext-conditional noise\nmagnitude (Wen et al.)\nof the request and its\nsegments vs. tau (dev)", green)
    box(49, 26, 24, 18, "matched mitigation", "character: LLM rewrite +\nits name/keyword negative\nprompt (D4+D2b)\nartist: name as negative\nprompt (D2b-name)", blue)
    box(49, 2, 24, 18, "verified rewording", "mildest template whose own\nscore is below tau:\nD2a -> D3 -> D1\n(checked before rendering)", green)
    box(79, 14, 13, 16, "render", "nothing fired:\nthe request\nunchanged", gray)
    box(96, 13, 17, 18, "output check", "CLIP sees a protected\ncharacter the screens\nmissed? re-render\nonce with its\nmitigation", orange)
    arrow(14, 26, 19, 33)
    arrow(14, 20, 19, 13)
    arrow(43, 35, 49, 35, "fires", blue)
    arrow(43, 11, 49, 11, "fires", green)
    arrow(73, 33, 79, 26, "", blue)
    arrow(73, 12, 79, 18, "", green)
    arrow(92, 22, 96, 22)
    ax.text(55, 23.2, "no signal: no change, no refusal", ha="center", fontsize=8, color=gray, style="italic")
    ax.text(0, 44.5, "Demo 3 gate (G): detect the trigger, then apply the Demo 2 defense matched to it", fontsize=11,
            color=INK, ha="left")
    p = out / "fig_gate_flow.png"
    fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return [p.name]


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path(os.environ.get("GD_OUTPUT_ROOT", ROOT / "outputs")))
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "demo3")
    ap.add_argument("--labels", type=Path, default=ROOT / "results" / "demo3" / "onprompt_labels.csv")
    args = ap.parse_args()
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    runs = load_all(args.outputs)
    if not runs:
        raise SystemExit(f"no scored runs under {args.outputs}/demo2 or demo3")
    print("runs:", {k: sorted(v["defense_code"].unique()) for k, v in runs.items()})

    # LLM rewrite time per request (D4, and the gate when it detects a character): measured by run_rewriter.py
    rw_t = [pd.read_csv(f)["rewrite_seconds"] for f in (args.outputs / "rewrites").glob("demo3_*.csv")
            if "rewrite_seconds" in pd.read_csv(f, nrows=1)]
    rewrite_seconds = float(pd.concat(rw_t).mean()) if rw_t else np.nan
    f1, ba, rai, prot = family_tables(runs, rewrite_seconds)
    f1.round(4).to_csv(out / "f1.csv", index=False)
    rai.round(4).to_csv(out / "responsible_ai.csv", index=False)
    ba.round(4).to_csv(out / "before_after_all.csv", index=False)
    pick = pick_rows(ba)
    pick.round(4).to_csv(out / "before_after.csv", index=False)
    pd.concat({k: v for k, v in prot.items()}, names=["run"]).round(4).to_csv(out / "protection.csv")
    sub = subgroup_table(runs, ba)
    sub.round(4).to_csv(out / "subgroups.csv", index=False)
    gaps = subgroup_gaps(sub)
    gaps.round(4).to_csv(out / "subgroup_gaps.csv", index=False)
    if not gaps.empty:   # largest gap over the subgroupings of each threat/model, per guardrail
        mg = gaps.groupby(["threat", "model", "guardrail"])["gap"].max()
        rai["subgroup_gap"] = [mg.get((r.threat, r.model, r.guardrail), np.nan) for r in rai.itertuples()]
        rai.round(4).to_csv(out / "responsible_ai.csv", index=False)
    sig = gate_signals(runs)
    sig_sum = gate_signal_summary(sig)
    sig_sum.round(4).to_csv(out / "gate_signals.csv", index=False)
    (out / "gate_errors.md").write_text("# What still gets through the gate (G), and why\n\n" + gate_errors(runs))
    jv = judge_validation(runs, args.labels)
    jv.round(4).to_csv(out / "judge_validation.csv", index=False)
    mt = models_table(ba)
    mt.to_csv(out / "models.csv")
    figs = fig_gate_flow(out) + fig_f1(f1, out) + fig_tradeoff(ba, out) + fig_signals(sig, out) + fig_models(ba, out)

    # headline numbers
    h = {}
    for r in pick.to_dict("records"):
        h.setdefault(f"{r['threat']} / {r['model']}", {})[r["guardrail"]] = {
            k: (None if pd.isna(r[k]) else round(float(r[k]), 4)) for k in
            ("plain", "worst_attack", "mean_attacks", "risky_on_prompt", "benign_served", "benign_refused", "su_f1",
             "sec_per_request") if k in r}
    (out / "headline.json").write_text(json.dumps(h, indent=1, ensure_ascii=False))

    # README
    def view(df, cols, pcts):
        v = df[cols].copy()
        for c in pcts:
            if c in v:
                v[c] = v[c].map(pct)
        return v

    rep = ["# Demo 3 results: detect-then-defend gate (G) vs. the Demo 2 baselines", "",
           "Same test items, attacks and settings as Demo 2, seed 0 (the Demo 2 grid was re-rendered with one seed under the "
           "class GPU cap). † = the experiment handed this baseline the protected "
           "target (He et al.'s setup); a deployed system has to detect it, which is what G does. Protection numbers are "
           "failure rates (lower = better); F1 and served rates are higher = better.", "",
           "## Before vs. after", "",
           md(view(pick, ["threat", "model", "role", "guardrail", "plain", "worst_attack", "mean_attacks", "risky_on_prompt",
                          "benign_served", "benign_refused", "su_f1", "f1_ci", "sec_per_request"],
                   ["plain", "worst_attack", "mean_attacks", "risky_on_prompt", "benign_served", "benign_refused"])), "",
           "## Responsible-AI metrics", "",
           "| metric | what it measures |", "|---|---|",
           "| safe-utility F1 | harmonic mean of (a) risky requests protected AND still on-prompt AND not refused and (b) benign requests served (not refused, on-prompt); blocking everything or drawing something arbitrary scores 0 |",
           "| benign refused | over-blocking: benign requests refused (ordinary captions, near-miss generic requests, public-domain styles) |",
           "| benign utility lost | served rate on benign requests lost relative to no guardrail (percentage points) |",
           "| benign intervention | share of benign requests the guardrail changed at all |",
           "| attack lift | worst attack minus plain request (robustness to a non-cooperative user) |",
           "| subgroup gap | max - min failure rate across character origin / rights holder, artist group, memorization type ([subgroup_gaps.csv](subgroup_gaps.csv)) |",
           "| drawn unasked | benign near-miss requests on which the judge finds a protected character |",
           "| runtime | seconds per request, including the gate's screening and any second render |", "",
           md(view(rai.drop(columns=["oracle_target"]),
                   [c for c in ["threat", "model", "guardrail", "name", "safe_utility_f1", "benign_refused", "benign_utility_lost",
                                "benign_intervention", "drawn_unasked", "attack_lift", "worst_attack", "subgroup_gap",
                                "sec_per_request", "runtime_vs_D0"] if c in rai],
                   ["benign_refused", "benign_utility_lost", "benign_intervention", "drawn_unasked", "attack_lift",
                    "subgroup_gap"])), ""]
    if not gaps.empty:
        gv = gaps[gaps.guardrail.isin(["D0", "V0", "D2b-kw", "D2b-name", "D4", "D4+D2b", "D1", "D3", "G"])].copy()
        for c in ("max", "min", "gap"):
            gv[c] = gv[c].map(pct)
        rep += ["## Subgroups: is the protection equal?", "", md(gv), ""]
    if not sig_sum.empty:
        sv = sig_sum.copy()
        sv["share"] = sv["share"].map(pct)
        rep += ["## Why it works: which signal fired", "", md(sv), "", "What still gets through: [gate_errors.md](gate_errors.md)", ""]
    if not jv.empty:
        rep += ["## On-prompt judge validation", "", md(jv), ""]
    if len(mt):
        rep += ["## Across model families", "", mt.to_markdown(), ""]
    rep += ["## Figures", ""] + [f"![{f}]({f})" for f in figs]
    (out / "README.md").write_text("\n".join(rep) + "\n")
    print(f"wrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    main()
