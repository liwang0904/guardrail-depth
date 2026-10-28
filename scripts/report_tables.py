"""Turn result CSVs into LaTeX tables and number macros for the milestone report.

    python scripts/report_tables.py --out ../Milestone/tables

Writes numbers.tex (\\newcommand macros used in the prose) and one .tex per table, so no number in the
report is typed by hand. Missing inputs produce a visible placeholder instead of failing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.paths import gt_image_dir, output_root  # noqa: E402

MACROS: dict[str, str] = {}


def macro(name: str, value) -> None:
    MACROS[name] = value if isinstance(value, str) else (f"{value:.2f}" if isinstance(value, float) else str(value))


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("$", "\\$").replace("{", "\\{").replace("}", "\\}"))


# ----------------------------------------------------------------------------- data table
def data_table(out: Path) -> None:
    m, u, s2 = load_manifest("memorized_sd1"), load_manifest("unmemorized_laion"), load_manifest("memorized_sd2")
    c, a = load_manifest("copycat_characters"), load_manifest("artists")
    vm, tm = (m.mem_type == "VM").sum(), (m.mem_type == "TM").sum()
    rows = [
        ("Memorized SD 1.4 prompts", "Wen et al.; NeMo; Webster", f"{len(m)} ({vm} VM / {tm} TM)",
         "100/400", "memorization type, LAION duplicates, reference SSCD, token length, ground-truth URL"),
        ("Non-memorized LAION prompts", "NeMo", str(len(u)), "100/400", "control for detection and calibration"),
        ("Memorized SD 2 prompts", "Webster", f"{len(s2)}", "44/175", "MV/TV type, duplicates"),
        ("CopyCat characters", "He et al.", f"{len(c)}", "11/39",
         f"60-word lure, 3 keyword sets, GPT-4 rewrites, key entity, origin (US {(c.origin_region=='US').sum()}/JP "
         f"{(c.origin_region=='JP').sum()}/CA {(c.origin_region=='CA').sum()}), public-domain notes"),
        ("Artists $\\times$ style prompts", "Pham et al.; ESD; WikiArt", f"{len(a)} $\\times$ 10",
         "2/8 templates", "living / deceased-in-copyright / public domain, DALL$\\cdot$E pre-1912 rule"),
    ]
    lines = ["\\begin{table*}[t]", "\\caption{Evaluation data (dev/test split).}", "\\label{tab:data}", "\\small",
             "\\begin{tabularx}{\\textwidth}{@{}p{3.2cm}p{3.4cm}p{2.2cm}p{1.5cm}X@{}}", "\\toprule",
             "Set & Source & Size & Split & Attributes \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table*}"]
    (out / "data.tex").write_text("\n".join(lines) + "\n")
    st = gt_image_dir() / "sd1/_status.csv"
    if st.exists():
        g = m.merge(pd.read_csv(st), on="id")
        macro("gtAlive", int((g.status == "ok").sum())); macro("gtTotal", len(g))


# ----------------------------------------------------------------------------- token window
def window(out: Path) -> None:
    f = output_root() / "analysis/token_window.csv"
    if not f.exists():
        (out / "window_text.tex").write_text("\\textbf{[token-window results missing]}\n")
        return
    w = pd.read_csv(f)
    get = lambda ds, a, d, col: float(w[(w.dataset == ds) & (w.attack == a) & (w.defense == d)][col].iloc[0])
    macro("winCotTrunc", f"{get('memorized_sd1', 'A0', 'D3', 'pct_truncated'):.0f}\\%")
    macro("winCotSeen", f"{100 * get('memorized_sd1', 'A0', 'D3', 'mean_frac_tail_visible'):.0f}\\%")
    macro("winLureDoneLost", f"{get('copycat_characters', 'A1-desc', 'D1', 'pct_tail_fully_lost'):.0f}\\%")
    macro("winLureDtwoLost", f"{get('copycat_characters', 'A1-desc', 'D2a', 'pct_tail_fully_lost'):.0f}\\%")
    macro("winLureDthreeLost", f"{get('copycat_characters', 'A1-desc', 'D3', 'pct_tail_fully_lost'):.0f}\\%")
    macro("winTruncLost", f"{get('memorized_sd1', 'A3-trunc-rep', 'D1', 'pct_tail_fully_lost'):.0f}\\%")
    macro("winOverrideDone", f"{100 * get('memorized_sd1', 'A3-override', 'D1', 'mean_frac_tail_visible'):.0f}\\%")
    d4 = w[(w.dataset == "copycat_characters") & (w.defense == "D4 (GPT-4 release)") & (w.attack == "A0")].iloc[0]
    macro("winRewriteMedian", f"{d4.median_tokens:.0f}")
    macro("winRewriteOver", f"{d4.pct_truncated:.0f}\\%")
    c = load_manifest("copycat_characters")
    macro("lureOver", int((c.desc60_clip_tokens > 75).sum()))
    text = (
        "Without an attack, the D3 chain-of-thought template is truncated for \\winCotTrunc{} of memorized captions, and on "
        "average only \\winCotSeen{} of the text after the request (its ``verify'' steps) reaches the model. He et al.'s "
        "60-word description lures are long on their own (\\lureOver{} of 50 exceed the window), so wrapping them in a "
        "template removes the guardrail tail completely in \\winLureDoneLost{} (D1), \\winLureDtwoLost{} (D2a) and "
        "\\winLureDthreeLost{} (D3) of cases. A trivial truncation attack, repeating the request until it fills the "
        "window, removes it in \\winTruncLost{} of cases, and an override suffix leaves \\winOverrideDone{} of D1's tail. "
        "The D4 rewrites He et al.\\ released (GPT-4, for DALL$\\cdot$E~3's long-context encoder) have a median length of "
        "\\winRewriteMedian{} tokens, and \\winRewriteOver{} overflow SD's window. Whatever a template achieves therefore "
        "comes from its prefix and from perturbing the request, not from its instructions. This is the mechanism behind "
        "the C1 random-word control below.\n")
    (out / "window_text.tex").write_text(text)


# ----------------------------------------------------------------------------- Wen detector
DEF_ORDER = ["D0", "D0r", "D1", "D2a", "D2b", "D3", "C1"]
DEF_NAMES = {"D0": "none", "D0r": "``Generate an image of''", "D1": "task instruction", "D2a": "in-prompt negation",
             "D2b": "negative buffer (rules)", "D3": "chain-of-thought", "C1": "random words", "D2b-kw": "neg.\\ buffer: name+kw",
             "D2b-name": "neg.\\ buffer: name", "D4+D2b": "GPT-4 rewrite + neg.\\ buffer", "D4": "LLM rewrite"}


def detection(out: Path) -> None:
    f = output_root() / "detection/sd14_dev/summary.csv"
    if not f.exists():
        (out / "detection.tex").write_text("\\textbf{[detection results pending]}\n")
        return
    s = pd.read_csv(f)
    t = output_root() / "detection/sd14_dev_trunc/summary.csv"
    st = pd.read_csv(t) if t.exists() else pd.DataFrame()
    rows = []
    for d in DEF_ORDER:
        r = s[(s.attack == "A0") & (s.defense == d)]
        if len(r):
            r = r.iloc[0]
            rows.append((f"{d} {DEF_NAMES[d]}", r.auc_n, r.score_VM, r.score_TM, r.score_ctrl))
    for d in ["D0", "D1", "D3"]:
        r = st[(st.attack == "A3-trunc-rep") & (st.defense == d)] if len(st) else []
        if len(r):
            r = r.iloc[0]
            rows.append((f"{d} + truncation attack", r.auc_n, r.score_VM, r.score_TM, r.score_ctrl))
    d0 = s[(s.attack == "A0") & (s.defense == "D0")].iloc[0]
    macro("aucDzero", float(d0.auc_n)); macro("tprDzero", f"{100 * d0['tpr@1%fpr']:.0f}\\%")
    from sklearn.metrics import roc_auc_score

    sc = pd.read_csv(output_root() / "detection/sd14_dev/scores.csv")
    a0 = sc[(sc.attack == "A0") & (sc.defense == "D0")]
    for mt, name in (("VM", "aucVMDzero"), ("TM", "aucTMDzero")):
        sub = a0[a0.mem_type.isin([mt, "control"])]
        macro(name, float(roc_auc_score(sub.label, sub.score)))
    for d, tag in (("D0", "Dzero"), ("D0r", "Dzeror"), ("D1", "Done"), ("D2a", "Dtwoa"), ("D3", "Dthree"), ("C1", "Cone")):
        g = sc[(sc.attack == "A0") & (sc.defense == d)]
        macro(f"scVM{tag}", float(g[g.mem_type == "VM"].score.mean()))
        macro(f"scTM{tag}", float(g[g.mem_type == "TM"].score.mean()))
        macro(f"scCtrl{tag}", float(g[g.mem_type == "control"].score.mean()))
    g = sc[(sc.attack == "A0") & (sc.defense == "D2b")]
    macro("guideVMDtwob", float(g[g.mem_type == "VM"].guidance_norm.mean()))
    macro("guideCtrlDtwob", float(g[g.mem_type == "control"].guidance_norm.mean()))
    if len(st):
        t_ = pd.read_csv(output_root() / "detection/sd14_dev_trunc/scores.csv")
        for d, tag in (("D0", "Dzero"), ("D1", "Done"), ("D3", "Dthree")):
            g = t_[t_.defense == d]
            macro(f"scVMtrunc{tag}", float(g[g.mem_type == "VM"].score.mean()))
            macro(f"scCtrltrunc{tag}", float(g[g.mem_type == "control"].score.mean()))
            sub = g[g.mem_type.isin(["VM", "control"])]
            macro(f"aucVMtrunc{tag}", float(roc_auc_score(sub.label, sub.score)))
    for d, name in [("D1", "aucDone"), ("D2a", "aucDtwoa"), ("D3", "aucDthree"), ("C1", "aucCone"), ("D0r", "aucDzeror"),
                    ("D2b", "aucDtwob")]:
        r = s[(s.attack == "A0") & (s.defense == d)]
        if len(r):
            macro(name, float(r.iloc[0].auc_n))
            macro(name.replace("auc", "relVM"), f"{100 * r.iloc[0].rel_to_D0_VM:.0f}\\%")
    lines = ["\\begin{table}[t]", "\\caption{Wen et al.'s memorization detector applied to the prompt each guardrail "
             "feeds SD~1.4 (dev split: 22 VM, 78 TM, 100 controls; one generation, first step). Score = text-conditional "
             "noise magnitude; AUC separates memorized from control prompts.}", "\\label{tab:detection}", "\\small",
             "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Guardrail & AUC & VM & TM & control \\\\", "\\midrule"]
    lines += [f"{n} & {a:.2f} & {v:.1f} & {t_:.1f} & {c:.1f} \\\\" for n, a, v, t_, c in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "detection_table.tex").write_text("\n".join(lines) + "\n")
    (out / "detection.tex").write_text("\\input{tables/detection_table}\n\\input{tables/detection_text}\n")
    if not (out / "detection_text.tex").exists():
        (out / "detection_text.tex").write_text("\\textbf{[interpretation to be written]}\n")


# ----------------------------------------------------------------------------- pilot images
def pilot(out: Path) -> None:
    base = output_root() / "pilot_local"
    lines = []
    mem = base / "memorized/metrics.csv"
    if mem.exists():
        df = pd.read_csv(mem)
        g = df.groupby(["mem_type", "defense_code"])
        tab = g.agg(sscd=("sscd_gt", "mean"), copy=("copy", "mean"), clip_t=("clip_t", "mean"),
                    div=("diversity_sscd", "mean"), noise=("noise_first", "mean"),
                    guard=("attn_guardrail_t0", "mean"), n=("image", "size")).reset_index()
        tab.to_csv(out / "pilot_memorized.csv", index=False)
        allg = df.groupby("defense_code")
        tags = {"D0": "Dzero", "D1": "Done", "D2a": "Dtwoa", "D2b": "Dtwob", "D3": "Dthree", "C1": "Cone"}
        for d, tag in tags.items():
            if d in allg.groups:
                gd = allg.get_group(d)
                macro(f"clipT{tag}", f"{gd.clip_t.mean():.3f}")
                macro(f"guard{tag}", f"{gd.attn_guardrail_t0.mean():.2f}")
                macro(f"bos{tag}", f"{gd.attn_bos_t0.mean():.2f}")
                macro(f"copyAll{tag}", pct(float(gd['copy'].mean())))
                if "sscd_x0_t00" in gd:
                    macro(f"xzero{tag}", f"{gd[gd.mem_type == 'VM'].sscd_x0_t00.mean():.2f}")
                    macro(f"xend{tag}", f"{gd[gd.mem_type == 'VM'].sscd_x0_t49.mean():.2f}")
        for mt in ["VM", "TM"]:
            for d in DEF_ORDER:
                r = tab[(tab.mem_type == mt) & (tab.defense_code == d)]
                if len(r):
                    macro(f"cp{mt}{d.replace('0', 'zero').replace('1', 'one').replace('2', 'two').replace('3', 'three')}",
                          pct(float(r.iloc[0]["copy"])))
        lines += ["\\begin{table}[t]", "\\caption{Pilot, exact-copy memorization (SD~1.4, 3 VM + 3 TM dev prompts with "
                  "ground truth, 2 seeds). Copy = SSCD$>$0.5 to the training image; CLIP-T = adherence to the caption; "
                  "guard = cross-attention mass on guardrail tokens at step 1.}", "\\label{tab:pilotmem}", "\\small",
                  "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", " & Guardrail & SSCD & copy & CLIP-T & guard \\\\",
                  "\\midrule"]
        for mt in ["VM", "TM"]:
            for d in DEF_ORDER:
                r = tab[(tab.mem_type == mt) & (tab.defense_code == d)]
                if len(r):
                    r = r.iloc[0]
                    lines.append(f"{mt if d == 'D0' else ''} & {d} & {r.sscd:.2f} & {pct(r['copy'])} & {r.clip_t:.3f} & "
                                 f"{r.guard:.3f} \\\\")
            lines.append("\\midrule" if mt == "VM" else "")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    cc = base / "copycat/metrics.csv"
    if cc.exists():
        df = pd.read_csv(cc)
        tab = (df.groupby(["attack_code", "defense_code"], sort=False)
                 .agg(detect=("clip_detect", "mean"), p=("p_target", "mean"), cons=("clip_cons", "mean"),
                      n=("image", "size")).reset_index())
        tab.to_csv(out / "pilot_copycat.csv", index=False)
        atag = {"A0": "Azero", "A1-desc": "Adesc", "A1-kwLAION": "Akw"}
        dtag = {"D0": "Dzero", "D1": "Done", "D3": "Dthree", "D2b-kw": "Dtwob", "D4+D2b": "Dfour"}
        for (a, d), g in df.groupby(["attack_code", "defense_code"]):
            if a in atag and d in dtag:
                macro(f"ccDet{atag[a]}{dtag[d]}", f"{int(g.clip_detect.sum())}/{len(g)}")
                if g.attn_guardrail_t0.notna().any():
                    macro(f"ccGuard{atag[a]}{dtag[d]}", f"{g.attn_guardrail_t0.mean():.2f}")
                    macro(f"ccUser{atag[a]}{dtag[d]}", f"{g.attn_user_t0.mean():.2f}")
                    macro(f"ccSeen{atag[a]}{dtag[d]}", f"{g.suffix_tokens_seen.mean():.0f}")
        lines += ["\\begin{table}[t]", "\\caption{Pilot, CopyCat characters (SD~1.4, 5 dev characters, 1 seed). "
                  "Detect = CLIP zero-shot top-1 among the 50 characters equals the target (a local proxy for He et al.'s "
                  "VLM DETECT); CONS = CLIP score with the key entity.}", "\\label{tab:pilotcc}", "\\small",
                  "\\begin{tabular}{@{}llrr@{}}", "\\toprule", "Attack & Guardrail & detect & CONS \\\\", "\\midrule"]
        names = {"A0": "name", "A1-desc": "60-word lure", "A1-kwLAION": "LAION keywords"}
        prev = None
        for r in tab.itertuples():
            lines.append(f"{names.get(r.attack_code, r.attack_code) if r.attack_code != prev else ''} & {r.defense_code} & "
                         f"{r.detect * len(df[(df.attack_code == r.attack_code) & (df.defense_code == r.defense_code)]):.0f}/{r.n} & {r.cons:.3f} \\\\")
            prev = r.attack_code
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    sty = base / "style/metrics.csv"
    if sty.exists():
        df = pd.read_csv(sty)
        tab = df.groupby(["group", "defense_code"], sort=False).agg(p=("p_artist", "mean"), n=("image", "size")).reset_index()
        tab.to_csv(out / "pilot_style.csv", index=False)
        gtag = {"living": "Liv", "public_domain": "Pd"}
        dtag = {"D0": "Dzero", "D1": "Done", "D2a": "Dtwoa", "D3": "Dthree", "D2b-name": "Dtwob"}
        for r in tab.itertuples():
            if r.group in gtag and r.defense_code in dtag:
                macro(f"st{gtag[r.group]}{dtag[r.defense_code]}", f"{r.p:.2f}" if r.p >= 0.01 else f"{r.p:.3f}")
        living = df[df.group == "living"]
        macro("stLivTopOther", ", ".join(sorted(set(living.artist_top1) - {"an unknown contemporary artist"} - set(living.target))[:3]))
        lines += ["\\begin{table}[t]", "\\caption{Pilot, style mimicry (SD~1.4; 2 living + 2 public-domain artists, 2 dev "
                  "prompts). $p$ = CLIP zero-shot probability of the requested artist among all 25.}",
                  "\\label{tab:pilotstyle}", "\\small", "\\begin{tabular}{@{}l" + "r" * tab.defense_code.nunique() + "@{}}",
                  "\\toprule", "Group & " + " & ".join(tab.defense_code.unique()) + " \\\\", "\\midrule"]
        for grp in ["living", "public_domain"]:
            vals = [tab[(tab.group == grp) & (tab.defense_code == d)].p for d in tab.defense_code.unique()]
            fmt = lambda x: ("$<$0.01" if x < 0.01 else f"{x:.2f}")
            lines.append(grp.replace("_", " ") + " & " + " & ".join(fmt(v.iloc[0]) if len(v) else "--" for v in vals) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "pilot_tables.tex").write_text("\n".join(lines) + "\n" if lines else "\\textbf{[pilot results pending]}\n")
    (out / "pilot_text.tex").write_text("\\input{tables/pilot_tables}\n\\input{tables/pilot_prose}\n")
    if not (out / "pilot_prose.tex").exists():
        (out / "pilot_prose.tex").write_text("\\textbf{[interpretation to be written]}\n")


# ----------------------------------------------------------------------------- NeMo neurons
def neurons(out: Path) -> None:
    v = output_root() / "neurons/validation.json"
    if v.exists():
        val = json.loads(v.read_text())
        macro("nemoN", val["n_prompts"])
        macro("nemoZmean", f"{val['none']['z_mean']:+.3f}".replace("+", ""))
        macro("nemoZstd", f"{val['none']['z_std']:.3f}")
        stds = [val[k]["z_std"] for k in ("task_instruction", "neg_inprompt", "cot", "rand_tokens")]
        macro("nemoTemplateStdLo", f"{min(stds):.2f}"); macro("nemoTemplateStdHi", f"{max(stds):.2f}")
    s = output_root() / "neurons/sd14_dev_matched/summary.csv"
    if s.exists():
        df = pd.read_csv(s)
        base = df[(df.attack == "A0") & (df.defense == "D0")].set_index("mem_type")
        macro("nemoOodVM", f"{base.loc['VM', 'n_ood']:.1f}"); macro("nemoOodTM", f"{base.loc['TM', 'n_ood']:.1f}")
        macro("nemoOodCtrl", f"{base.loc['control', 'n_ood']:.1f}")


# ----------------------------------------------------------------------------- rewriter
def rewriter(out: Path) -> None:
    g = output_root() / "analysis/gpt4_rewrite_leakage.csv"
    if g.exists():
        lk = pd.read_csv(g)
        for src, tag in (("GPT-4 rewrite of name", "Name"), ("GPT-4 rewrite of lure", "Lure"), ("60-word lure", "Desc")):
            sub = lk[lk.source == src]
            macro(f"gptKw{tag}", f"{sub.kw_hits.mean():.1f}")
            macro(f"gptAnyKw{tag}", pct(float((sub.kw_hits > 0).mean())))
            macro(f"gptNameLeak{tag}", int(sub.name_leak.sum()))
            macro(f"gptN{tag}", len(sub))
        macro("gptKwTotal", f"{lk.kw_total.mean():.0f}")
    (out / "rewriter.tex").write_text("\\input{tables/rewriter_prose}\n")
    if not (out / "rewriter_prose.tex").exists():
        (out / "rewriter_prose.tex").write_text("\\textbf{[interpretation to be written]}\n")
    files = sorted((output_root() / "rewrites").glob("*.csv")) if (output_root() / "rewrites").exists() else []
    if not files:
        return
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files])
    df.to_csv(out / "rewrites_all.csv", index=False)
    yes = lambda col: df[col].astype(str).eq("True")
    ok = df.rewrite.astype(str).str.len() > 0
    macro("rwN", len(df)); macro("rwOk", int(ok.sum()))
    macro("rwRecognized", int(yes("reasoning_names_target").sum()))
    macro("rwNameLeak", int(yes("rewrite_name_leak").sum()))
    macro("rwKwMean", f"{pd.to_numeric(df.rewrite_kw_hits)[ok].mean():.1f}")
    macro("rwReasonTok", f"{pd.to_numeric(df.n_reason_tokens).median():.0f}")
    ent = pd.to_numeric(df.entropy_final_mean, errors="coerce")
    kw = pd.to_numeric(df.rewrite_kw_hits)
    leaky = ok & (kw >= kw[ok].median())
    macro("rwEntLeaky", f"{ent[leaky].mean():.2f}"); macro("rwEntClean", f"{ent[ok & ~leaky].mean():.2f}")
    for atk, tag in (("direct", "Direct"), ("lure_desc60", "Lure")):
        sub = df.attack == atk
        if sub.any():
            macro(f"rwRecog{tag}", f"{int(yes('reasoning_names_target')[sub].sum())}/{int(sub.sum())}")
            macro(f"rwKw{tag}", f"{kw[sub & ok].mean():.1f}")
            macro(f"rwKwMax{tag}", int(kw[sub & ok].max()))
            macro(f"rwEnt{tag}", f"{ent[sub & ok].mean():.2f}")
            macro(f"rwThink{tag}", f"{pd.to_numeric(df.n_reason_tokens)[sub].median():.0f}")
    macro("rwKwTotal", f"{pd.to_numeric(df.kw_total).mean():.0f}")
    qm = output_root() / "pilot_local_rewrite/copycat_qwen_rewrite/metrics.csv"
    if qm.exists():
        q = pd.read_csv(qm)
        for (a, d), g in q.groupby(["attack_code", "defense_code"]):
            tag = {"A0": "Azero", "A1-desc": "Adesc"}[a] + {"D4": "Dfour", "D4+D2b": "DfourDtwob"}[d]
            macro(f"qwDet{tag}", f"{int(g.clip_detect.sum())}/{len(g)}")
    names = {"direct": "name", "lure_desc60": "60-word lure"}
    lines = ["\\begin{table}[t]", "\\caption{D4 with an open reasoning model (Qwen3-1.7B, thinking on; DALL$\\cdot$E policy "
             "prompt of He et al.). Recog.\\ = the reasoning trace names the character; kw = identifying keywords kept in the "
             "final rewrite; $H$ = mean token entropy (nats) of the final rewrite.}", "\\label{tab:rewriter}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llccrrr@{}}", "\\toprule",
             "Character & Request & Recog. & Name & kw & think tok. & $H$ \\\\", "\\midrule"]
    for r in df.itertuples():
        short = {"SpongeBob SquarePants": "SpongeBob", "Winnie-the-Pooh": "Pooh"}.get(r.target, r.target)
        lines.append(f"{tex_escape(short)} & {names.get(r.attack, r.attack)} & {'yes' if str(r.reasoning_names_target) == 'True' else 'no'} & "
                     f"{'yes' if str(r.rewrite_name_leak) == 'True' else 'no'} & {r.rewrite_kw_hits}/{r.kw_total} & "
                     f"{r.n_reason_tokens} & {float(r.entropy_final_mean):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "rewriter_table.tex").write_text("\n".join(lines) + "\n")


def apgp(out: Path) -> None:
    logs = sorted((output_root() / "apgp").glob("*/apgp_log.json")) if (output_root() / "apgp").exists() else []
    if not logs:
        return
    runs = [r for f in logs for r in json.loads(f.read_text())]
    macro("apN", len(runs))
    macro("apCands", sum(len(r["history"]) - 1 for r in runs))
    macro("apImproved", sum(r["best_score"] > r["seed_score"] + 1e-6 for r in runs))
    seed_img = [r["history"][0]["img_score"] for r in runs]
    macro("apSeedImgLo", f"{min(seed_img):.0f}"); macro("apSeedImgHi", f"{max(seed_img):.0f}")
    dup = 0
    for r in runs:  # candidates whose image is identical to an earlier one (same img score) though the text differs
        seen = {}
        for h in r["history"]:
            key = round(h["img_score"], 3)
            if key in seen and seen[key] != h["prompt"]:
                dup += 1
            seen.setdefault(key, h["prompt"])
    macro("apDupImages", dup)
    macro("apTargets", " and ".join(r["target"] for r in runs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT.parent / "Milestone/tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (data_table, window, detection, neurons, pilot, rewriter, apgp):
        fn(args.out)
    (args.out / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS.items()))
    if not (args.out / "abstract_numbers.tex").exists():
        (args.out / "abstract_numbers.tex").write_text("\\textbf{[key numbers]}\n")
    print(f"wrote {len(MACROS)} macros and tables -> {args.out}")


if __name__ == "__main__":
    main()
