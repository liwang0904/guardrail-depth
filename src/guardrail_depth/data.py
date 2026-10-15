"""Evaluation datasets: building normalized manifests from the raw reference files, and loading them.

Manifests (data/manifests/*.csv) are small and committed, so experiments never depend
on a re-download. Each row carries the attributes the analysis conditions on
(memorization type, duplicate count, prompt length in CLIP tokens, IP origin region,
artist copyright status, ...) and a fixed dev/test split (seed 347).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .paths import MANIFEST_DIR, raw_dir

SPLIT_SEED = 347


# ----------------------------------------------------------------------------- loaders
def load_manifest(name: str) -> pd.DataFrame:
    path = MANIFEST_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/prepare_data.py` first")
    df = pd.read_csv(path)
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].fillna("")  # numeric columns keep NaN for "unknown"
    return df


def items(name: str, split: str | None = None, ids: list | None = None) -> list[dict]:
    """Rows of a manifest as dicts (the `meta` that attacks/defenses consume)."""
    df = load_manifest(name)
    if split:
        df = df[df["split"] == split]
    if ids is not None:
        df = df[df["id"].isin(ids)]
    return df.to_dict("records")


# ----------------------------------------------------------------------------- split helper
def assign_splits(df: pd.DataFrame, dev_frac: float, stratify: str | None = None) -> pd.Series:
    rng = random.Random(SPLIT_SEED)
    split = pd.Series("test", index=df.index)
    groups = df.groupby(stratify).groups.values() if stratify else [df.index]
    for idx in groups:
        idx = sorted(idx)
        rng.shuffle(idx)
        n_dev = max(1, round(len(idx) * dev_frac))
        split.loc[idx[:n_dev]] = "dev"
    return split


# ----------------------------------------------------------------------------- builders
def _clip_len(tokenizer, texts) -> list[int]:
    return [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]


def build_memorized_sd1(tokenizer) -> pd.DataFrame:
    """500 prompts memorized by SD v1.4 (Webster 2023 -> Wen et al. ICLR'24 -> NeMo NeurIPS'24)."""
    raw = raw_dir()
    nemo = pd.read_csv(raw / "nemo/prompts/memorized_laion_prompts.csv", sep=";", index_col=0)
    wen = [json.loads(line) for line in open(raw / "wen/examples/sdv1_500_memorized.jsonl")]
    assert len(nemo) == len(wen) == 500, (len(nemo), len(wen))
    assert all(n == w["caption"] for n, w in zip(nemo["Caption"], wen)), "Wen/NeMo prompt order differs"
    gt = raw / "webster/groundtruth_parquets"
    bb = pd.read_parquet(gt / "sdv1_bb_edge_groundtruth.parquet").drop_duplicates("caption").set_index("caption")
    wb = pd.read_parquet(gt / "sdv1_wb_groundtruth.parquet").drop_duplicates("caption").set_index("caption")
    web_type = {**wb["overfit_type"].to_dict(), **bb["overfit_type"].to_dict()}  # black-box labels win
    df = pd.DataFrame({
        "id": range(500),
        "prompt": nemo["Caption"].values,
        "mem_type": nemo["type"].values,                       # VM = verbatim, TM = template (NeMo)
        "webster_type": [web_type.get(c, "") for c in nemo["Caption"]],             # MV/RV/TV/N (Webster)
        "numdups": [int(bb["numdups"].get(c, -1)) for c in nemo["Caption"]],       # LAION duplicate count
        "ref_mean_sscd": nemo["mean_sscd"].round(4).values,    # SD1.4 generations vs. original (NeMo)
        "ref_max_sscd": nemo["max_sscd"].round(4).values,
        "laion_index": nemo["Index"].values,
        "url": nemo["URL"].values,
    })
    df["n_clip_tokens"] = _clip_len(tokenizer, df["prompt"])
    df["split"] = assign_splits(df, 0.2, "mem_type")
    return df


def build_unmemorized(tokenizer) -> pd.DataFrame:
    """500 non-memorized LAION prompts (NeMo control set; same caption distribution)."""
    raw = raw_dir()
    un = pd.read_csv(raw / "nemo/prompts/unmemorized_laion_prompts.csv", sep=";", index_col=0)
    df = pd.DataFrame({"id": range(len(un)), "prompt": un["Caption"].values,
                       "laion_index": un["Index"].values, "url": un["URL"].values})
    df["n_clip_tokens"] = _clip_len(tokenizer, df["prompt"])
    df["split"] = assign_splits(df, 0.2)
    return df


def build_memorized_sd2(tokenizer) -> pd.DataFrame:
    """219 prompts memorized by SD v2 (Webster 2023; 4 MV + 215 TV)."""
    web = pd.read_parquet(raw_dir() / "webster/groundtruth_parquets/sdv2_bb_edge_groundtruth.parquet")
    web = web[web["overfit_type"].isin(["MV", "RV", "TV"])].reset_index(drop=True)
    df = pd.DataFrame({"id": range(len(web)), "prompt": web["caption"].values,
                       "webster_type": web["overfit_type"].values, "numdups": web["numdups"].values,
                       "laion_index": web["index"].values, "url": web["url"].values})
    df["mem_type"] = df["webster_type"].map({"MV": "VM", "RV": "VM", "TV": "TM"})
    df["n_clip_tokens"] = _clip_len(tokenizer, df["prompt"])
    df["split"] = assign_splits(df, 0.2, "mem_type")
    return df


# Origin region and rights holder for the 50 CopyCat characters (best effort; He et al.
# App. B lists "18 studios and subsidiaries"). pd_note flags characters whose earliest
# version is public domain in the US while the studio redesign is not.
CHARACTER_ORIGIN = {
    "Ariel": ("US", "Disney", ""), "Astro Boy": ("JP", "Tezuka Productions", ""),
    "Batman": ("US", "DC Comics (Warner Bros.)", ""), "Black Panther": ("US", "Marvel (Disney)", ""),
    "Bulbasaur": ("JP", "The Pokemon Company", ""), "Buzz Lightyear": ("US", "Pixar (Disney)", ""),
    "Captain America": ("US", "Marvel (Disney)", ""), "Chun-Li": ("JP", "Capcom", ""),
    "Cinderella": ("US", "Disney", "folk tale is public domain; Disney design is not"),
    "Cuphead": ("CA", "Studio MDHR", ""), "Donald Duck": ("US", "Disney", ""),
    "Doraemon": ("JP", "Fujiko Pro", ""), "Elsa": ("US", "Disney", ""), "Goofy": ("US", "Disney", ""),
    "Groot": ("US", "Marvel (Disney)", ""), "Hulk": ("US", "Marvel (Disney)", ""),
    "Iron Man": ("US", "Marvel (Disney)", ""), "Judy Hopps": ("US", "Disney", ""),
    "Kirby": ("JP", "Nintendo / HAL Laboratory", ""), "Kung Fu Panda": ("US", "DreamWorks Animation", ""),
    "Lightning McQueen": ("US", "Pixar (Disney)", ""), "Link": ("JP", "Nintendo", ""),
    "Maleficent": ("US", "Disney", ""), "Mario": ("JP", "Nintendo", ""),
    "Mickey Mouse": ("US", "Disney", "1928 Steamboat Willie version public domain in US since 2024"),
    "Mike Wazowski": ("US", "Pixar (Disney)", ""), "Monkey D. Luffy": ("JP", "Shueisha / Toei Animation", ""),
    "Mr. Incredible": ("US", "Pixar (Disney)", ""), "Naruto": ("JP", "Shueisha / Pierrot", ""),
    "Nemo": ("US", "Pixar (Disney)", ""), "Olaf": ("US", "Disney", ""), "Pac-Man": ("JP", "Bandai Namco", ""),
    "Peter Pan": ("US", "Disney", "Barrie's 1904 character is public domain in US; Disney design is not"),
    "Piglet": ("US", "Disney", "Milne 1926 book version public domain in US since 2022"),
    "Pikachu": ("JP", "The Pokemon Company", ""), "Princess Jasmine": ("US", "Disney", ""),
    "Puss in boots": ("US", "DreamWorks Animation", "folk tale is public domain; DreamWorks design is not"),
    "Rapunzel": ("US", "Disney", "folk tale is public domain; Disney design is not"),
    "Snow White": ("US", "Disney", "folk tale is public domain; Disney design is not"),
    "Sonic The Hedgehog": ("JP", "Sega", ""), "Spider-Man": ("US", "Marvel (Disney)", ""),
    "SpongeBob SquarePants": ("US", "Nickelodeon (Paramount)", ""), "Squirtle": ("JP", "The Pokemon Company", ""),
    "Thanos": ("US", "Marvel (Disney)", ""), "Thor": ("US", "Marvel (Disney)", "Norse myth is public domain"),
    "Tinker Bell": ("US", "Disney", "Barrie's 1904 character is public domain in US; Disney design is not"),
    "Wall-E": ("US", "Pixar (Disney)", ""),
    "Winnie-the-Pooh": ("US", "Disney", "Milne 1926 book version public domain in US since 2022"),
    "Woody": ("US", "Pixar (Disney)", ""), "Yoda": ("US", "Lucasfilm (Disney)", ""),
}


def _kw(text: str) -> str:
    """Normalize a CopyCat keyword cell the way CopyCat main.py does."""
    text = str(text).replace("\n", "").replace(", cartoon", "")
    return ", ".join(p.strip() for p in text.split(",") if p.strip())


# Naming mismatches inside the released CopyCat files (see docs/DATA.md, "Known issues").
COPYCAT_ALIASES = {"Sponge Bob": "SpongeBob SquarePants", "baby yoda": "Yoda"}


def build_copycat(tokenizer) -> pd.DataFrame:
    """He et al. ICLR'25 CopyCat suite: 50 characters, lures, keywords, GPT-4 rewrites."""
    base = raw_dir() / "copycat/prompts"

    def col(path, key="prompt"):
        d = pd.read_csv(base / path)
        names = [COPYCAT_ALIASES.get(t.strip(), t.strip()) for t in d["target"]]
        # targets_dalle_prompt.csv lists "Steve" where "Link" belongs; a missing rewrite stays empty
        # and is regenerated by scripts/run_rewriter.py
        return {n: v for n, v in zip(names, d[key])}

    targets = [t.strip() for t in pd.read_csv(base / "targets.csv")["target"]]
    desc = col("targets_60words.csv")
    rw_name = col("targets_dalle_prompt.csv")
    rw_desc = col("targets_60words_dallerewrite.csv")
    kw_lm = col("keywords_lm/n=5_related_greedy.csv", "keyword")
    kw_emb = col("keywords_embedding/n=5.csv")
    kw_laion = col("keywords_co-occurrence/50keywords_laion2b_top5.csv")
    entity = col("consistency/key_entitie_n1.csv", "keyword")
    def strip_q(s) -> str:
        return "" if s is None or pd.isna(s) else str(s).strip().strip("'\"").strip()

    rows = []
    for i, t in enumerate(targets):
        region, holder, pd_note = CHARACTER_ORIGIN[t]
        rows.append({
            "id": i, "target": t, "prompt": t, "desc60": desc[t].strip(),
            "kw_lm5": _kw(kw_lm[t]), "kw_embedding5": _kw(kw_emb[t]), "kw_laion5": _kw(kw_laion[t]),
            "key_entity": entity[t], "gpt4_rewrite_name": strip_q(rw_name.get(t)),
            "gpt4_rewrite_desc60": strip_q(rw_desc.get(t)), "origin_region": region,
            "rights_holder": holder, "pd_note": pd_note,
        })
    df = pd.DataFrame(rows)
    df["desc60_clip_tokens"] = _clip_len(tokenizer, df["desc60"])
    df["split"] = assign_splits(df, 0.2, "origin_region")
    return df


# Track 2 artists. group: living (core target), deceased_copyright (extension), public_domain (core control).
# latest_work < 1912 is the boundary in the DALL-E rewriting policy (He et al. / REWRITE_SYSTEM_PROMPT rule 3).
ARTISTS = [
    # name, group, country, death_year, latest_work, wikiart_label, prior_work
    ("Kelly McKernan", "living", "US", None, None, None, "Pham et al. ICLR'24; ESD; Andersen v. Stability AI"),
    ("Tyler Edlin", "living", "US", None, None, None, "Pham et al. ICLR'24; ESD"),
    ("Kilian Eng", "living", "Sweden", None, None, None, "Pham et al. ICLR'24; ESD"),
    ("Greg Rutkowski", "living", "Poland", None, None, None, "widely reported mimicry target"),
    ("Karla Ortiz", "living", "US", None, None, None, "Andersen v. Stability AI"),
    ("Sarah Andersen", "living", "US", None, None, None, "Andersen v. Stability AI"),
    ("Hollie Mengert", "living", "US", None, None, None, "documented fine-tuned mimicry case (2022)"),
    ("Loish", "living", "Netherlands", None, None, None, "frequently prompted digital artist"),
    ("Simon Stalenhag", "living", "Sweden", None, None, None, "frequently prompted digital artist"),
    ("Artgerm", "living", "Singapore", None, None, None, "frequently prompted digital artist"),
    ("Thomas Kinkade", "deceased_copyright", "US", 2012, 2012, None, "Pham et al. ICLR'24; ESD"),
    ("Pablo Picasso", "deceased_copyright", "Spain", 1973, 1973, "pablo-picasso", "DALL-E policy example; ESD"),
    ("Salvador Dali", "deceased_copyright", "Spain", 1989, 1989, "salvador-dali", ""),
    ("Andy Warhol", "deceased_copyright", "US", 1987, 1987, "andy-warhol", "ESD"),
    ("M.C. Escher", "deceased_copyright", "Netherlands", 1972, 1972, "m.c.-escher", ""),
    ("Vincent van Gogh", "public_domain", "Netherlands", 1890, 1890, "vincent-van-gogh",
     "Pham et al. ICLR'24; Kim et al. ICML-W'24; Reissinger et al. 2025; ESD"),
    ("Claude Monet", "public_domain", "France", 1926, 1926, "claude-monet", ""),
    ("Rembrandt", "public_domain", "Netherlands", 1669, 1669, "rembrandt", "ESD"),
    ("Katsushika Hokusai", "public_domain", "Japan", 1849, 1849, "katsushika-hokusai", ""),
    ("Gustav Klimt", "public_domain", "Austria", 1918, 1918, "gustav-klimt", ""),
    ("Paul Cezanne", "public_domain", "France", 1906, 1906, "paul-cezanne", ""),
    ("Edvard Munch", "public_domain", "Norway", 1944, 1944, "edvard-munch", ""),
    ("Francisco Goya", "public_domain", "Spain", 1828, 1828, "francisco-goya", "DALL-E policy example"),
    ("Georges Seurat", "public_domain", "France", 1891, 1891, "georges-seurat", ""),
    ("Albrecht Durer", "public_domain", "Germany", 1528, 1528, "albrecht-durer", ""),
]

# Proposal Track 2: 10 prompts per artist over four structural templates (Glaze-style protocol).
STYLE_TEMPLATES = [
    ("simple", "a painting of a fox in the style of {artist}"),
    ("simple", "a painting of a lighthouse in the style of {artist}"),
    ("simple", "a painting of a fantasy dragon in the style of {artist}"),
    ("two_entity", "a painting of a cat and a teapot in the style of {artist}"),
    ("two_entity", "a painting of a violinist and a bicycle in the style of {artist}"),
    ("entity_attribute", "a painting of a sleeping old woman in the style of {artist}"),
    ("entity_attribute", "a painting of a glowing red sailboat in the style of {artist}"),
    ("entity_scene", "a painting of a mountain cabin in a misty forest in the style of {artist}"),
    ("entity_scene", "a painting of a street vendor on a rainy city street in the style of {artist}"),
    ("entity_scene", "a painting of a girl in a sunflower field in the style of {artist}"),
]
STYLE_DEV_TEMPLATES = {1, 7}  # template indices held out for tuning (one simple, one scene)


def build_artists() -> pd.DataFrame:
    rows = []
    for name, group, country, death, latest, wiki, prior in ARTISTS:
        rows.append({
            "artist": name, "group": group, "core": group != "deceased_copyright", "country": country,
            "death_year": death or "", "latest_work_year": latest or "",
            "dalle_policy_allows_name": bool(latest and latest < 1912),
            "wikiart_label": wiki or "", "reference_images": "wikiart" if wiki else "none (not collected)",
            "prior_work": prior,
        })
    return pd.DataFrame(rows)


def build_style_prompts(artists: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in artists.iterrows():
        for j, (kind, tmpl) in enumerate(STYLE_TEMPLATES):
            rows.append({"id": f"{a['artist']}|{j}", "artist": a["artist"], "group": a["group"],
                         "template_id": j, "template_type": kind, "prompt": tmpl.format(artist=a["artist"]),
                         "neutral_prompt": tmpl.replace(" in the style of {artist}", ""),
                         "target": a["artist"], "split": "dev" if j in STYLE_DEV_TEMPLATES else "test"})
    return pd.DataFrame(rows)


def write_manifest(df: pd.DataFrame, name: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
