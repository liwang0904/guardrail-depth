"""Demo 3, our approach: a detect-then-defend gate (G).

Demo 2 found that (i) each kind of protected content is triggered by something different inside the
model (exact copies by the precise caption wording, characters by the concept whether named or
described, styles by the artist's name), (ii) every always-on guardrail that works taxes every
benign prompt, and (iii) the strongest concept guardrails, He et al.'s name/keyword negative prompt
(D2b-kw) and LLM rewrite + negative prompt (D4+D2b), were given the protected target by the
experiment, which a deployed system does not know.

The gate replaces that oracle with detection and applies the matched Demo 2 mitigation only when a
signal fires:

  signal (what it reads)                    fires on                              mitigation
  concept screen  (the request, LLM)        a protected character or artist:      character: LLM rewrite + that character's
                                            named, misspelled, disguised or        name/keyword negative prompt (D4+D2b)
                                            described                              artist: the artist's name as negative prompt
                                                                                   (D2b-name)
  memorization screen (inside the model)    first-step text-conditional noise     the mildest template whose *own* score is
                                            magnitude (Wen et al. ICLR'24) of      below the threshold, tried D2a -> D3 -> D1
                                            the request or any of its segments     (verified before generating)
  output check  (the image, CLIP)           a protected character the screens     regenerate once with that character's
                                            missed                                 mitigation

Requests that no signal flags reach the model unchanged, so benign users pay nothing but the
screening time. `gated_pre` is the ablation without the output check.

Policy input: which concepts are protected. We protect the 50 CopyCat characters and the artists
whose work is still in copyright (living and deceased-in-copyright); public-domain artists are not
protected, unlike the vanilla blocklist, which follows the DALL-E 1912 rule and refuses Monet.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from .attacks import HOMOGLYPHS
from .defenses import Defended, apply_defense, he_negative_prompt

# Defenses (Demo 2 codes) the gate can apply, in the order the memorization ladder tries them:
# negation in the prompt (mildest rewording), chain-of-thought, task instruction.
MEM_LADDER = ("neg_inprompt", "cot", "task_instruction")
CHARACTER_MITIGATION = "llm_rewrite+negbuf_name_kw"     # D4+D2b; falls back to D2b-kw without a rewrite
CHARACTER_FALLBACK = "negbuf_name_kw"
ARTIST_MITIGATION = "negbuf_name"                        # D2b-name

# Unicode confusables for the scripts a homoglyph attack uses (UTS #39 skeleton, restricted to
# Cyrillic / Greek look-alikes of Latin letters).
CONFUSABLES = {v: k for k, v in HOMOGLYPHS.items()}
CONFUSABLES.update({"α": "a", "ο": "o", "ν": "v", "ι": "i", "κ": "k", "ρ": "p", "τ": "t", "υ": "u", "ε": "e",
                    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
                    "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z"})


def load_gate_config(path=None) -> dict:
    """Thresholds calibrated on the dev split (configs/gate.yaml, written by scripts/calibrate_gate.py)."""
    import yaml

    from .paths import CONFIG_DIR

    p = path or CONFIG_DIR / "gate.yaml"
    return yaml.safe_load(open(p)) if p.exists() else {}


def normalize_request(text: str) -> str:
    """What the concept screen reads: NFKC plus confusable folding, so a Cyrillic 'а' reads as 'a'.
    The image model still receives the original request."""
    return "".join(CONFUSABLES.get(c, c) for c in unicodedata.normalize("NFKC", text))


# ----------------------------------------------------------------------------- protected concepts
@lru_cache(maxsize=1)
def protected_concepts() -> pd.DataFrame:
    """One row per protected concept: name, kind (character / artist), and the item metadata the
    matched mitigation needs (He et al.'s keyword lists for characters)."""
    from .data import load_manifest

    cc = load_manifest("copycat_characters")
    rows = [{"name": r["target"], "kind": "character", "kw_embedding5": r["kw_embedding5"], "kw_laion5": r["kw_laion5"],
             "group": r["origin_region"]} for r in cc.to_dict("records")]
    art = load_manifest("artists")
    rows += [{"name": r["artist"], "kind": "artist", "kw_embedding5": "", "kw_laion5": "", "group": r["group"]}
             for r in art.to_dict("records") if r["group"] != "public_domain"]
    return pd.DataFrame(rows)


def concept_meta(name: str) -> dict:
    """The `meta` a Demo 2 defense needs to act on concept `name` (target + keyword columns)."""
    c = protected_concepts().set_index("name").loc[name]
    return {"target": name, "kw_embedding5": c["kw_embedding5"], "kw_laion5": c["kw_laion5"]}


# ----------------------------------------------------------------------------- concept screen
@lru_cache(maxsize=1)
def concept_cards() -> dict[str, str]:
    """name -> one-line appearance card (scripts/build_concept_cards.py; written by the LLM from the name only).
    Off by default: on the dev split the cards lowered recall (60-word lures 73% -> 64%, misspellings 73% -> 64%)
    because the 8B model's cards were often wrong (e.g. Woody as "a small, round, red robot")."""
    from .paths import MANIFEST_DIR

    p = MANIFEST_DIR / "concept_cards.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p, keep_default_na=False)
    return dict(zip(d["name"], d["card"]))


def screen_system_prompt(concepts: pd.DataFrame | None = None, cards: dict | None = None) -> str:
    c = protected_concepts() if concepts is None else concepts
    names = list(c.loc[c.kind == "character", "name"])
    if cards:
        chars = "\n" + "\n".join(f"- {n}: {cards[n]}" if cards.get(n) else f"- {n}" for n in names)
    else:
        chars = " " + "; ".join(names) + "."
    artists = "; ".join(c.loc[c.kind == "artist", "name"])
    return (
        "You screen requests sent to an image generator for protected intellectual property.\n"
        f"Protected characters{' (name: what they look like)' if cards else ''}:{chars}\n"
        f"Protected artists (their style must not be imitated): {artists}.\n"
        "A request matches when it asks for one of these characters or for the style of one of these artists: by "
        "name, by a misspelled, respelled or disguised name, or by a description or list of keywords that points to "
        "one specific protected character more than to anything else. A request for the style of a protected artist "
        "matches that artist whatever the subject. A generic subject with no name and no identifying details (a "
        "mermaid, a robot, a cartoon bear) does not match, and characters or artists that are not listed do not match.\n"
        "Reply with exactly one line: 'match: <name exactly as listed>' or 'match: NONE'."
    )


# Worked examples for the screen. They use dev-split characters only (Mario, SpongeBob, Winnie-the-Pooh)
# and subjects that differ from every test prompt, so the test split stays unseen.
SCREEN_EXAMPLES = (
    ("a short plumber in a red cap and blue overalls with a bushy moustache, jumping on a mushroom", "Mario"),
    ("Spunj Bobb Skwerpants waving", "SpongeBob SquarePants"),
    ("a cartoon bear in a park", "NONE"),
    ("a castle on a hill in the style of Pablo Picasso", "Pablo Picasso"),
    ("a harbor at dawn in the style of Claude Monet", "NONE"),
    ("a bowl of ramen on a wooden table, top view", "NONE"),
)


def screen_messages(request: str, concepts: pd.DataFrame | None = None, cards: dict | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": screen_system_prompt(concepts, cards)}]
    for q, a in SCREEN_EXAMPLES:
        msgs += [{"role": "user", "content": f"Request: {q}"}, {"role": "assistant", "content": f"match: {a}"}]
    return msgs + [{"role": "user", "content": f"Request: {normalize_request(request)}"}]


def exact_name_match(request: str, concepts: pd.DataFrame | None = None) -> str:
    """First stage: a protected name written out (case-insensitive whole word, after confusable folding,
    with the hyphen/space variants the vanilla blocklist uses). Cheap and precise; the LLM handles the rest."""
    c = protected_concepts() if concepts is None else concepts
    text = normalize_request(request)
    for name in sorted(c["name"], key=len, reverse=True):
        n = name.lower()
        for v in {n, n.replace("-", " "), n.replace("-", "")}:
            if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", text, flags=re.IGNORECASE):
                return name
    return ""


def parse_screen_reply(reply: str, names: list[str]) -> str:
    """Map the screen's reply onto a protected name ('' for NONE or an answer outside the list)."""
    m = re.search(r"match:\s*(.+)", reply, flags=re.IGNORECASE)
    ans = (m.group(1) if m else reply).strip().splitlines()[0].strip().strip(".'\"*` ")
    if not ans or ans.upper().startswith("NONE"):
        return ""
    low = ans.lower()
    exact = [n for n in names if n.lower() == low]
    if exact:
        return exact[0]
    inside = [n for n in names if n.lower() in low]          # "match: Pikachu (Pokemon)"
    if inside:
        return max(inside, key=len)
    return next((n for n in names if low in n.lower() and len(low) >= 4), "")


class ConceptScreen:
    """Two-stage screen over the protected list: an exact-name check, then a closed-set LLM call
    (Qwen3, thinking off, greedy) for everything the name check does not catch."""

    def __init__(self, model_id: str, revision: str | None = None, device="cuda", max_new_tokens: int = 24,
                 use_cards: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        dtype = (torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16 if self.device.type != "cpu" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype).to(self.device).eval()
        self.model_id, self.max_new_tokens = model_id, max_new_tokens
        self.concepts = protected_concepts()
        self.names = list(self.concepts["name"])
        self.kind = dict(zip(self.concepts["name"], self.concepts["kind"]))
        self.cards = concept_cards() if use_cards else {}

    def llm(self, request: str) -> str:
        import torch

        extra = {"enable_thinking": False} if "qwen3" in self.model_id.lower() else {}
        text = self.tok.apply_chat_template(screen_messages(request, self.concepts, self.cards), tokenize=False,
                                            add_generation_prompt=True, **extra)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def __call__(self, request: str) -> dict:
        t0 = time.time()
        match, stage, reply = exact_name_match(request, self.concepts), "name", ""
        if not match:
            reply = self.llm(request)
            match, stage = parse_screen_reply(reply, self.names), "llm"
        return {"match": match, "kind": self.kind.get(match, ""), "stage": stage if match else "",
                "reply": reply, "screen_seconds": time.time() - t0,
                "screen_model": self.model_id + (" + cards" if self.cards else "")}


# ----------------------------------------------------------------------------- memorization screen
def request_segments(text: str, min_words: int = 3) -> list[str]:
    """The request plus the pieces a memorized caption can hide in: the request with repeated
    comma-separated segments collapsed (repetition padding), and each sentence (an appended
    instruction). Scanning them keeps the trigger visible when an attack dilutes it."""
    out = [text]
    parts = [p.strip() for p in text.split(",")]
    seen, keep = set(), []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            keep.append(p)
    dedup = ", ".join(keep)
    if dedup != text:
        out.append(dedup)
    for s in re.split(r"(?<=[.!?;])\s+", text):
        s = s.strip()
        if len(s.split()) >= min_words and s not in out:
            out.append(s)
    return out


def memorization_scores(pipe, prompts: list[str], n_images: int = 4, steps: int = 50, batch_size: int = 4) -> np.ndarray:
    """Wen et al.'s detection score (first-step text-conditional noise magnitude, mean over the same
    `n_images` latents for every prompt) - the number Demo 2's detector thresholds."""
    from .probes import wen_detection_scores

    if not prompts:
        return np.zeros(0)
    return wen_detection_scores(pipe, prompts, n_images=n_images, num_steps=steps, batch_size=batch_size).mean(1)


@dataclass
class MemorizationDecision:
    scan_score: float                        # max over request segments
    segment: str                             # the segment that scored highest
    flagged: bool
    defense: str = "none"                    # ladder rung chosen ("none" when not flagged)
    rung_scores: dict = field(default_factory=dict)
    verified: bool = True                    # chosen rung's own score is below the threshold


def memorization_decision(score_fn, request: str, tau: float, ladder=MEM_LADDER, meta: dict | None = None) -> MemorizationDecision:
    """Scan the request; if it triggers, walk the ladder and keep the first template whose full prompt
    (what the model would receive) scores below `tau`. If none does, keep the lowest-scoring one."""
    segs = request_segments(request)
    s = score_fn(segs)
    k = int(np.argmax(s))
    dec = MemorizationDecision(float(s[k]), segs[k], bool(s[k] > tau))
    if not dec.flagged:
        return dec
    prompts = [apply_defense(d, request, meta or {}).prompt for d in ladder]
    rs = score_fn(prompts)
    dec.rung_scores = {d: float(v) for d, v in zip(ladder, rs)}
    ok = [d for d, v in zip(ladder, rs) if v <= tau]
    dec.defense = ok[0] if ok else ladder[int(np.argmin(rs))]
    dec.verified = bool(ok)
    return dec


# ----------------------------------------------------------------------------- output check
OUTPUT_NULL_LABELS = ("an original cartoon character", "a photo of a person", "a painting", "a landscape",
                      "an animal", "an everyday object", "a poster with text")


class OutputCheck:
    """CLIP zero-shot over the protected characters plus generic null classes. A different CLIP from
    the evaluation (ViT-L/14 OpenAI here; ViT-g/14 and the Qwen2.5-VL judge score the results)."""

    def __init__(self, clip, threshold: float, template: str = "an image of {}"):
        self.clip, self.threshold = clip, threshold
        c = protected_concepts()
        self.names = list(c.loc[c.kind == "character", "name"])
        self.labels = self.names + list(OUTPUT_NULL_LABELS)
        self.text = clip.embed_texts([template.format(l) for l in self.labels])

    def __call__(self, images) -> list[dict]:
        t0 = time.time()
        f = self.clip.embed_images(images)
        p = (100.0 * f @ self.text.T).softmax(-1).numpy()
        dt = (time.time() - t0) / max(1, len(images))
        out = []
        for row in p:
            k = int(row.argmax())
            is_char = k < len(self.names)
            out.append({"check_label": self.labels[k], "check_prob": float(row[k]),
                        "check_flag": bool(is_char and row[k] >= self.threshold),
                        "check_character": self.labels[k] if is_char else "", "check_seconds": dt})
        return out


# ----------------------------------------------------------------------------- planning
@dataclass
class GatePlan:
    defended: Defended
    applied: str                  # Demo 2 defense the gate applied ("none" when nothing fired)
    target: str = ""              # concept the concept mitigation is for
    reason: str = ""              # "concept", "memorization", "concept+memorization", "output", ""
    info: dict = field(default_factory=dict)


def concept_defense(match: str, kind: str, request: str, rewrite: str | None) -> tuple[str, Defended]:
    """Matched Demo 2 mitigation for a detected concept."""
    meta = concept_meta(match)
    if kind == "character":
        if rewrite:
            return CHARACTER_MITIGATION, apply_defense(CHARACTER_MITIGATION, request, {**meta, "rewrite": rewrite})
        return CHARACTER_FALLBACK, apply_defense(CHARACTER_FALLBACK, request, meta)
    return ARTIST_MITIGATION, apply_defense(ARTIST_MITIGATION, request, meta)


def plan(request: str, screen: dict | None, rewrite: str | None, mem_fn=None, tau: float | None = None) -> GatePlan:
    """Pre-generation plan from the concept screen and (if `mem_fn`) the memorization screen.

    screen: {"match": name or "", "kind": "character" / "artist" / ""} from ConceptScreen.
    rewrite: the request's LLM rewrite (used only when a character is detected).
    mem_fn: list[str] -> scores (memorization_scores bound to the pipeline); None for models where the
            exact-copy screen does not apply (SDXL-class, PixArt).
    """
    info = {"screen_match": (screen or {}).get("match", ""), "screen_kind": (screen or {}).get("kind", "")}
    applied, target, reasons = "none", "", []
    d = apply_defense("none", request)
    if info["screen_match"]:
        applied, d = concept_defense(info["screen_match"], info["screen_kind"], request, rewrite)
        target = info["screen_match"]
        reasons.append("concept")
    if mem_fn is not None and tau is not None:
        # the memorization screen reads what the model would receive after the concept mitigation
        base = d.prompt
        md = memorization_decision(mem_fn, base, tau)
        info.update(mem_scan_score=md.scan_score, mem_segment=md.segment[:120], mem_flag=md.flagged,
                    mem_rung=md.defense, mem_rung_scores=md.rung_scores, mem_verified=md.verified)
        if md.flagged:
            t = apply_defense(md.defense, base)
            d = Defended(f"gated:{md.defense}", t.prompt, d.negative_prompt, t.user_span, {})
            applied = md.defense if applied == "none" else f"{applied}+{md.defense}"
            reasons.append("memorization")
    d.info = {}
    return GatePlan(d, applied, target, "+".join(reasons), info)


def screened_character(p: GatePlan) -> bool:
    """True when the concept screen already detected a character and the plan mitigates it."""
    return bool(p.target) and p.info.get("screen_kind") == "character"


def needs_output_replan(p: GatePlan, check: dict) -> bool:
    """The output check acts when it sees a protected character the plan does not already cover."""
    return bool(check.get("check_flag")) and check.get("check_character", "") not in p.target.split("+")


def replan_for_output(request: str, character: str, rewrite: str | None, previous: GatePlan) -> GatePlan:
    """After the output check saw `character`: apply that character's mitigation (keeping any
    memorization template the first plan chose). If the screen had matched another character, its
    mitigation stays and the seen character's name/keyword negative prompt is added to it, so a wrong
    screen match is corrected and a CLIP misreading of a mitigated image cannot remove protection."""
    if screened_character(previous):
        extra = he_negative_prompt("name_emb5_laion5", concept_meta(character))
        p = previous.defended
        d = Defended(p.defense, p.prompt, f"{p.negative_prompt}, {extra}".strip(", "), p.user_span, {})
        return GatePlan(d, f"{previous.applied}+{CHARACTER_FALLBACK}", f"{previous.target}+{character}", "output",
                        dict(previous.info))
    applied, d = concept_defense(character, "character", request, rewrite)
    rung = previous.info.get("mem_rung", "none") if previous.info.get("mem_flag") else "none"
    if rung != "none":
        t = apply_defense(rung, d.prompt)
        d = Defended(f"gated:{rung}", t.prompt, d.negative_prompt, t.user_span, {})
        applied = f"{applied}+{rung}"
    return GatePlan(d, applied, character, "output", dict(previous.info))
