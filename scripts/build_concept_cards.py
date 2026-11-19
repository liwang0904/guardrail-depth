"""One-line appearance card per protected character, written by the screen's own LLM from the name alone.

    python scripts/build_concept_cards.py [--llm rewriter_reasoning]   # -> data/manifests/concept_cards.csv

The concept screen lists these cards next to the names so it can match a description that never names
the character. They come from the character's name only (no CopyCat text, no test prompt) and are
committed, so every run screens with the same cards.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.gate import protected_concepts  # noqa: E402
from guardrail_depth.models import pick_device, registry  # noqa: E402
from guardrail_depth.paths import MANIFEST_DIR  # noqa: E402

PROMPT = ("In one line of at most 16 words, list what the character {name} looks like: body or species, colours, "
          "clothing and the most distinctive visual features. Do not write the name, the franchise or a story.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="rewriter_reasoning")
    ap.add_argument("--device")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = registry()["auxiliary"][args.llm]
    dev = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"])
    dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    llm = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], revision=cfg["revision"], dtype=dtype).to(dev).eval()
    rows = []
    for name in protected_concepts().query("kind == 'character'")["name"]:
        msgs = [{"role": "user", "content": PROMPT.format(name=name)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = llm.generate(**ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        card = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().splitlines()[0]
        for part in re.split(r"[\s-]+", name):          # keep the card name-free
            if len(part) > 3:
                card = re.sub(rf"\b{re.escape(part)}\b", "", card, flags=re.IGNORECASE)
        card = re.sub(r"\s+", " ", card).strip(" .,;")
        rows.append({"name": name, "card": card, "card_model": cfg["hf_id"]})
        print(f"{name:24s} {card}", flush=True)
    pd.DataFrame(rows).to_csv(MANIFEST_DIR / "concept_cards.csv", index=False)
    print(f"wrote {len(rows)} cards -> {MANIFEST_DIR / 'concept_cards.csv'}")


if __name__ == "__main__":
    main()
