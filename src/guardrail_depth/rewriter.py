"""D4: LLM prompt rewriting (He et al. ICLR'25 / DALL-E 3 policy), with reasoning-trace logging.

He et al. used GPT-4; we use open models so the whole pipeline runs on Markov and so we can read
the model's internals.  For reasoning models (Qwen3 with thinking enabled) the trace between
<think> and </think> is kept separately from the final rewrite, and we log per-token predictive
entropy for both parts - the "does the reasoning process show a pattern / does entropy change in
failures" analysis requested in the proposal feedback.
"""

from __future__ import annotations

import re

import torch

from .defenses import REWRITE_SYSTEM_PROMPT, REWRITE_USER_TEMPLATE


def _dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device.type == "mps" else torch.float32


class LLMRewriter:
    def __init__(self, model_id: str, revision: str | None = None, device: torch.device | str = "cpu",
                 thinking: bool = False, max_new_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.95):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.model_id, self.thinking = model_id, thinking
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=_dtype(self.device))
        self.model = self.model.to(self.device).eval()
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p
        self.think_end = self.tok.convert_tokens_to_ids("</think>") if "</think>" in self.tok.get_vocab() else None

    @torch.no_grad()
    def rewrite(self, request: str, seed: int = 0, system_prompt: str = REWRITE_SYSTEM_PROMPT) -> dict:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": REWRITE_USER_TEMPLATE.format(request=request)}]
        extra = {"enable_thinking": self.thinking} if self.think_end is not None else {}
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None, top_p=self.top_p,
            output_logits=True, return_dict_in_generate=True, pad_token_id=self.tok.eos_token_id,
        )
        gen = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        logits = torch.stack(out.logits)[:, 0].float()
        logp = torch.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1).cpu().numpy()          # nats per generated token
        if self.think_end in gen:
            split = gen.index(self.think_end) + 1
        else:  # thinking ran out of tokens before answering: no rewrite, keep the trace
            split = len(gen) if self.thinking and self.think_end is not None else 0
        reasoning = self.tok.decode(gen[:split], skip_special_tokens=True)
        final = self.tok.decode(gen[split:], skip_special_tokens=True)
        reasoning = re.sub(r"</?think>", "", reasoning).strip()
        return {
            "rewrite": final.strip().strip('"').strip(),
            "reasoning": reasoning,
            "n_reason_tokens": split,
            "n_final_tokens": len(gen) - split,
            "entropy_reason_mean": float(entropy[:split].mean()) if split else float("nan"),
            "entropy_final_mean": float(entropy[split:].mean()) if len(gen) > split else float("nan"),
            "entropy_final_max": float(entropy[split:].max()) if len(gen) > split else float("nan"),
            "truncated": len(gen) >= self.max_new_tokens,
            "rewriter": self.model_id + (" (thinking)" if self.thinking else ""),
        }


def _variants(name: str) -> set[str]:
    n = name.lower().strip()
    return {n, n.replace("-", " "), n.replace("-", ""), n.replace(" ", "")}


def leakage(text: str, target: str, keywords: list[str]) -> dict:
    """Does a rewrite (or reasoning trace) still carry the protected identity?"""
    low = text.lower()
    kws = list(dict.fromkeys(k.strip().lower() for k in keywords if k and k.strip()))  # LM and LAION lists overlap
    hits = [k for k in kws if k in low]
    return {
        "name_leak": any(v in low for v in _variants(target)) if target else False,
        "kw_hits": len(hits),
        "kw_total": len(kws),
        "kw_hit_list": "; ".join(hits),
    }
