"""Model-level probes: what happens *inside* the diffusion model when a guardrail holds or fails.

Each probe re-implements the signal used by a closest-reference paper so that our
prompt-level results can be read against established memorization mechanisms:

  NoiseNormProbe        Wen et al. ICLR'24 - magnitude of the text-conditional noise prediction
                        ||eps(x_t, c) - eps(x_t, u)|| per step (u = empty or negative prompt)
  wen_detection_scores  the same signal at the first step with a true empty-prompt baseline
                        (their detection metric; AUC vs. non-memorized prompts)
  CrossAttentionProbe   Ren et al. ECCV'24 / Chen et al. ICLR'25 - cross-attention over the 77 text
                        tokens: entropy, BOS / end-token ("summary") mass, and how much attention the
                        guardrail's own tokens receive compared with the user's tokens
  X0TrajectoryProbe     predicted clean image x0 at selected steps (when does a copy "lock in"?)
  neuron_activations    Hintersdorf et al. NeurIPS'24 (NeMo) - |value-projection| activations in the
                        UNet cross-attention layers, z-scored against non-memorized prompts

Probes attach to an unmodified diffusers pipeline through hooks/attention processors, so
generation with and without probes is the same computation (up to SDPA vs. explicit softmax).
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch


# ----------------------------------------------------------------------------- noise magnitude
class NoiseNormProbe:
    """Records ||eps_cond - eps_uncond||_2 for every UNet call under classifier-free guidance."""

    def __init__(self):
        self.records: list[torch.Tensor] = []
        self._handle = None

    def _hook(self, module, args, output):
        eps = output[0] if isinstance(output, tuple) else output.sample
        if eps.shape[0] % 2:
            return  # no CFG batch
        uncond, cond = eps.float().chunk(2)
        self.records.append((cond - uncond).flatten(1).norm(dim=1))   # stays on device: no per-step sync

    def attach(self, unet):
        self._handle = unet.register_forward_hook(self._hook)

    def detach(self):
        if self._handle:
            self._handle.remove()

    def reset(self):
        self.records = []

    def result(self) -> np.ndarray:
        """(steps, batch) array."""
        return torch.stack(self.records).cpu().numpy() if self.records else np.zeros((0, 0))


@torch.no_grad()
def encode_text(pipe, prompts: list[str]) -> torch.Tensor:
    """CLIP text embeddings exactly as SD1/SD2 pipelines build them (77 tokens, truncated)."""
    tok = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    return pipe.text_encoder(tok.input_ids.to(pipe.device))[0]


@torch.no_grad()
def wen_detection_scores(pipe, prompts: list[str], n_images: int = 4, seed: int = 0, num_steps: int = 50,
                         negative_prompts: list[str] | None = None, batch_size: int = 8) -> np.ndarray:
    """First-step text-conditional noise magnitude (Wen et al., ICLR'24, Table 1 setting).

    Returns (len(prompts), n_images). High values flag memorization. With
    `negative_prompts`, the reference branch is the negative prompt instead of "",
    i.e. the guidance direction that a negative-buffer defense actually applies.
    """
    unet, sched = pipe.unet, pipe.scheduler
    sched.set_timesteps(num_steps, device=pipe.device)
    t0 = sched.timesteps[0]
    h = w = unet.config.sample_size
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        cond = encode_text(pipe, chunk).repeat_interleave(n_images, 0)
        refs = negative_prompts[i:i + batch_size] if negative_prompts else [""] * len(chunk)
        uncond = encode_text(pipe, refs).repeat_interleave(n_images, 0)
        # the same n_images initial latents for every prompt, so prompts are compared on paired noise
        gens = [torch.Generator("cpu").manual_seed(seed + k) for _ in chunk for k in range(n_images)]
        lat = torch.stack([torch.randn((unet.config.in_channels, h, w), generator=g) for g in gens])
        lat = (lat * sched.init_noise_sigma).to(pipe.device, unet.dtype)
        x = sched.scale_model_input(torch.cat([lat, lat]), t0)
        eps = unet(x, t0, encoder_hidden_states=torch.cat([uncond, cond])).sample.float()
        e_u, e_c = eps.chunk(2)
        out.append((e_c - e_u).flatten(1).norm(dim=1).view(len(chunk), n_images).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- cross-attention
class _RecordingCrossAttnProcessor:
    """diffusers AttnProcessor (explicit softmax) that also reports attention over text tokens."""

    def __init__(self, probe: "CrossAttentionProbe", name: str):
        self.probe, self.name = probe, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, hh, ww = hidden_states.shape
            hidden_states = hidden_states.view(b, c, hh * ww).transpose(1, 2)
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None
                                          else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query, key, value = attn.head_to_batch_dim(query), attn.head_to_batch_dim(key), attn.head_to_batch_dim(value)
        # Not attn.get_attention_scores: its baddbmm(beta=0) reads uninitialized memory, which
        # propagates NaN on Apple MPS. Scores and softmax in fp32, as the upcast path would.
        scores = torch.bmm(query.float(), key.float().transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        probs = scores.softmax(dim=-1)                                         # (B*heads, Q, K)
        self.probe.record(self.name, probs.view(batch_size, attn.heads, *probs.shape[1:]))
        hidden_states = attn.batch_to_head_dim(torch.bmm(probs.to(value.dtype), value))
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, hh, ww)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class CrossAttentionProbe:
    """Per-step, per-layer attention mass on each of the 77 text tokens (conditional branch).

    Optionally keeps the spatial map of one token (e.g. the end token, Chen et al. ICLR'25)
    at the final step.
    """

    def __init__(self, cfg: bool = True, keep_spatial_token: int | None = None):
        self.cfg = cfg
        self.keep_spatial_token = keep_spatial_token
        self.step = -1
        self.token_mass: dict[int, dict[str, torch.Tensor]] = {}
        self.spatial: dict[str, torch.Tensor] = {}
        self._orig = None
        self._handle = None

    def record(self, name: str, probs: torch.Tensor):
        if self.cfg:
            probs = probs[probs.shape[0] // 2:]                 # conditional half
        mass = probs.float().mean(dim=(1, 2))                    # (B, K), kept on device until result()
        self.token_mass.setdefault(self.step, {})[name] = mass
        if self.keep_spatial_token is not None:
            self.spatial[name] = probs[..., self.keep_spatial_token].float().mean(1)  # (B, Q), last step wins

    def _pre_hook(self, module, args):
        self.step += 1

    def attach(self, unet):
        self._orig = unet.attn_processors
        procs = {k: (_RecordingCrossAttnProcessor(self, k) if "attn2" in k else v) for k, v in self._orig.items()}
        unet.set_attn_processor(procs)
        self._handle = unet.register_forward_pre_hook(self._pre_hook)

    def detach(self, unet):
        if self._orig is not None:
            unet.set_attn_processor(self._orig)
        if self._handle:
            self._handle.remove()

    def reset(self):
        self.step, self.token_mass, self.spatial = -1, {}, {}

    def result(self) -> np.ndarray:
        """(steps, layers, batch, 77) array, layers in UNet order."""
        steps = sorted(self.token_mass)
        layers = list(self.token_mass[steps[0]]) if steps else []
        if not steps:
            return np.zeros(0)
        return torch.stack([torch.stack([self.token_mass[s][l] for l in layers]) for s in steps]).cpu().numpy()


def attention_summary(mass: np.ndarray, n_prompt_tokens: int, user_tokens: tuple[int, int] | None) -> dict:
    """Scalar summaries of a (..., 77) attention-mass vector.

    Token layout: 0 = BOS, 1..L = prompt, L+1 = EOS, the rest padding (Ren et al. call
    EOS+padding the "summary" tokens). `user_tokens` are offsets within the prompt.
    """
    L = min(n_prompt_tokens, mass.shape[-1] - 2)
    eps = 1e-12
    prompt = mass[..., 1:L + 1]
    p_norm = prompt / (prompt.sum(-1, keepdims=True) + eps)
    out = {
        "attn_bos": mass[..., 0],
        "attn_prompt": prompt.sum(-1),
        "attn_summary": mass[..., L + 1:].sum(-1),
        "attn_entropy": -(mass * np.log(mass + eps)).sum(-1) / math.log(mass.shape[-1]),
        "attn_prompt_entropy": (-(p_norm * np.log(p_norm + eps)).sum(-1) / math.log(max(L, 2))),
    }
    if user_tokens is not None:
        s, e = user_tokens
        s, e = min(s, L), min(e, L)
        user = mass[..., 1 + s:1 + e].sum(-1)
        out["attn_user"] = user
        out["attn_guardrail"] = out["attn_prompt"] - user   # template tokens that survived truncation
    return out


# ----------------------------------------------------------------------------- x0 trajectory
class X0TrajectoryProbe:
    """Captures the scheduler's predicted x0 (latent) at chosen step indices."""

    def __init__(self, steps: list[int]):
        self.steps = set(steps)
        self.latents: dict[int, torch.Tensor] = {}
        self._i = 0
        self._orig = None

    def attach(self, scheduler):
        self._orig = scheduler.step

        def step(*args, **kwargs):
            out = self._orig(*args, **kwargs)
            x0 = out[1] if isinstance(out, tuple) and len(out) > 1 else getattr(out, "pred_original_sample", None)
            if self._i in self.steps and x0 is not None:
                self.latents[self._i] = x0.detach().clone()
            self._i += 1
            return out

        scheduler.step = step

    def detach(self, scheduler):
        if self._orig is not None:
            scheduler.step = self._orig

    def reset(self):
        self.latents, self._i = {}, 0

    @torch.no_grad()
    def decode(self, pipe) -> dict[int, list]:
        out = {}
        for i, lat in sorted(self.latents.items()):
            img = pipe.vae.decode(lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor).sample
            out[i] = pipe.image_processor.postprocess(img, output_type="pil")
        return out


# ----------------------------------------------------------------------------- NeMo neurons
def nemo_value_layers(unet) -> list:
    """The 7 cross-attention value projections NeMo inspects (down blocks 0-2 + mid block)."""
    layers = []
    for block in unet.down_blocks:
        for attn in getattr(block, "attentions", None) or []:
            layers.append(attn.transformer_blocks[0].attn2.to_v)
    layers.append(unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v)
    return layers


@torch.no_grad()
def neuron_activations(pipe, prompts: list[str], batch_size: int = 32) -> list[np.ndarray]:
    """Mean |to_v(text embedding)| over tokens, per layer: list of (n_prompts, d_layer).

    The value projection only sees the text embedding, so these activations depend on the
    prompt alone (NeMo collects them at the first denoising step, which is equivalent).
    """
    layers = nemo_value_layers(pipe.unet)
    acts = [[] for _ in layers]
    for i in range(0, len(prompts), batch_size):
        emb = encode_text(pipe, prompts[i:i + batch_size])
        for j, layer in enumerate(layers):
            acts[j].append(layer(emb).abs().mean(dim=1).float().cpu())
    return [torch.cat(a).numpy() for a in acts]


def neuron_reference_stats(acts: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(a.mean(0), a.std(0) + 1e-6) for a in acts]


def load_nemo_stats(path) -> list[tuple[np.ndarray, np.ndarray]]:
    """NeMo's released statistics (50k LAION prompts, SD v1.4); tensors only, so weights_only=True."""
    means, stds = torch.load(path, map_location="cpu", weights_only=True)
    return [(m.float().numpy(), s.float().numpy()) for m, s in zip(means, stds)]


def neuron_outliers(acts: list[np.ndarray], stats, theta: float = 5.0) -> dict:
    """Per prompt: number of OOD neurons (z > theta) and their identities (layer, index)."""
    n = acts[0].shape[0]
    counts = np.zeros(n, dtype=int)
    max_z = np.full(n, -np.inf)
    sets = [set() for _ in range(n)]
    for layer, (a, (mu, sd)) in enumerate(zip(acts, stats)):
        z = (a - mu) / sd
        max_z = np.maximum(max_z, z.max(1))
        for p, idx in zip(*np.nonzero(z > theta)):
            sets[p].add((layer, int(idx)))
            counts[p] += 1
    return {"n_ood": counts, "max_z": max_z, "neurons": sets}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


# ----------------------------------------------------------------------------- bundle
@contextmanager
def probes_attached(pipe, noise: bool = True, attention: bool = False, x0_steps: list[int] | None = None,
                    cfg: bool = True, keep_spatial_token: int | None = None):
    """Attach the requested probes for the duration of a `with` block."""
    p = {}
    if noise:
        p["noise"] = NoiseNormProbe()
        p["noise"].attach(pipe.unet)
    if attention:
        p["attention"] = CrossAttentionProbe(cfg=cfg, keep_spatial_token=keep_spatial_token)
        p["attention"].attach(pipe.unet)
    if x0_steps:
        p["x0"] = X0TrajectoryProbe(x0_steps)
        p["x0"].attach(pipe.scheduler)
    try:
        yield p
    finally:
        if "noise" in p:
            p["noise"].detach()
        if "attention" in p:
            p["attention"].detach(pipe.unet)
        if "x0" in p:
            p["x0"].detach(pipe.scheduler)
