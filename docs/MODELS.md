# Models

Pinned in `configs/models.yaml` (Hugging Face repo + commit). Download sizes are the fp16 diffusers
components that `scripts/download_models.py` fetches. "Min GPU" is the smallest Markov GPU type that
fits the job in fp16 (`-C` feature); the cluster guide asks us to use the smallest type that fits.

## Subject text-to-image models

| key | model @ revision | sampler settings (from) | size | min GPU | why |
|---|---|---|---|---|---|
| `sd14` | CompVis/stable-diffusion-v1-4 @ 133a221 | DDIM, 50 steps, cfg 7.5, 512 px (Wen et al.; NeMo) | 2.7 GB | gpu2080 | the 500 memorized prompts, the NeMo statistics and Pham et al.'s Concept Inversion are all defined on this checkpoint |
| `sd15` | stable-diffusion-v1-5/stable-diffusion-v1-5 @ 451f4fe | DDIM, 50, 7.5, 512 | 2.7 GB | gpu2080 | proposal model; official mirror after `runwayml/` was removed |
| `sd2base` | sd2-community/stable-diffusion-2-base @ f5bc1bd | DDIM, 50, 7.5, 512 | 2.6 GB | gpu2080 | Webster's 219 SD v2 memorized prompts; Chen et al. ICLR'25 |
| `sd21` | sd2-community/stable-diffusion-2-1 @ bb21548 | 30 steps, cfg 7.5, 768 px (Reißinger et al.) | 2.6 GB | gpu4090 | replicate the defense paper's SD 2 setting |
| `sdxl` | stabilityai/stable-diffusion-xl-base-1.0 @ 4621659 | 50 steps, cfg 3, 1024 px (He et al.) | 7.1 GB | gpu4090 | CopyCat model; proposal model |
| `playground25` | playgroundai/playground-v2.5-1024px-aesthetic @ 1e032f1 | 50 steps, cfg 3, 1024 px (He et al.) | 6.9 GB | gpu4090 | He et al.'s main model |
| `pixart512` | PixArt-alpha/PixArt-XL-2-512x512 @ 50f7021 | 50 steps, cfg 3 (He et al.) | 31.5 GB | gpul40s | optional; T5-XXL text encoder (no 77-token limit) is a useful contrast for the truncation finding |

The official `stabilityai/stable-diffusion-2*` repositories return HTTP 401 since late 2025; the
`sd2-community` mirrors reproduce the original repository layout and are pinned by commit, but are
not hash-verified against the originals. Training data of all subject models: LAION-5B subsets
(LAION-2B-en / LAION-Aesthetics for SD; undisclosed web data for Playground).

The SD 1.x/2.x NSFW safety checker is disabled (`safety_checker=None`), as in Wen et al., NeMo and
Pham et al.: it replaces unrelated images with black frames, which would read as "copy prevented".

## Auxiliary models

| role | model | replaces / follows | size | min GPU |
|---|---|---|---|---|
| infringement judge (DETECT), VQAScore-style consistency (CONS) | Qwen/Qwen2.5-VL-7B-Instruct @ cc59489 | GPT-4V (DETECT) and CLIP-FlanT5-XXL VQAScore (CONS) in He et al. | 16.6 GB | gpu4090 |
| small judge | Qwen/Qwen2.5-VL-3B-Instruct @ 6628554 | — | 7.5 GB | gpu2080 |
| D4 rewriter | Qwen/Qwen2.5-7B-Instruct @ a09a354 | GPT-4 in He et al. (same DALL·E policy system prompt) | 15.2 GB | gpu4090 |
| D4 rewriter with reasoning traces | Qwen/Qwen3-8B @ b968826 (thinking on) | — | 16.4 GB | gpul40s |
| laptop rewriter | Qwen/Qwen3-1.7B @ 70d244c | — | 4.1 GB | gpu2080 / M2 |
| copy detection | SSCD `sscd_disc_mixup.torchscript.pt` (Pizzi et al. CVPR'22) | Somepalli et al., Wen et al., NeMo | 0.1 GB | any |
| image-text / image-image similarity, zero-shot attribution | OpenCLIP ViT-g/14 laion2b_s12b_b42k (Markov), ViT-L/14 openai (laptop) | Wen et al. (ViT-g/14), Reißinger et al. (CLIP) | 5.5 / 0.9 GB | any |

Open models instead of GPT-4/GPT-4V: the judge and rewriter must run on Markov without API keys, and
the model-level analysis needs access to logits (token entropy) and reasoning traces. The judge
prompt is CopyCat's verbatim; agreement with human labels will be checked on a dev sample before the
judge is used for test numbers.

## Sizing on Markov

| job | GPU | ~time |
|---|---|---|
| SD 1.4, 512 px, 50 steps, batch 4 | 2080 Ti / 4090 | ~2–3 s/image (4090) |
| SDXL / Playground, 1024 px, 50 steps | 4090 / L40S | ~8–10 s/image |
| Wen detector, 1000 prompts × 7 defenses × 4 seeds | 2080 Ti | ~30 min |
| Qwen3-8B rewrites with thinking (≈1k tokens each) | L40S | ~30–40 s/rewrite |
| Qwen2.5-VL-7B judge | 4090 / L40S | ~1 s/image |

Measured locally: SD 1.4 on an Apple M2 (MPS, fp16) takes ~54 s per 512 px image at 50 steps.
