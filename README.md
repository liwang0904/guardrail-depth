# guardrail-depth

**How deep do prompt-level copyright guardrails go?** CSDS 347 project (Kent Manion, Lindsey Wang).

Text-to-image services increasingly rely on cheap, inference-time guardrails that rewrite the
user's prompt: task-instruction templates, negative prompts, "chain-of-thought" templates, and
LLM rewriting (Reißinger et al. 2025; He et al. ICLR'25). This repository evaluates that defense
class against a *non-cooperative* user and connects the outcome to what happens inside the model:

* **Track 1 – adversarial extraction.** Do guardrails still hold when the user lures, obfuscates or
  overrides? Exact-copy memorization (500 memorized Stable Diffusion prompts, Wen et al. ICLR'24 /
  NeMo NeurIPS'24) and copyrighted characters (CopyCat, He et al. ICLR'25).
* **Track 2 – style mimicry.** Does a guardrail built for exact copies reduce style replication of
  living artists, and is protection equal across artist groups?
* **Model level.** Text-conditional noise magnitude (Wen et al.), cross-attention to guardrail
  vs. user tokens (Ren et al. ECCV'24, Chen et al. ICLR'25), value-neuron outliers (NeMo), the
  77-token CLIP window, and reasoning traces / token entropy of LLM rewriters.

The steps below were run from a clean checkout on macOS (Apple M2).

---

## 1. Installation walkthrough

### 1.1 What you need

| | Laptop pilot | Full experiments |
|---|---|---|
| GPU | Apple Silicon (MPS) or any NVIDIA GPU with ≥ 8 GB | Markov: RTX 2080 Ti 11 GB for SD 1.x/2.x; RTX 4090 24 GB or L40S 48 GB for SDXL / Playground v2.5 and the 7–8B judge / rewriter |
| Disk | ~6 GB (SD 1.4 fp16, CLIP ViT-L/14, SSCD, data) | ~70 GB of weights + images → course storage, **not** the 25 GB home quota |
| Python | 3.11 | 3.11 |
| Time | ~55 s/image at 50 steps on an M2 | ~2–3 s/image (SD 1.4, RTX 4090) |

Pinned versions (`requirements.txt`): torch 2.8.0 (CUDA 12.6 wheels on Linux), diffusers 0.35.1,
transformers 4.56.2, accelerate 1.10.1, huggingface-hub 0.35.3, open-clip-torch 2.32.0,
numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2. We deliberately pin a 2025 stack: newer torch
releases default to CUDA 13 wheels, which need a newer driver than Markov's CUDA 12.x modules.

### 1.2 Install and check

```bash
git clone https://github.com/liwang0904/guardrail-depth.git guardrail-depth && cd guardrail-depth
conda create -y -n guardrail-depth -c conda-forge --override-channels python=3.11 pip
conda activate guardrail-depth
# Linux + NVIDIA only: CUDA wheels first
#   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python scripts/check_env.py                    # versions, accelerator, storage
python -m pytest -q                            # 26 tests; downloads ~20 MB of tiny test models
python scripts/prepare_data.py                 # reference data (pinned commits, SHA-256 verified) -> data/manifests/
```

`check_env.py` should end with `environment matches requirements.txt`, and `pytest` with
`26 passed` (the GPU regression test is skipped on CPU-only machines).

### 1.3 Run a baseline

```bash
python scripts/prepare_data.py --skip-raw --gt-images sd1 --gt-split dev   # ground-truth LAION images (dev split)
python scripts/run_generation.py --name demo --ids 202 --defenses none,task_instruction --steps 25
python scripts/evaluate.py outputs/adhoc/demo
```

The first run downloads SD 1.4 (fp16, ~2.7 GB), CLIP ViT-L/14 and the SSCD copy detector into the
Hugging Face cache. `run_generation.py` writes images, per-image records and probe arrays to
`outputs/adhoc/<name>`; `evaluate.py` adds SSCD-to-training-image, CLIP adherence and seed diversity.

### 1.4 On Markov

Use the same steps from course storage, so weights stay out of the 25 GB home quota:

```bash
cd /mnt/vstor/courses/csds447/$USER
export HF_HOME=$PWD/hf_cache GD_DATA_ROOT=$PWD/data GD_OUTPUT_ROOT=$PWD/outputs PIP_CACHE_DIR=$PWD/pip_cache
srun -A csds447 -p markov_gpu -C gpu2080 --gres=gpu:1 -c 4 --mem=16G -t 01:00:00 --pty bash
```

Then follow 1.2 (with the CUDA line) and 1.3 inside that session, and `exit` when done.

### 1.5 Troubleshooting (things we hit)

* **Stable Diffusion 2.x repos are gone.** `stabilityai/stable-diffusion-2*` now return HTTP 401;
  `configs/models.yaml` points at the `sd2-community/*` mirrors, pinned by commit.
* **Black images / NaN on Apple MPS.** diffusers' explicit attention path (`baddbmm(beta=0)`) reads
  uninitialized memory on MPS. Our attention probe computes scores with `bmm` in fp32; the
  generator raises instead of logging black images. Covered by a regression test.
* **`torch_dtype is deprecated`** warnings come from transformers 4.56 inside diffusers and are harmless.

---

## 2. Data

`python scripts/prepare_data.py` downloads every source at a pinned commit, checks it against
`data/SOURCES.lock.json`, and writes normalized manifests (committed) with a fixed dev/test split
(seed 347, dev used for pilots and tuning only).

| Manifest | Source (closest reference) | Size | Split dev/test | Attributes used in the analysis |
|---|---|---|---|---|
| `memorized_sd1` | Wen et al. ICLR'24 + NeMo NeurIPS'24 (from Webster 2023) | 500 prompts | 100 / 400 | memorization type VM/TM (NeMo) and MV/RV/TV (Webster), LAION duplicate count, reference SSCD, CLIP token length, ground-truth URL |
| `unmemorized_laion` | NeMo NeurIPS'24 | 500 prompts | 100 / 400 | control set from the same caption distribution |
| `memorized_sd2` | Webster 2023 | 219 prompts | 44 / 175 | MV/TV, duplicate count |
| `copycat_characters` | He et al. ICLR'25 (CopyCat) | 50 characters | 11 / 39 | 60-word lure, LM / LAION / embedding keywords, GPT-4 rewrites, key entity, origin region (US 36 / JP 13 / CA 1), rights holder, public-domain notes |
| `artists`, `style_prompts` | Pham et al. ICLR'24, ESD, DALL·E policy, WikiArt | 25 artists × 10 prompts | 2 / 8 templates | living vs. deceased-in-copyright vs. public domain, DALL·E pre-1912 rule, WikiArt reference availability |

Ground-truth training images are fetched from their LAION URLs on demand (77 of 100 dev images were
still online on 2026-10-14) and are never committed.

## 3. Models

All weights are pinned by commit in **[configs/models.yaml](configs/models.yaml)**; settings follow
the closest prior work.

| Role | Model | From |
|---|---|---|
| Subject T2I (memorization) | Stable Diffusion v1.4 (DDIM, 50 steps, cfg 7.5) | Wen et al., NeMo, Pham et al. |
| Subject T2I | SD 2-base / 2.1 (mirror) | Webster 2023; Reißinger et al. 2025 |
| Subject T2I (characters) | Playground v2.5, SDXL (50 steps, cfg 3) | He et al. ICLR'25 |
| Judge (DETECT/CONS) | Qwen2.5-VL-7B-Instruct (replaces GPT-4V / VQAScore) | He et al. protocol |
| LLM rewriter | Qwen3-8B (thinking traces), Qwen2.5-7B; Qwen3-1.7B locally | He et al. used GPT-4 |
| Copy detection | SSCD `sscd_disc_mixup` | Somepalli et al., Wen et al., NeMo |
| Alignment / attribution | OpenCLIP ViT-g/14 (Markov), ViT-L/14 (laptop) | Wen et al.; Reißinger et al. |

## 4. Baselines

| Baseline | Paper | Entry point |
|---|---|---|
| V0 keyword blocklist (vanilla baseline) | ours (CopyCat names + DALL·E artist policy) | `defenses.py` `blocklist` |
| D1–D3 prompt templates (task instruction, in-prompt negation, CoT) | Reißinger et al. 2025 | `defenses.py` (verbatim Table 1) |
| Negative-prompt buffer (name / name + keywords) | He et al. ICLR'25; SD-NP in Pham et al. | `defenses.py` `negbuf_*` |
| LLM rewrite (+ negative prompt) | He et al. ICLR'25 (DALL·E policy prompt) | `rewriter.py`, `llm_rewrite*` defenses |
| Indirect-anchoring lures (60-word, keywords) | He et al. ICLR'25 | `attacks.py` `lure_*` |
| Memorization detector (first-step noise magnitude) | Wen et al. ICLR'24 | `probes.wen_detection_scores` |
| Value-neuron outliers | Hintersdorf et al. NeurIPS'24 | `probes.neuron_activations` |
| DETECT / CONS judge | He et al. ICLR'25 | `judge.py` |

Defenses and attacks are chosen on the command line, e.g.
`--attacks direct,lure_desc60 --defenses none,task_instruction,cot,negbuf_name_kw` with
`--dataset copycat_characters`.

## 5. Demo 2: full baseline grid

`configs/experiments/demo2.yaml` runs every baseline (vanilla keyword blocklist V0, Reissinger et al.
templates, He et al. negative prompts and LLM rewrite, the random-word control) on SD 1.4, SD 2-base and
SDXL across the memorized, control, CopyCat and style test sets. See [docs/DEMO2.md](docs/DEMO2.md) for
the grid and metrics, [results/demo2/](results/demo2/) for the committed tables and figures, and
[docs/DEMO2_ANALYSIS.md](docs/DEMO2_ANALYSIS.md) for which baseline works for whom, and the Demo 3 plan.

## 6. Repository layout

```
configs/models.yaml, data_sources.yaml         pinned models and data
data/manifests/*.csv, data/SOURCES.lock.json   committed manifests + checksums
src/guardrail_depth/                           defenses, attacks, data, models, generation, probes, metrics, rewriter, judge
scripts/                                       check_env, prepare_data, run_generation, evaluate, demo2_report, ...
slurm/                                         Markov jobs (demo2_submit.sh runs the whole Demo 2 grid)
results/demo2/                                 committed Demo 2 tables and figures
tests/                                         unit + tiny-model integration tests
```

## 7. Responsible handling

* Ground-truth LAION images and all generations stay out of git (`.gitignore`); they are used only
  to measure similarity.
* We do not collect portfolios of living artists. Their style is measured with zero-shot attribution
  and similarity to the model's own unguarded outputs; reference images exist only for
  public-domain and deceased artists (WikiArt, non-commercial research terms).
* The SD 1.x/2.x NSFW checker is disabled, as in all prior memorization work, because it blacks out
  unrelated images; outputs are not shared outside the team.
