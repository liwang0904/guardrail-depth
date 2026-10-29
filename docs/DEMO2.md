# Demo 2: every baseline, every dataset, every model

Demo 1 feedback: "start thinking about what analysis could be derived from these results for demo 2."
Demo 2 answers it with a full grid on the untouched **test split**, a **vanilla baseline** next to the
state-of-the-art ones, **subgroup** breakdowns, and an **error analysis** of what still gets through.

## Baselines

| Code | Baseline | Kind | Source |
|---|---|---|---|
| D0 | no guardrail | reference | - |
| V0 | keyword blocklist: refuse if the request names a CopyCat character or an artist the DALL-E policy declines | **vanilla** (what a practitioner writes first: a list + case-insensitive match) | ours, list from He et al. + DALL-E policy |
| D1 / D2a / D3 | task instruction, negation in the prompt, chain-of-thought template | SOTA (prompt templates) | Reissinger et al. 2025, verbatim |
| D2b(-kw/-name) | negative prompt: rule list, or the character's name + keywords | SOTA | He et al. ICLR'25 |
| D4 (+D2b) | LLM rewrite under the DALL-E policy (+ negative prompt) | SOTA | He et al. ICLR'25 (their GPT-4 rewrites; Qwen3-8B where none were released) |
| C1 | random words, length-matched to D1 | control | ours |
| detector | first-step text-conditional noise magnitude | SOTA (detection) | Wen et al. ICLR'24 |

## Grid (`configs/experiments/demo2.yaml`)

| Job | Model | Dataset (test split) | Attacks | Seeds |
|---|---|---|---|---|
| sd14_memorized | SD 1.4 | 90 VM + 40 TM memorized prompts | plain, padding, override | 2 |
| sd14_controls | SD 1.4 | 100 non-memorized LAION captions | plain | 2 |
| sd2_memorized | SD 2-base | 3 MV + 97 TV memorized prompts | plain | 2 |
| sd14_copycat | SD 1.4 | 39 characters | name, 60-word lure, keyword lure, misspelling | 2 |
| sdxl_copycat | SDXL | 39 characters | same | 1 |
| sd14_style | SD 1.4 | 25 artists x 8 templates | plain, misspelled name | 1 |
| detector | SD 1.4, SD 2-base | all 500 + 219 memorized, 500 controls | plain, padding, override | 4 noise draws |

Items were sampled once with `random.Random(347)` (all VM prompts; random TM / TV / control prompts).

## Metrics

| | Metric | Meaning |
|---|---|---|
| copying | copy rate | % of images with SSCD > 0.5 to the training image (Somepalli et al.) |
| | character detected | % of images the Qwen2.5-VL judge names as the target (He et al. DETECT) |
| | style attributed | % of images CLIP ViT-g/14 attributes to the requested artist among the 25 |
| usefulness | CLIP adherence | image vs. what the user asked for; CONS for characters |
| | refusal rate | share of requests the blocklist refuses (a refusal counts as "no copy") |
| detection | AUC, TPR / precision / F1 at 1% FPR | memorized vs. control prompts |
| responsible AI | subgroup gap | max - min protection across groups (artist legal status, character origin / rights holder, memorization type, duplicate count) |
| | benign cost | adherence lost on the benign controls relative to D0 |
| | attack lift | worst attack minus the plain request, per guardrail |

## Running it on Markov

```bash
source slurm/env.sh
bash slurm/demo2_download.sh     # login node, once (~50 GB)
bash slurm/demo2_submit.sh       # 3 lanes, <= 3 GPUs at a time, ends with scoring + tables
squeue -u $USER                  # watch
```

`slurm/demo2_evaluate.sbatch` scores every run and writes `results/demo2/` (tables, figures,
`headline.json`, `errors.md`) with `scripts/demo2_report.py`; those files are what we commit.
Images and probe arrays stay in course storage.
