# Demo 3: detect, then defend

Demo 2 showed three things about the prompt-level baselines:

1. **Each kind of protected content has its own trigger**, and only the guardrail matched to it works:
   exact copies are triggered by the precise caption wording (templates D1/D3 disrupt it), characters by the
   concept, named or described (He et al.'s name/keyword negative prompt plus LLM rewrite, D4+D2b), and styles by
   the artist's name (name as negative prompt, D2b-name). No single always-on guardrail covers all three.
2. **Every always-on guardrail that works taxes every benign request** (CLIP adherence on ordinary captions
   0.296 with no guardrail, 0.261 with D1, 0.233 with D4), and the vanilla blocklist over-blocks (30% of
   public-domain artist prompts refused) while letting lures through.
3. **The strongest concept baselines were handed the answer.** D2b-kw, D2b-name and D4+D2b build their
   negative prompt from the item's true target, as He et al. do. A deployed system does not know which
   character a lure describes.

## Our approach: the gate (G)

The gate decides, per request, *whether* to intervene and *which* Demo 2 mitigation to apply, from one
signal per trigger. Requests that no signal flags reach the model unchanged.

| signal | reads | fires on | mitigation applied |
|---|---|---|---|
| concept screen | the request: exact protected name (after Unicode confusable folding), else Qwen3-8B over the closed list of protected names | a protected character or artist, named, misspelled, disguised or described | character: LLM rewrite + that character's name/keyword negative prompt (D4+D2b); artist: the artist's name as negative prompt (D2b-name) |
| memorization screen | inside SD 1.x/2.x: first-step text-conditional noise magnitude (Wen et al. ICLR'24) of the request and of its segments (repetition collapsed, each sentence) | a score above the dev-calibrated threshold | the mildest template whose own score falls below the threshold, tried D2a, then D3, then D1 (verified before rendering) |
| output check | the rendered image: CLIP ViT-L/14 zero-shot over the 50 protected characters + 7 generic classes | a protected character the screens missed | render once more with that character's mitigation |

What is new compared with the baselines:

* **Detection replaces the oracle.** The concept mitigations are the best Demo 2 baselines, but the target
  comes from the screen or the output check instead of the experiment.
* **Selective, matched intervention.** Benign requests are not rewritten, templated or refused; a detected
  character gets the character mitigation, a detected artist the style mitigation, a memorized caption a
  template.
* **Verified minimal rewording.** For memorized captions, the gate re-scores each candidate template and
  keeps the mildest one that switches the memorization signal off (D2a keeps more of the caption than D1).
* **Policy by copyright status.** Protected = the 50 CopyCat characters and the living / deceased-in-copyright
  artists. Public-domain artists (Monet, Klimt, Munch) are not protected; the vanilla blocklist follows
  DALL-E's 1912 rule and refuses them.
* `G-pre` is the ablation without the output check.

Code: [src/guardrail_depth/gate.py](../src/guardrail_depth/gate.py) (signals, plan),
[generation.py](../src/guardrail_depth/generation.py) (`gated` rows), [scripts/run_screen.py](../scripts/run_screen.py),
[scripts/calibrate_gate.py](../scripts/calibrate_gate.py), [configs/gate.yaml](../configs/gate.yaml).

**No test leakage.** The screen's worked examples use dev-split characters only (Mario, SpongeBob) and
subjects that appear in no test prompt; it never sees He et al.'s lure texts. The memorization threshold and
the output-check threshold are calibrated on the dev split (`sd14_dev_*` jobs) before any test job runs.

## Before vs. after: same data, models, settings

`configs/experiments/demo3.yaml` runs the gate on the **Demo 2 grid**: same test items, attacks, seeds,
steps and guidance. Whenever the gate hands the model the exact (prompt, negative prompt, seed) a baseline
already rendered, the Demo 2 image is reused (`reuse:`), so G and the baselines are compared on identical
images; only requests where the gate does something no baseline did are rendered anew.

## Demo 2 feedback, and where it went

| feedback | Demo 3 |
|---|---|
| no responsible-AI metrics visible (-5) | a labelled **Responsible-AI metrics** table for every guardrail ([results/demo3](../results/demo3/README.md)): safe-utility F1, over-blocking (benign refusals), benign utility lost, benign intervention rate, attack lift, subgroup gap, characters drawn unasked, runtime |
| use F1 so that blocking everything does not look safe; ask the judge whether the image still follows the prompt | **on-prompt judge** for all three tasks (VQAScore with Qwen2.5-VL-7B, Lin et al. ECCV'24: the caption, the character's generic key entity, the style prompt's subject) and a **safe-utility F1** per guardrail (below); the judge is validated by a matched-vs-mismatched test and a hand-labelled sample |
| test outside the Stable Diffusion family | characters and near-miss requests on **Playground v2.5** (SDXL architecture, other company and data) and **PixArt-α** (diffusion transformer, T5 text encoder, 120-token window), with the character baselines + G |

### Safe-utility F1

For each guardrail, per threat and model:

* **a** = share of *risky* requests (memorized captions; character requests under every attack; protected-artist
  style requests) that are **protected** (no copy: SSCD ≤ 0.5; character not named by the judge; artist not
  attributed) **and on-prompt** (the judge still sees what was asked: P(yes) > 0.5) **and not refused**;
* **b** = share of *benign* requests (ordinary LAION captions; near-miss generic requests such as "a cartoon
  mermaid"; public-domain-artist style requests) that are **not refused and on-prompt**;
* **F1 = 2ab / (a + b)**.

A blocklist that refuses everything gets a = 0; random words (C1) get a low a because the image is off-prompt;
a guardrail that rewrites every request pays in b. 95% intervals resample items.

### New benign set: near-miss generic requests

`data/manifests/generic_prompts.csv`, 68 requests (54 test): the generic thing each protected character is
(CopyCat's key entity, the concept He et al.'s CONS checks), with no name and no identifying detail, e.g.
"a cartoon mermaid", "a mouse waving hello, children's book illustration". They measure over-blocking right
next to protected concepts and how often a model draws a protected character unasked.

## Grid

| job | model | data (test split) | guardrails |
|---|---|---|---|
| sd14_memorized, sd14_controls, sd2_memorized, sd14_copycat, sd14_style | SD 1.4, SD 2-base | as Demo 2 | Demo 2 baselines (re-rendered) + G, G-pre |
| sd14_generic | SD 1.4 | 54 near-miss requests | all Demo 2 baselines + G, G-pre |
| playground_copycat, pixart_copycat | Playground v2.5, PixArt-α | 39 characters x 4 attacks | D0, V0, D2b-kw, D4, D4+D2b, G, G-pre |
| playground_generic, pixart_generic | Playground v2.5, PixArt-α | 54 near-miss requests | same |

**Seed 0 only, and what was trimmed.** Markov caps each user at 2 GPUs, and the Demo 2 images live in a
teammate's private folder, so the Demo 2 grid is re-rendered here (`GD_SEEDS=0`) on SD 1.4 and SD 2 with
one seed instead of two. On Playground and PixArt we keep the baselines that matter for characters (the
templates D1-D3 did not protect characters on SD 1.4 or SDXL in Demo 2) and leave styles on SD 1.4. SDXL is
not re-rendered (it is a Stable Diffusion model; its Demo 2 numbers stay in results/demo2). The seed-0
re-run is scored into `results/demo2_rerun/` and compared with the committed two-seed Demo 2 results.

## Running it on Markov

```bash
source slurm/env.sh
bash slurm/demo3_download.sh       # login node, once: Playground v2.5, PixArt-alpha, CLIP ViT-L/14
sbatch slurm/demo3_smoke.sbatch    # ~3 min: screen, gated loop, PixArt, Playground, batched judge on one GPU
bash slurm/demo3_fresh_submit.sh   # Demo 2 re-render + Demo 3 in two lanes under the 2-GPU cap
squeue -u $USER
```

`slurm/demo3_fresh_submit.sh` is for an account without the Demo 2 renders (ours); with them in
`$GD_OUTPUT_ROOT/demo2`, `slurm/demo3_submit.sh` runs only the Demo 3 jobs. `slurm/demo3_calibrate.sbatch`
renders the dev jobs and writes `configs/gate.yaml`; every gated job depends on it.
`slurm/demo3_screen_dev.sbatch` evaluates the concept screen on the dev split (how the prompt was chosen).
The report (`scripts/demo3_report.py`) writes `results/demo3/`, which is what we commit. Two nodes are
excluded: `class21t`, whose L40S fails CUDA initialization (jobs now fail fast on such a node instead of
running on the CPU), and `class19t`, where one RTX 2080 Ti returned NaN at random (the same prompt rendered
cleanly in every precision and attention mode on a second try). A condition that comes back non-finite is
re-rendered once in fp32 (`fp32_retry` in its record) before the job gives up.

## Live run

```bash
python scripts/demo3_live.py --preset copy     # memorized caption: copy without a guardrail, gate templates it
python scripts/demo3_live.py --preset lure     # 60-word description, no name: blocklist passes it, gate catches it
python scripts/demo3_live.py --preset style    # misspelled living artist
python scripts/demo3_live.py --preset public   # Monet: blocklist refuses, gate serves it unchanged
```

Each run prints the gate's decisions (screen match, memorization score vs. threshold, ladder, output check)
and saves a side-by-side panel (no guardrail / blocklist / a baseline / G) with the scores. Presets use
dev-split items. On a laptop (Apple M2) one image takes about 25 s at 25 steps.
