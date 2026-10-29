# Baselines

Every baseline comes from one of the closest prior works and is implemented with that work's
prompts, settings and metric definitions. "Pilot" = what has been run so far on the dev split on a
laptop GPU (Apple M2); "Markov" = the full grid in `configs/experiments/` and `slurm/`.

## Defenses (the class under test)

| code | defense | source | implementation |
|---|---|---|---|
| D0 | raw prompt | proposal | — |
| D0r | "Generate an image of {caption}." | Reißinger et al. 2025, Table 1 (their baseline) | `defenses.REISSINGER_BASELINE` |
| D1 | task instruction template | Reißinger et al. 2025, Table 1 | `defenses.TASK_INSTRUCTION` (verbatim) |
| D2a | negation inside the prompt | Reißinger et al. 2025, Table 1 | `defenses.NEGATIVE_INPROMPT` (verbatim) |
| D2b | CFG negative-prompt buffer: rule list / "copyrighted character" / name / name + 5 EmbeddingSim + 5 LAION keywords | proposal D2; He et al. ICLR'25 Table 1; SD-NP in Pham et al. ICLR'24 | `negbuf_rules`, `negbuf_generic`, `negbuf_name`, `negbuf_name_kw` |
| D3 | chain-of-thought template | Reißinger et al. 2025, Table 1 | `defenses.CHAIN_OF_THOUGHT` (verbatim) |
| D4 | LLM rewrite under the DALL·E policy (+ D2b) | He et al. ICLR'25 (`get_dalle_rewrite.py`, verbatim system prompt) | GPT-4 rewrites from the CopyCat release, or `scripts/run_rewriter.py` with Qwen |
| C1 | random words, length-matched to D1 | random-token-addition control (Somepalli et al. 2023; Wen et al. ICLR'24) | `defenses.random_token_wrap` |

Reißinger et al. report on SD 2 that D3 cuts generations above CLIP-similarity 0.85 from 41.4 % to
9.6 %, D1 to 20.4 %, while D2a is not significant (34.9 %). He et al. report on Playground v2.5 that
rewriting alone lowers DETECT from 30.3 to 14.3 of 50 characters and rewriting + name/keyword negative
prompts to 4.3. C1 tests whether the templates work through their *meaning* or merely by
perturbing a memorized caption.

## Attacks

| code | attack | source |
|---|---|---|
| A0 | direct request (name / memorized caption) | — |
| A1 | description lure (60 words) and 5-keyword indirect anchors (LM, LAION co-occurrence, embedding) | He et al. ICLR'25 (released prompts) |
| A2 | homoglyph, typo (TextBugger-style), leetspeak, phonetic respelling of the protected name | proposal; PromptRobust (Zhu et al. 2023) |
| A3 | instruction-override suffix; truncation padding that fills CLIP's 75-token window | proposal |
| A4 | APGP: LLM optimizer + score (image sim, text sim, keyword penalty) against the guarded pipeline | Kim et al. ICML'24 workshop (`scripts/run_apgp.py`) |
| A5 | Concept Inversion against the negative-prompt buffer (white-box upper bound) | Pham et al. ICLR'24 (authors' `np/` code, SD v1.4) |

## Measurement baselines

| what | source | implementation |
|---|---|---|
| exact copy: SSCD to the training image, copy if > 0.5 | Somepalli et al. CVPR'23; Wen et al.; NeMo | `metrics.SSCD` (NeMo preprocessing) |
| DETECT (GPT-4V → Qwen2.5-VL-7B, verbatim question) and CONS (VQAScore-style P(yes)) | He et al. ICLR'25 | `judge.py`, `scripts/run_judge.py` |
| prompt adherence: CLIP image–text | Reißinger et al. 2025 | `metrics.CLIPScorer` |
| memorization detector: first-step text-conditional noise magnitude | Wen et al. ICLR'24 | `probes.wen_detection_scores`, `scripts/run_detection.py` |
| memorization neurons: value-projection z-scores | Hintersdorf et al. NeurIPS'24 | `probes.neuron_activations`, `scripts/run_neurons.py` |
| cross-attention over tokens (entropy, BOS/summary/guardrail/user mass, end-token maps) | Ren et al. ECCV'24; Chen et al. ICLR'25 | `probes.CrossAttentionProbe` |

## Validation of the re-implementations

* **NeMo activations.** On 200 non-memorized LAION prompts our value-neuron activations reproduce
  NeMo's released reference statistics: mean z = −0.002, std z = 1.002, per-layer mean activation
  within 0.5 % of theirs.
* **Probes do not change the computation.** With the attention probe attached the generated image is
  pixel-identical to the unprobed pipeline (tested on GPU fp16).
* **Wen et al. detector.** Results and AUC are in `outputs/detection/` (see the milestone report).

## Pilot results (dev split, SD v1.4 on an Apple M2)

| baseline | result |
|---|---|
| Wen et al. detector, raw prompts | AUC 0.99 verbatim-memorized vs. control (0.90 incl. template-memorized; paper: 0.96 at n = 1) |
| same, under D1 / D2a / D3 / C1 | memorized signal 4.68 → 1.02 / 1.51 / 0.76 / 1.01 (controls ≈ 1.0); D2b leaves it at 4.68 |
| exact copies (SSCD > 0.5), verbatim prompts | D0 100 %, D2b 100 %, D1 17 %, D2a 17 %, D3 0 %, C1 0 % |
| CLIP prompt adherence | D0 0.244, D1 0.251, D2a 0.254, D3 0.225, C1 0.141 |
| CopyCat (CLIP proxy, 5 characters) | name: D0 5/5, D1 4/5, D4+D2b 0/5; 60-word lure: D0 5/5, D1 3/5, D2b-kw 1/5, D4+D2b 1/5 |
| CLIP 77-token window | lure + D1: guardrail tail fully truncated in 100 % of characters; D3 truncated on 100 % of memorized captions |
| GPT-4 rewrites (He et al. release) | keep 3.1 of ~9 identifying keywords on average; 100 % keep at least one |
| Qwen3-1.7B rewriter with reasoning | recognizes the character in 5/5 name requests, 0/5 lures; lure rewrites keep 2.4 keywords at lower entropy |
| style attribution (CLIP, public-domain artists) | D0 0.70, D1 0.69, D2a 0.83, D3 0.58, negative buffer with name < 0.01; living artists 0.03 even unguarded |
| APGP (2 characters, 2 rounds) | no gain over He et al.'s lure seed; 10 of 12 candidates edited only text past the 75-token window |
| NeMo neurons | reproduce NeMo's statistics (z mean −0.002, s.d. 1.002); outlier counts do not separate memorized from control |

Raw files: `outputs/analysis/token_window.csv`, `outputs/detection/*/summary.csv`, `outputs/neurons/*`,
`outputs/pilot_local*/*/metrics.csv`, `outputs/rewrites/*.csv`, `outputs/apgp/*/apgp_log.json`. Figures and
LaTeX tables are generated by `scripts/make_figures.py` and `scripts/report_tables.py`.
