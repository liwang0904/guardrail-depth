# Demo 2 analysis: which baseline works, for whom, and why

All numbers come from `results/demo2/` (regenerate with `python scripts/demo2_report.py`). Intervals are
95% bootstrap intervals over items (`results/demo2/ci.csv`). Test split only.

## Bottom line

No single prompt-level guardrail protects all three kinds of IP. The best baseline depends on what is
being copied, and the reason lines up with how the model is triggered:

| What is copied | Trigger inside the model | What works (plain request) | What fails |
|---|---|---|---|
| Exact training image (SD 1.4) | the exact caption wording | task instruction D1 1.5% [0, 3.7], CoT D3 0.7% [0, 2.2], LLM rewrite D4 0% (no guardrail: 33.1% [22.8, 43.4]) | negative prompt D2b 29.4%, blocklist V0 33.1% |
| Copyrighted character | the concept (name or description) | LLM rewrite + negative prompt D4+D2b 0% (SD 1.4) / 5.3% (SDXL); name+keyword negative prompt D2b-kw 28.2% / 20.5% | templates D1/D2a/D3 67-80%, no better than no guardrail (82.1% / 56.4%) |
| Artist style | the artist's name | negative prompt with the name D2b-name 5.0% [2.0, 8.5] (no guardrail: 31.0%) | templates D1/D2a/D3 32-38% |

Templates beat copying by **disrupting the wording**, not by what they say: random words of the same
length (C1) also give 0% copies, and the memorization signal inside the model drops to the level of
normal prompts under D1 and C1 alike. That works for exact copies, whose trigger is a precise caption,
and does nothing for characters and styles, whose trigger is a concept that survives rewording.

## 1. Which baseline works best, and why

**Exact copies.** On SD 1.4, 33% of images copy their training image without a guardrail; almost all of it
comes from verbatim-memorized prompts (63.2% of VM vs. 2.9% of TM images). D1 and D3 cut copying to 1-2% and
stay at or below 4% under padding and override attacks. The Wen et al. detector explains why: under D1 the
first-step noise magnitude of verbatim prompts falls from 4.49 to 0.94, the level of normal prompts (0.90),
and the detector's AUC falls from 0.988 to 0.675. The copy trigger is switched off, not hidden. C1 does the
same (0.77), so the mechanism is disruption. The negative prompt D2b leaves the user's caption intact and
fails (29.4%). SD 2-base copies less to begin with (9.3%), and the ranking is the same.

**Characters.** Templates do not protect characters at all: on SDXL they are nominally *worse* than no
guardrail (D2a 79.5% vs. D0 56.4%; intervals overlap). Only baselines that act on the concept work: He et
al.'s negative prompt with the character's name and keywords, and the LLM rewrite; combined, they reach 0%
on SD 1.4, consistent with He et al., who also found rewriting plus a negative prompt strongest.

**Style.** Only the negative prompt naming the artist works (31% -> 5%). Templates leave style attribution
unchanged or higher (32-38%).

**Vanilla baseline (V0 blocklist).** It refuses every plain character name (0% by construction), but:
keyword lures get 28% (SD 1.4) and 38% (SDXL) of characters through, the same as no guardrail; it has no
effect on exact copies (it refused 1 of 130 prompts, a news headline that mentions "Black Panther"); and it
over-blocks: 30% of public-domain artist prompts are refused (Monet, Klimt, Munch) because the DALL-E policy
rule is a 1912 cut-off, not copyright status. The research baselines are worth their complexity, but only
the ones matched to the trigger.

## 2. Subgroup and error analysis

**Living artists are protected least.** Under the best style guardrail (D2b-name), every remaining failure is
a living artist: 10 of 80 living-artist prompts are still attributed to the artist vs. 0 of 120 for
deceased-in-copyright and public-domain artists (Fisher exact p < 0.001). The failures are Karla Ortiz, Greg
Rutkowski, Tyler Edlin, Kilian Eng and Hollie Mengert - the artists whose cases motivated the
copyright debate. Relative to no guardrail, living artists keep 10 of 18 attributions; the other groups keep
0 of 44. The group with the most at stake gets the least protection.

**Memorization type.** Verbatim-memorized prompts are where copying happens (63% vs. 3% for template
memorization); every guardrail that works removes most of the VM risk, and D2b leaves 57% of it.

**Character origin.** On SDXL, the LLM rewrite leaves US characters more exposed than Japanese ones (14% vs.
0% detected, mean over attacks), and so does the blocklist (20% vs. 0%). With 29 US and 10 Japanese test
characters and one SDXL seed, these gaps are suggestive, not conclusive.

**What still gets through the best guardrail.** `results/demo2/errors.md` lists every failure: under
D4+D2b, Captain America through a 60-word description (SD 1.4), and Batman and Captain America by plain name on
SDXL; under D4 for exact copies, one verbatim prompt with the override suffix (SSCD 0.53, just over the line).

**The rewriter only protects what it recognizes.** The Qwen3 rewriter named the character in its reasoning
for 44% of misspelled names and 21% of keyword lures; it removed the name from the final rewrite almost
always (0% / 8% leaked), but a description it does not recognize passes through as a description.

## 3. Where the baselines fall short

1. **Cost on benign prompts.** Every guardrail that works lowers adherence on normal captions (CLIP 0.296
   with no guardrail): D1 0.261, D3 0.272, D4 0.233. C1 (0.107) shows the extreme: protection by destroying
   the request. A deployed system pays this cost on every prompt, although only a few are risky.
2. **No guardrail covers all three threats.** The best for exact copies (D1/D3) fails on characters and
   style; the best for characters and style (concept negative prompts) fails on exact copies.
3. **Unequal protection.** Living artists and (on SDXL) US characters are the least protected subgroups.
4. **The vanilla blocklist is brittle in both directions:** lures and keywords pass, public-domain artists
   are refused.

## 4. Proposal for Demo 3: detect inside the model, then apply the matched defense

The results point to a defense that (a) decides *whether* a prompt is risky from signals inside the model,
which survive rewording, and (b) applies the guardrail that matches the trigger only when needed:

- **Exact copies:** use the first-step noise magnitude as the gate (AUC 0.988, F1 0.92 at 1% FPR on SD 1.4;
  AUC 0.994 on SD 2-base). Only flagged prompts get the disruption (D1-style rewording or a perturbed text
  embedding), so benign prompts keep full adherence.
- **Characters and styles:** compare the prompt's CLIP text embedding with embeddings of the protected
  concepts (names, keywords, descriptions). This works on meaning rather than spelling, so lures and
  misspellings that beat the blocklist should still match. A match triggers concept-negative guidance with
  the matched name and keywords - the mechanism that worked best here.
- **Equity check built in:** report every result per subgroup, and tune the gate so living artists are
  protected at least as well as public-domain ones.

Evaluation: the same Demo 2 grid and test split, plus the adaptive attack (Kim et al.'s APGP), comparing
protection, benign adherence and subgroup gaps against the best baseline for each threat.

## Limitations

- Training images are still online for 68 of 130 sampled SD 1.4 prompts (34 of 90 verbatim) and 91 of 100 SD 2
  prompts; copy rates use only those (`results/demo2/coverage.csv`).
- 39 test characters, SDXL with one seed: character differences under about 15 points are within noise.
- Style attribution is CLIP zero-shot among 25 artists; CLIP recognizes public-domain artists more readily
  (44% vs. 22% for living artists with no guardrail), so absolute style numbers are conservative.
- D4 uses He et al.'s released GPT-4 rewrites for names and 60-word lures and Qwen3-8B for keywords and
  misspellings; one character (Link) has no GPT-4 rewrite.
- The detector rows for D2b equal D0 by construction (it reads the positive prompt only).
