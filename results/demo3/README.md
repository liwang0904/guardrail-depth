# Demo 3 results: detect-then-defend gate (G) vs. the Demo 2 baselines

Same test items, attacks and settings as Demo 2, seed 0 (the Demo 2 grid was re-rendered with one seed under the class GPU cap). † = the experiment handed this baseline the protected target (He et al.'s setup); a deployed system has to detect it, which is what G does. Protection numbers are failure rates (lower = better); F1 and served rates are higher = better. Ablations: G-pre = G without the output check; G-rw = G-pre with the LLM rewrite alone (D4) as the character mitigation, assembled from existing renders (the D4 image of the same request and seed) and added after the test results, so G stays the method fixed before the test run.

## Before vs. after

| threat       | model           | role                            | guardrail                                               | plain   | worst_attack   | mean_attacks   | risky_on_prompt   | benign_served   | benign_refused   |   su_f1 | f1_ci        |   sec_per_request |
|:-------------|:----------------|:--------------------------------|:--------------------------------------------------------|:--------|:---------------|:---------------|:------------------|:----------------|:-----------------|--------:|:-------------|------------------:|
| exact copies | SD 1.4          | no guardrail                    | D0 none                                                 | 33%     | 33%            | 22%            | 72%               | 66%             | 0%               |   0.592 | [0.52, 0.66] |             5.161 |
| exact copies | SD 1.4          | vanilla baseline                | V0 blocklist (vanilla)                                  | 33%     | 33%            | 22%            | 72%               | 66%             | 0%               |   0.592 | [0.52, 0.66] |             5.104 |
| exact copies | SD 1.4          | most protective Demo 2 baseline | D4 LLM rewrite                                          | 0%      | 1%             | 1%             | 56%               | 59%             | 0%               |   0.574 | [0.49, 0.64] |             6.745 |
| exact copies | SD 1.4          | best Demo 2 baseline (F1)       | D2a negation in prompt                                  | 7%      | 7%             | 5%             | 65%               | 63%             | 0%               |   0.624 | [0.56, 0.69] |             5.168 |
| exact copies | SD 1.4          | ours (ablation)                 | G-pre gate, no output check (ablation)                  | 6%      | 6%             | 4%             | 66%               | 65%             | 0%               |   0.639 | [0.56, 0.70] |             6.537 |
| exact copies | SD 1.4          | ours                            | G gate (ours)                                           | 6%      | 6%             | 4%             | 66%               | 65%             | 0%               |   0.639 | [0.56, 0.70] |             6.567 |
| exact copies | SD 2-base       | no guardrail                    | D0 none                                                 | 8%      | 8%             | 8%             | 79%               |                 |                  | nan     |              |             4.852 |
| exact copies | SD 2-base       | vanilla baseline                | V0 blocklist (vanilla)                                  | 8%      | 8%             | 8%             | 79%               |                 |                  | nan     |              |             4.852 |
| exact copies | SD 2-base       | best Demo 2 baseline            | D1 task instruction                                     | 0%      | 0%             | 0%             | 46%               |                 |                  | nan     |              |             4.868 |
| exact copies | SD 2-base       | ours (ablation)                 | G-pre gate, no output check (ablation)                  | 0%      | 0%             | 0%             | 35%               |                 |                  | nan     |              |             6.086 |
| exact copies | SD 2-base       | ours                            | G gate (ours)                                           | 0%      | 0%             | 0%             | 35%               |                 |                  | nan     |              |             6.119 |
| characters   | SD 1.4          | no guardrail                    | D0 none                                                 | 79%     | 79%            | 37%            | 45%               | 98%             | 0%               |   0.415 | [0.32, 0.50] |             5.449 |
| characters   | SD 1.4          | vanilla baseline                | V0 blocklist (vanilla)                                  | 0%      | 38%            | 17%            | 29%               | 98%             | 0%               |   0.374 | [0.29, 0.45] |             4.076 |
| characters   | SD 1.4          | most protective Demo 2 baseline | D4+D2b† LLM rewrite + neg. prompt                       | 0%      | 3%             | 1%             | 34%               | 59%             | 0%               |   0.428 | [0.34, 0.51] |             7.042 |
| characters   | SD 1.4          | best Demo 2 baseline (F1)       | D4 LLM rewrite                                          | 18%     | 18%            | 9%             | 50%               | 74%             | 0%               |   0.576 | [0.49, 0.66] |             7.036 |
| characters   | SD 1.4          | ours (ablation)                 | G-pre gate, no output check (ablation)                  | 0%      | 10%            | 5%             | 36%               | 91%             | 0%               |   0.494 | [0.40, 0.59] |             8.306 |
| characters   | SD 1.4          | ours (ablation)                 | G-rw gate, rewrite-only character mitigation (ablation) | 18%     | 18%            | 13%            | 48%               | 93%             | 0%               |   0.593 | [0.49, 0.68] |             8.323 |
| characters   | SD 1.4          | ours                            | G gate (ours)                                           | 0%      | 5%             | 2%             | 34%               | 91%             | 0%               |   0.494 | [0.39, 0.59] |             8.825 |
| characters   | Playground v2.5 | no guardrail                    | D0 none                                                 | 67%     | 67%            | 33%            | 55%               | 98%             | 0%               |   0.553 | [0.44, 0.65] |             6.262 |
| characters   | Playground v2.5 | vanilla baseline                | V0 blocklist (vanilla)                                  | 0%      | 33%            | 16%            | 38%               | 98%             | 0%               |   0.469 | [0.38, 0.56] |             4.656 |
| characters   | Playground v2.5 | most protective Demo 2 baseline | D4+D2b† LLM rewrite + neg. prompt                       | 11%     | 11%            | 5%             | 53%               | 87%             | 0%               |   0.653 | [0.56, 0.73] |             7.866 |
| characters   | Playground v2.5 | best Demo 2 baseline (F1)       | D4 LLM rewrite                                          | 16%     | 23%            | 11%            | 55%               | 89%             | 0%               |   0.663 | [0.57, 0.74] |             7.845 |
| characters   | Playground v2.5 | ours (ablation)                 | G-pre gate, no output check (ablation)                  | 10%     | 15%            | 9%             | 53%               | 98%             | 0%               |   0.663 | [0.56, 0.75] |             7.628 |
| characters   | Playground v2.5 | ours (ablation)                 | G-rw gate, rewrite-only character mitigation (ablation) | 15%     | 21%            | 13%            | 55%               | 98%             | 0%               |   0.674 | [0.57, 0.76] |             7.612 |
| characters   | Playground v2.5 | ours                            | G gate (ours)                                           | 10%     | 10%            | 7%             | 53%               | 96%             | 0%               |   0.664 | [0.57, 0.75] |             8.464 |
| characters   | PixArt-α        | no guardrail                    | D0 none                                                 | 46%     | 46%            | 26%            | 42%               | 96%             | 0%               |   0.474 | [0.37, 0.57] |             1.312 |
| characters   | PixArt-α        | vanilla baseline                | V0 blocklist (vanilla)                                  | 0%      | 36%            | 14%            | 32%               | 96%             | 0%               |   0.413 | [0.32, 0.50] |             0.968 |
| characters   | PixArt-α        | most protective Demo 2 baseline | D4+D2b† LLM rewrite + neg. prompt                       | 3%      | 8%             | 3%             | 46%               | 94%             | 0%               |   0.611 | [0.51, 0.70] |             2.888 |
| characters   | PixArt-α        | best Demo 2 baseline (F1)       | D4 LLM rewrite                                          | 29%     | 29%            | 14%            | 52%               | 96%             | 0%               |   0.644 | [0.54, 0.73] |             2.885 |
| characters   | PixArt-α        | ours (ablation)                 | G-pre gate, no output check (ablation)                  | 3%      | 15%            | 5%             | 45%               | 94%             | 0%               |   0.590 | [0.48, 0.68] |             2.654 |
| characters   | PixArt-α        | ours (ablation)                 | G-rw gate, rewrite-only character mitigation (ablation) | 28%     | 28%            | 15%            | 49%               | 94%             | 0%               |   0.614 | [0.51, 0.70] |             2.653 |
| characters   | PixArt-α        | ours                            | G gate (ours)                                           | 3%      | 8%             | 3%             | 44%               | 94%             | 0%               |   0.590 | [0.48, 0.69] |             2.893 |
| styles       | SD 1.4          | no guardrail                    | D0 none                                                 | 22%     | 22%            | 15%            | 63%               | 64%             | 0%               |   0.591 | [0.53, 0.65] |             5.584 |
| styles       | SD 1.4          | vanilla baseline                | V0 blocklist (vanilla)                                  | 0%      | 8%             | 4%             | 32%               | 54%             | 15%              |   0.383 | [0.34, 0.42] |             3.600 |
| styles       | SD 1.4          | most protective Demo 2 baseline | D2b-name† negative prompt (name)                        | 8%      | 8%             | 6%             | 60%               | 62%             | 0%               |   0.583 | [0.52, 0.64] |             5.577 |
| styles       | SD 1.4          | best Demo 2 baseline (F1)       | D4 LLM rewrite                                          | 16%     | 16%            | 14%            | 62%               | 67%             | 0%               |   0.599 | [0.54, 0.65] |             7.179 |
| styles       | SD 1.4          | ours (ablation)                 | G-pre gate, no output check (ablation)                  | 8%      | 8%             | 7%             | 61%               | 64%             | 0%               |   0.588 | [0.53, 0.65] |             5.988 |
| styles       | SD 1.4          | ours                            | G gate (ours)                                           | 8%      | 8%             | 7%             | 61%               | 64%             | 0%               |   0.588 | [0.53, 0.65] |             6.047 |

## Responsible-AI metrics

| metric | what it measures |
|---|---|
| safe-utility F1 | harmonic mean of (a) risky requests protected AND still on-prompt AND not refused and (b) benign requests served (not refused, on-prompt); blocking everything or drawing something arbitrary scores 0 |
| benign refused | over-blocking: benign requests refused (ordinary captions, near-miss generic requests, public-domain styles) |
| benign utility lost | served rate on benign requests lost relative to no guardrail (percentage points) |
| benign intervention | share of benign requests the guardrail changed at all |
| attack lift | worst attack minus plain request (robustness to a non-cooperative user) |
| subgroup gap | max - min failure rate across character origin / rights holder, artist group, memorization type ([subgroup_gaps.csv](subgroup_gaps.csv)) |
| drawn unasked | benign near-miss requests on which the judge finds a protected character |
| runtime | seconds per risky request (and per benign request), including the gate's screening, LLM rewrite and any second render |

| threat       | model           | guardrail   | name                                               |   safe_utility_f1 | benign_refused   | benign_utility_lost   | benign_intervention   | drawn_unasked   | attack_lift   | worst_attack   | subgroup_gap   |   sec_per_request |   runtime_vs_D0 |   sec_per_benign_request |
|:-------------|:----------------|:------------|:---------------------------------------------------|------------------:|:-----------------|:----------------------|:----------------------|:----------------|:--------------|:---------------|:---------------|------------------:|----------------:|-------------------------:|
| exact copies | SD 1.4          | C1          | random words                                       |             0.135 | 0%               | 56%                   | 100%                  |                 | 0%            | plain request  | 0%             |             5.167 |           1.001 |                    5.254 |
| exact copies | SD 1.4          | D0          | none                                               |             0.592 | 0%               | 0%                    | 0%                    |                 | 0%            | plain request  | 41%            |             5.161 |           1.000 |                    5.227 |
| exact copies | SD 1.4          | D1          | task instruction                                   |             0.511 | 0%               | 14%                   | 100%                  |                 | 1%            | padding        | 3%             |             5.156 |           0.999 |                    5.217 |
| exact copies | SD 1.4          | D2a         | negation in prompt                                 |             0.624 | 0%               | 3%                    | 100%                  |                 | 0%            | plain request  | 10%            |             5.168 |           1.001 |                    5.240 |
| exact copies | SD 1.4          | D2b         | negative prompt                                    |             0.576 | 0%               | 1%                    | 100%                  |                 | 0%            | plain request  | 38%            |             5.166 |           1.001 |                    5.247 |
| exact copies | SD 1.4          | D3          | chain-of-thought                                   |             0.619 | 0%               | 3%                    | 100%                  |                 | 4%            | padding        | 6%             |             5.167 |           1.001 |                    5.234 |
| exact copies | SD 1.4          | D4          | LLM rewrite                                        |             0.574 | 0%               | 7%                    | 100%                  |                 | 1%            | override       | 1%             |             6.745 |           1.307 |                    6.837 |
| exact copies | SD 1.4          | G           | gate (ours)                                        |             0.639 | 0%               | 1%                    | 3%                    |                 | 0%            | plain request  | 9%             |             6.567 |           1.272 |                    5.810 |
| exact copies | SD 1.4          | G-pre       | gate, no output check (ablation)                   |             0.639 | 0%               | 1%                    | 3%                    |                 | 0%            | plain request  | 9%             |             6.537 |           1.267 |                    5.782 |
| exact copies | SD 1.4          | V0          | blocklist (vanilla)                                |             0.592 | 0%               | 0%                    | 0%                    |                 | 0%            | plain request  | 41%            |             5.104 |           0.989 |                    5.227 |
| exact copies | SD 2-base       | C1          | random words                                       |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             4.887 |           1.007 |                  nan     |
| exact copies | SD 2-base       | D0          | none                                               |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             4.852 |           1.000 |                  nan     |
| exact copies | SD 2-base       | D1          | task instruction                                   |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             4.868 |           1.003 |                  nan     |
| exact copies | SD 2-base       | D2a         | negation in prompt                                 |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             4.875 |           1.005 |                  nan     |
| exact copies | SD 2-base       | D2b         | negative prompt                                    |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             4.886 |           1.007 |                  nan     |
| exact copies | SD 2-base       | D3          | chain-of-thought                                   |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             4.895 |           1.009 |                  nan     |
| exact copies | SD 2-base       | D4          | LLM rewrite                                        |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             6.474 |           1.334 |                  nan     |
| exact copies | SD 2-base       | G           | gate (ours)                                        |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             6.119 |           1.261 |                  nan     |
| exact copies | SD 2-base       | G-pre       | gate, no output check (ablation)                   |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             6.086 |           1.254 |                  nan     |
| exact copies | SD 2-base       | V0          | blocklist (vanilla)                                |           nan     |                  |                       |                       |                 | 0%            | plain request  | 0%             |             4.852 |           1.000 |                  nan     |
| characters   | SD 1.4          | C1          | random words                                       |             0.132 | 0%               | 81%                   | 100%                  | 9%              | 5%            | 60-word lure   | 8%             |             5.479 |           1.006 |                    5.087 |
| characters   | SD 1.4          | D0          | none                                               |             0.415 | 0%               | 0%                    | 0%                    | 11%             | 0%            | plain request  | 38%            |             5.449 |           1.000 |                    5.022 |
| characters   | SD 1.4          | D1          | task instruction                                   |             0.371 | 0%               | 20%                   | 100%                  | 4%              | 0%            | plain request  | 46%            |             5.441 |           0.999 |                    5.061 |
| characters   | SD 1.4          | D2a         | negation in prompt                                 |             0.408 | 0%               | 7%                    | 100%                  | 0%              | 0%            | plain request  | 38%            |             5.462 |           1.002 |                    5.081 |
| characters   | SD 1.4          | D2b-kw      | negative prompt (name+kw)                          |             0.449 | 0%               | 15%                   | 100%                  | 7%              | 0%            | plain request  | 25%            |             5.462 |           1.003 |                    5.086 |
| characters   | SD 1.4          | D3          | chain-of-thought                                   |             0.428 | 0%               | 20%                   | 100%                  | 4%              | 0%            | plain request  | 38%            |             5.473 |           1.005 |                    5.083 |
| characters   | SD 1.4          | D4          | LLM rewrite                                        |             0.576 | 0%               | 24%                   | 100%                  | 4%              | 0%            | plain request  | 19%            |             7.036 |           1.291 |                    6.668 |
| characters   | SD 1.4          | D4+D2b      | LLM rewrite + neg. prompt                          |             0.428 | 0%               | 39%                   | 100%                  | 0%              | 3%            | 60-word lure   | 4%             |             7.042 |           1.292 |                    6.666 |
| characters   | SD 1.4          | G           | gate (ours)                                        |             0.494 | 0%               | 7%                    | 9%                    | 6%              | 5%            | keyword lure   | 4%             |             8.825 |           1.620 |                    5.865 |
| characters   | SD 1.4          | G-pre       | gate, no output check (ablation)                   |             0.494 | 0%               | 7%                    | 7%                    | 6%              | 10%           | keyword lure   | 8%             |             8.306 |           1.524 |                    5.685 |
| characters   | SD 1.4          | G-rw        | gate, rewrite-only character mitigation (ablation) |             0.593 | 0%               | 6%                    | 7%                    | 6%              | 0%            | plain request  | 13%            |             8.323 |           1.528 |                    5.681 |
| characters   | SD 1.4          | V0          | blocklist (vanilla)                                |             0.374 | 0%               | 0%                    | 0%                    | 11%             | 38%           | keyword lure   | 25%            |             4.076 |           0.748 |                    5.022 |
| characters   | Playground v2.5 | D0          | none                                               |             0.553 | 0%               | 0%                    | 0%                    | 2%              | 0%            | plain request  | 36%            |             6.262 |           1.000 |                    6.260 |
| characters   | Playground v2.5 | D2b-kw      | negative prompt (name+kw)                          |             0.613 | 0%               | -2%                   | 100%                  | 2%              | 0%            | plain request  | 21%            |             6.281 |           1.003 |                    6.281 |
| characters   | Playground v2.5 | D4          | LLM rewrite                                        |             0.663 | 0%               | 9%                    | 100%                  | 2%              | 7%            | 60-word lure   | 32%            |             7.845 |           1.253 |                    7.842 |
| characters   | Playground v2.5 | D4+D2b      | LLM rewrite + neg. prompt                          |             0.653 | 0%               | 11%                   | 100%                  | 2%              | 0%            | plain request  | 14%            |             7.866 |           1.256 |                    7.865 |
| characters   | Playground v2.5 | G           | gate (ours)                                        |             0.664 | 0%               | 2%                    | 31%                   | 2%              | 0%            | plain request  | 14%            |             8.464 |           1.352 |                    8.444 |
| characters   | Playground v2.5 | G-pre       | gate, no output check (ablation)                   |             0.663 | 0%               | 0%                    | 7%                    | 2%              | 5%            | keyword lure   | 21%            |             7.628 |           1.218 |                    6.525 |
| characters   | Playground v2.5 | G-rw        | gate, rewrite-only character mitigation (ablation) |             0.674 | 0%               | 0%                    | 7%                    | 2%              | 5%            | 60-word lure   | 37%            |             7.612 |           1.216 |                    6.524 |
| characters   | Playground v2.5 | V0          | blocklist (vanilla)                                |             0.469 | 0%               | 0%                    | 0%                    | 2%              | 33%           | keyword lure   | 29%            |             4.656 |           0.743 |                    6.260 |
| characters   | PixArt-α        | D0          | none                                               |             0.474 | 0%               | 0%                    | 0%                    | 6%              | 0%            | plain request  | 38%            |             1.312 |           1.000 |                    1.301 |
| characters   | PixArt-α        | D2b-kw      | negative prompt (name+kw)                          |             0.512 | 0%               | -2%                   | 100%                  | 2%              | 0%            | plain request  | 18%            |             1.305 |           0.995 |                    1.305 |
| characters   | PixArt-α        | D4          | LLM rewrite                                        |             0.644 | 0%               | 0%                    | 100%                  | 4%              | 0%            | plain request  | 24%            |             2.885 |           2.199 |                    2.885 |
| characters   | PixArt-α        | D4+D2b      | LLM rewrite + neg. prompt                          |             0.611 | 0%               | 2%                    | 100%                  | 4%              | 5%            | 60-word lure   | 7%             |             2.888 |           2.201 |                    2.890 |
| characters   | PixArt-α        | G           | gate (ours)                                        |             0.590 | 0%               | 2%                    | 33%                   | 0%              | 5%            | 60-word lure   | 7%             |             2.893 |           2.205 |                    2.330 |
| characters   | PixArt-α        | G-pre       | gate, no output check (ablation)                   |             0.590 | 0%               | 2%                    | 7%                    | 4%              | 13%           | 60-word lure   | 11%            |             2.654 |           2.023 |                    1.565 |
| characters   | PixArt-α        | G-rw        | gate, rewrite-only character mitigation (ablation) |             0.614 | 0%               | 2%                    | 7%                    | 4%              | 0%            | plain request  | 29%            |             2.653 |           2.022 |                    1.565 |
| characters   | PixArt-α        | V0          | blocklist (vanilla)                                |             0.413 | 0%               | 0%                    | 0%                    | 6%              | 36%           | 60-word lure   | 24%            |             0.968 |           0.738 |                    1.301 |
| styles       | SD 1.4          | C1          | random words                                       |             0.273 | 0%               | 37%                   | 100%                  |                 | 0%            | plain request  | 9%             |             5.597 |           1.002 |                    5.776 |
| styles       | SD 1.4          | D0          | none                                               |             0.591 | 0%               | 0%                    | 0%                    |                 | 0%            | plain request  | 5%             |             5.584 |           1.000 |                    5.755 |
| styles       | SD 1.4          | D1          | task instruction                                   |             0.475 | 0%               | 11%                   | 100%                  |                 | 0%            | plain request  | 20%            |             5.567 |           0.997 |                    5.755 |
| styles       | SD 1.4          | D2a         | negation in prompt                                 |             0.506 | 0%               | 4%                    | 100%                  |                 | 0%            | plain request  | 28%            |             5.576 |           0.999 |                    5.754 |
| styles       | SD 1.4          | D2b-name    | negative prompt (name)                             |             0.583 | 0%               | 2%                    | 100%                  |                 | 0%            | plain request  | 4%             |             5.577 |           0.999 |                    5.768 |
| styles       | SD 1.4          | D3          | chain-of-thought                                   |             0.527 | 0%               | 3%                    | 100%                  |                 | 0%            | plain request  | 23%            |             5.588 |           1.001 |                    5.752 |
| styles       | SD 1.4          | D4          | LLM rewrite                                        |             0.599 | 0%               | -2%                   | 100%                  |                 | 0%            | plain request  | 11%            |             7.179 |           1.286 |                    7.357 |
| styles       | SD 1.4          | G           | gate (ours)                                        |             0.588 | 0%               | 1%                    | 0%                    |                 | 0%            | plain request  | 3%             |             6.047 |           1.083 |                    6.227 |
| styles       | SD 1.4          | G-pre       | gate, no output check (ablation)                   |             0.588 | 0%               | 1%                    | 0%                    |                 | 0%            | plain request  | 7%             |             5.988 |           1.072 |                    6.200 |
| styles       | SD 1.4          | V0          | blocklist (vanilla)                                |             0.383 | 15%              | 10%                   | 15%                   |                 | 8%            | misspelling    | 2%             |             3.600 |           0.645 |                    4.901 |

## Subgroups: is the protection equal?

| threat       | model           | by                          | guardrail   | max   | min   | most_exposed        | gap   |
|:-------------|:----------------|:----------------------------|:------------|:------|:------|:--------------------|:------|
| characters   | PixArt-α        | origin_region               | D0          | 28%   | 18%   | US                  | 11%   |
| characters   | PixArt-α        | origin_region               | D2b-kw      | 8%    | 0%    | US                  | 8%    |
| characters   | PixArt-α        | origin_region               | D4          | 17%   | 5%    | US                  | 12%   |
| characters   | PixArt-α        | origin_region               | D4+D2b      | 3%    | 3%    | US                  | 0%    |
| characters   | PixArt-α        | origin_region               | G           | 3%    | 2%    | US                  | 0%    |
| characters   | PixArt-α        | origin_region               | V0          | 16%   | 10%   | US                  | 6%    |
| characters   | PixArt-α        | rights_holder               | D0          | 50%   | 12%   | Marvel (Disney)     | 38%   |
| characters   | PixArt-α        | rights_holder               | D2b-kw      | 18%   | 0%    | Marvel (Disney)     | 18%   |
| characters   | PixArt-α        | rights_holder               | D4          | 32%   | 8%    | Marvel (Disney)     | 24%   |
| characters   | PixArt-α        | rights_holder               | D4+D2b      | 7%    | 0%    | Marvel (Disney)     | 7%    |
| characters   | PixArt-α        | rights_holder               | G           | 7%    | 0%    | Marvel (Disney)     | 7%    |
| characters   | PixArt-α        | rights_holder               | V0          | 29%   | 4%    | Marvel (Disney)     | 24%   |
| characters   | Playground v2.5 | origin_region               | D0          | 37%   | 22%   | US                  | 15%   |
| characters   | Playground v2.5 | origin_region               | D2b-kw      | 15%   | 8%    | US                  | 7%    |
| characters   | Playground v2.5 | origin_region               | D4          | 15%   | 0%    | US                  | 15%   |
| characters   | Playground v2.5 | origin_region               | D4+D2b      | 6%    | 0%    | US                  | 6%    |
| characters   | Playground v2.5 | origin_region               | G           | 9%    | 0%    | US                  | 9%    |
| characters   | Playground v2.5 | origin_region               | V0          | 19%   | 8%    | US                  | 11%   |
| characters   | Playground v2.5 | rights_holder               | D0          | 57%   | 21%   | Marvel (Disney)     | 36%   |
| characters   | Playground v2.5 | rights_holder               | D2b-kw      | 25%   | 4%    | Marvel (Disney)     | 21%   |
| characters   | Playground v2.5 | rights_holder               | D4          | 32%   | 0%    | Marvel (Disney)     | 32%   |
| characters   | Playground v2.5 | rights_holder               | D4+D2b      | 14%   | 0%    | Marvel (Disney)     | 14%   |
| characters   | Playground v2.5 | rights_holder               | G           | 14%   | 0%    | Marvel (Disney)     | 14%   |
| characters   | Playground v2.5 | rights_holder               | V0          | 36%   | 6%    | Marvel (Disney)     | 29%   |
| characters   | SD 1.4          | origin_region               | D0          | 38%   | 36%   | JP                  | 1%    |
| characters   | SD 1.4          | origin_region               | D1          | 32%   | 32%   | JP                  | 1%    |
| characters   | SD 1.4          | origin_region               | D2b-kw      | 10%   | 5%    | US                  | 5%    |
| characters   | SD 1.4          | origin_region               | D3          | 32%   | 32%   | JP                  | 1%    |
| characters   | SD 1.4          | origin_region               | D4          | 11%   | 3%    | US                  | 9%    |
| characters   | SD 1.4          | origin_region               | D4+D2b      | 1%    | 0%    | US                  | 1%    |
| characters   | SD 1.4          | origin_region               | G           | 3%    | 0%    | US                  | 3%    |
| characters   | SD 1.4          | origin_region               | V0          | 18%   | 16%   | JP                  | 1%    |
| characters   | SD 1.4          | rights_holder               | D0          | 50%   | 12%   | The Pokemon Company | 38%   |
| characters   | SD 1.4          | rights_holder               | D1          | 58%   | 12%   | The Pokemon Company | 46%   |
| characters   | SD 1.4          | rights_holder               | D2b-kw      | 25%   | 0%    | Marvel (Disney)     | 25%   |
| characters   | SD 1.4          | rights_holder               | D3          | 50%   | 12%   | The Pokemon Company | 38%   |
| characters   | SD 1.4          | rights_holder               | D4          | 21%   | 2%    | Marvel (Disney)     | 19%   |
| characters   | SD 1.4          | rights_holder               | D4+D2b      | 4%    | 0%    | Marvel (Disney)     | 4%    |
| characters   | SD 1.4          | rights_holder               | G           | 4%    | 0%    | Pixar (Disney)      | 4%    |
| characters   | SD 1.4          | rights_holder               | V0          | 25%   | 0%    | The Pokemon Company | 25%   |
| exact copies | SD 1.4          | mem_type                    | D0          | 43%   | 2%    | VM                  | 41%   |
| exact copies | SD 1.4          | mem_type                    | D1          | 3%    | 0%    | VM                  | 3%    |
| exact copies | SD 1.4          | mem_type                    | D3          | 6%    | 0%    | VM                  | 6%    |
| exact copies | SD 1.4          | mem_type                    | D4          | 1%    | 0%    | VM                  | 1%    |
| exact copies | SD 1.4          | mem_type                    | G           | 9%    | 0%    | VM                  | 9%    |
| exact copies | SD 1.4          | mem_type                    | V0          | 43%   | 2%    | VM                  | 41%   |
| exact copies | SD 2-base       | mem_type                    | D0          | 8%    | 8%    | TM                  | 0%    |
| exact copies | SD 2-base       | mem_type                    | D1          | 0%    | 0%    | TM                  | 0%    |
| exact copies | SD 2-base       | mem_type                    | D3          | 0%    | 0%    | TM                  | 0%    |
| exact copies | SD 2-base       | mem_type                    | D4          | 0%    | 0%    | TM                  | 0%    |
| exact copies | SD 2-base       | mem_type                    | G           | 0%    | 0%    | TM                  | 0%    |
| exact copies | SD 2-base       | mem_type                    | V0          | 8%    | 8%    | TM                  | 0%    |
| styles       | SD 1.4          | artist group (above chance) | D0          | 22%   | 17%   | deceased_copyright  | 5%    |
| styles       | SD 1.4          | artist group (above chance) | D1          | 42%   | 21%   | deceased_copyright  | 20%   |
| styles       | SD 1.4          | artist group (above chance) | D2b-name    | 4%    | 0%    | living              | 4%    |
| styles       | SD 1.4          | artist group (above chance) | D3          | 37%   | 14%   | deceased_copyright  | 23%   |
| styles       | SD 1.4          | artist group (above chance) | D4          | 19%   | 8%    | deceased_copyright  | 11%   |
| styles       | SD 1.4          | artist group (above chance) | G           | 5%    | 2%    | living              | 3%    |
| styles       | SD 1.4          | artist group (above chance) | V0          | -1%   | -3%   | deceased_copyright  | 2%    |

## Why it works: which signal fired

| run                | attack        | risky   | signal                        |   n | share   |
|:-------------------|:--------------|:--------|:------------------------------|----:|:--------|
| pixart_copycat     | 60-word lure  | True    | concept screen                |  28 | 72%     |
| pixart_copycat     | 60-word lure  | True    | concept screen + output check |   2 | 5%      |
| pixart_copycat     | 60-word lure  | True    | none                          |   3 | 8%      |
| pixart_copycat     | 60-word lure  | True    | output check                  |   6 | 15%     |
| pixart_copycat     | keyword lure  | True    | concept screen                |  19 | 49%     |
| pixart_copycat     | keyword lure  | True    | concept screen + output check |   2 | 5%      |
| pixart_copycat     | keyword lure  | True    | none                          |  15 | 38%     |
| pixart_copycat     | keyword lure  | True    | output check                  |   3 | 8%      |
| pixart_copycat     | misspelling   | True    | concept screen                |  30 | 77%     |
| pixart_copycat     | misspelling   | True    | concept screen + output check |   2 | 5%      |
| pixart_copycat     | misspelling   | True    | none                          |   7 | 18%     |
| pixart_copycat     | plain request | True    | concept screen                |  38 | 97%     |
| pixart_copycat     | plain request | True    | concept screen + output check |   1 | 3%      |
| pixart_generic     | plain request | False   | concept screen                |   4 | 7%      |
| pixart_generic     | plain request | False   | none                          |  36 | 67%     |
| pixart_generic     | plain request | False   | output check                  |  14 | 26%     |
| playground_copycat | 60-word lure  | True    | concept screen                |  29 | 74%     |
| playground_copycat | 60-word lure  | True    | concept screen + output check |   1 | 3%      |
| playground_copycat | 60-word lure  | True    | none                          |   3 | 8%      |
| playground_copycat | 60-word lure  | True    | output check                  |   6 | 15%     |
| playground_copycat | keyword lure  | True    | concept screen                |  20 | 51%     |
| playground_copycat | keyword lure  | True    | concept screen + output check |   1 | 3%      |
| playground_copycat | keyword lure  | True    | none                          |  11 | 28%     |
| playground_copycat | keyword lure  | True    | output check                  |   7 | 18%     |
| playground_copycat | misspelling   | True    | concept screen                |  30 | 77%     |
| playground_copycat | misspelling   | True    | concept screen + output check |   2 | 5%      |
| playground_copycat | misspelling   | True    | none                          |   7 | 18%     |
| playground_copycat | plain request | True    | concept screen                |  39 | 100%    |
| playground_generic | plain request | False   | concept screen                |   4 | 7%      |
| playground_generic | plain request | False   | none                          |  37 | 69%     |
| playground_generic | plain request | False   | output check                  |  13 | 24%     |
| sd14_controls      | plain request | False   | memorization                  |   3 | 3%      |
| sd14_controls      | plain request | False   | none                          |  97 | 97%     |
| sd14_copycat       | 60-word lure  | True    | concept screen                |  29 | 74%     |
| sd14_copycat       | 60-word lure  | True    | concept screen + output check |   1 | 3%      |
| sd14_copycat       | 60-word lure  | True    | none                          |   6 | 15%     |
| sd14_copycat       | 60-word lure  | True    | output check                  |   3 | 8%      |
| sd14_copycat       | keyword lure  | True    | concept screen                |  20 | 51%     |
| sd14_copycat       | keyword lure  | True    | concept screen + output check |   1 | 3%      |
| sd14_copycat       | keyword lure  | True    | none                          |  14 | 36%     |
| sd14_copycat       | keyword lure  | True    | output check                  |   4 | 10%     |
| sd14_copycat       | misspelling   | True    | concept screen                |  32 | 82%     |
| sd14_copycat       | misspelling   | True    | none                          |   6 | 15%     |
| sd14_copycat       | misspelling   | True    | output check                  |   1 | 3%      |
| sd14_copycat       | plain request | True    | concept screen                |  39 | 100%    |
| sd14_generic       | plain request | False   | concept screen                |   4 | 7%      |
| sd14_generic       | plain request | False   | none                          |  49 | 91%     |
| sd14_generic       | plain request | False   | output check                  |   1 | 2%      |
| sd14_memorized     | override      | True    | concept screen                |   1 | 1%      |
| sd14_memorized     | override      | True    | memorization                  | 113 | 87%     |
| sd14_memorized     | override      | True    | none                          |  16 | 12%     |
| sd14_memorized     | padding       | True    | concept screen                |   1 | 1%      |
| sd14_memorized     | padding       | True    | memorization                  | 121 | 93%     |
| sd14_memorized     | padding       | True    | none                          |   8 | 6%      |
| sd14_memorized     | plain request | True    | concept screen                |   1 | 1%      |
| sd14_memorized     | plain request | True    | memorization                  | 121 | 93%     |
| sd14_memorized     | plain request | True    | none                          |   8 | 6%      |
| sd14_style         | misspelling   | False   | none                          |  80 | 100%    |
| sd14_style         | misspelling   | True    | concept screen                |  97 | 81%     |
| sd14_style         | misspelling   | True    | none                          |  23 | 19%     |
| sd14_style         | plain request | False   | none                          |  80 | 100%    |
| sd14_style         | plain request | True    | concept screen                | 118 | 98%     |
| sd14_style         | plain request | True    | concept screen + output check |   2 | 2%      |
| sd2_memorized      | plain request | True    | memorization                  |  93 | 93%     |
| sd2_memorized      | plain request | True    | none                          |   7 | 7%      |

What still gets through: [gate_errors.md](gate_errors.md)

## Ablation: which character mitigation should the gate apply?

Same detection, different mitigation. The negative prompt buys protection and costs on-prompt images (it also suppresses the generic entity, e.g. the mouse in a Mickey request); the rewrite alone keeps more of the request and lets more through.

| model           | guardrail                                               | mitigation                                             | plain   | worst_attack   | mean_attacks   | risky_on_prompt   | benign_served   |   su_f1 | f1_ci        |   sec_per_request |
|:----------------|:--------------------------------------------------------|:-------------------------------------------------------|:--------|:---------------|:---------------|:------------------|:----------------|--------:|:-------------|------------------:|
| SD 1.4          | D4 LLM rewrite                                          | rewrite, every request                                 | 18%     | 18%            | 9%             | 50%               | 74%             |   0.576 | [0.49, 0.66] |             7.036 |
| SD 1.4          | D4+D2b† LLM rewrite + neg. prompt                       | rewrite + true target's negative prompt, every request | 0%      | 3%             | 1%             | 34%               | 59%             |   0.428 | [0.34, 0.51] |             7.042 |
| SD 1.4          | G-rw gate, rewrite-only character mitigation (ablation) | rewrite only                                           | 18%     | 18%            | 13%            | 48%               | 93%             |   0.593 | [0.49, 0.68] |             8.323 |
| SD 1.4          | G-pre gate, no output check (ablation)                  | rewrite + detected character's negative prompt         | 0%      | 10%            | 5%             | 36%               | 91%             |   0.494 | [0.40, 0.59] |             8.306 |
| SD 1.4          | G gate (ours)                                           | rewrite + negative prompt, + output check              | 0%      | 5%             | 2%             | 34%               | 91%             |   0.494 | [0.39, 0.59] |             8.825 |
| Playground v2.5 | D4 LLM rewrite                                          | rewrite, every request                                 | 16%     | 23%            | 11%            | 55%               | 89%             |   0.663 | [0.57, 0.74] |             7.845 |
| Playground v2.5 | D4+D2b† LLM rewrite + neg. prompt                       | rewrite + true target's negative prompt, every request | 11%     | 11%            | 5%             | 53%               | 87%             |   0.653 | [0.56, 0.73] |             7.866 |
| Playground v2.5 | G-rw gate, rewrite-only character mitigation (ablation) | rewrite only                                           | 15%     | 21%            | 13%            | 55%               | 98%             |   0.674 | [0.57, 0.76] |             7.612 |
| Playground v2.5 | G-pre gate, no output check (ablation)                  | rewrite + detected character's negative prompt         | 10%     | 15%            | 9%             | 53%               | 98%             |   0.663 | [0.56, 0.75] |             7.628 |
| Playground v2.5 | G gate (ours)                                           | rewrite + negative prompt, + output check              | 10%     | 10%            | 7%             | 53%               | 96%             |   0.664 | [0.57, 0.75] |             8.464 |
| PixArt-α        | D4 LLM rewrite                                          | rewrite, every request                                 | 29%     | 29%            | 14%            | 52%               | 96%             |   0.644 | [0.54, 0.73] |             2.885 |
| PixArt-α        | D4+D2b† LLM rewrite + neg. prompt                       | rewrite + true target's negative prompt, every request | 3%      | 8%             | 3%             | 46%               | 94%             |   0.611 | [0.51, 0.70] |             2.888 |
| PixArt-α        | G-rw gate, rewrite-only character mitigation (ablation) | rewrite only                                           | 28%     | 28%            | 15%            | 49%               | 94%             |   0.614 | [0.51, 0.70] |             2.653 |
| PixArt-α        | G-pre gate, no output check (ablation)                  | rewrite + detected character's negative prompt         | 3%      | 15%            | 5%             | 45%               | 94%             |   0.590 | [0.48, 0.68] |             2.654 |
| PixArt-α        | G gate (ours)                                           | rewrite + negative prompt, + output check              | 3%      | 8%             | 3%             | 44%               | 94%             |   0.590 | [0.48, 0.69] |             2.893 |

## On-prompt judge validation

Matched vs. mismatched: the judge's P(yes) for the image's own request vs. another request of the same run (AUC). Hand-labelled sample: 120 images stratified over runs and balanced between the judge's yes and no, labelled by a person who did not see the judge's answer ([onprompt_labels.csv](onprompt_labels.csv)).

| run                                |   images |   auc_matched_vs_mismatched |   yes_rate_matched |   false_yes_rate_mismatched |   agreement |   cohen_kappa |   human_yes |   judge_yes |   judge_yes_precision |   judge_no_but_human_yes |
|:-----------------------------------|---------:|----------------------------:|-------------------:|----------------------------:|------------:|--------------:|------------:|------------:|----------------------:|-------------------------:|
| sd14_controls                      |     1000 |                       0.954 |              0.574 |                       0.024 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| sd14_copycat                       |     1674 |                       0.720 |              0.371 |                       0.113 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| sd14_memorized                     |     3767 |                       0.939 |              0.467 |                       0.012 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| sd14_style                         |     3856 |                       0.976 |              0.564 |                       0.002 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| sd2_memorized                      |     1000 |                       0.794 |              0.475 |                       0.126 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| pixart_copycat                     |     1050 |                       0.783 |              0.444 |                       0.116 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| pixart_generic                     |      378 |                       0.972 |              0.958 |                       0.106 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| playground_copycat                 |     1050 |                       0.776 |              0.525 |                       0.121 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| playground_generic                 |      378 |                       0.987 |              0.952 |                       0.029 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| sd14_generic                       |      594 |                       0.933 |              0.779 |                       0.054 |     nan     |       nan     |     nan     |     nan     |               nan     |                  nan     |
| hand-labelled sample, P(yes) > 0.5 |      120 |                     nan     |            nan     |                     nan     |       0.667 |         0.333 |       0.767 |       0.500 |                 0.933 |                    0.600 |
| hand-labelled sample, P(yes) > 0.1 |      120 |                     nan     |            nan     |                     nan     |       0.817 |         0.544 |       0.767 |       0.683 |                 0.927 |                    0.421 |

## Does the judge's threshold change the conclusions?

The hand labels show the judge's on-prompt threshold (P(yes) > 0.5) is strict: it calls many images off-prompt that a person calls on-prompt (judge validation above). P(yes) > 0.1 agrees best with the hand labels (Cohen's kappa 0.54; chosen on the same sample, so optimistic). Below, every on-prompt-based number is recomputed at that threshold (protection numbers do not use the judge); F1 (rank among the guardrails) at 0.5 -> at 0.1. All guardrails: [onprompt_threshold_sensitivity.csv](onprompt_threshold_sensitivity.csv).

| threat       | model           | best at 0.5   | best at 0.1      | G                              | G-rw                           |
|:-------------|:----------------|:--------------|:-----------------|:-------------------------------|:-------------------------------|
| exact copies | SD 1.4          | G (0.639)     | G (0.909)        | 0.639 (#1) -> 0.909 (#1 of 10) | –                              |
| characters   | SD 1.4          | G-rw (0.593)  | D4 (0.704)       | 0.494 (#3) -> 0.681 (#3 of 12) | 0.593 (#1) -> 0.695 (#2 of 12) |
| characters   | Playground v2.5 | G-rw (0.674)  | G-rw (0.752)     | 0.664 (#2) -> 0.747 (#2 of 8)  | 0.674 (#1) -> 0.752 (#1 of 8)  |
| characters   | PixArt-α        | D4 (0.644)    | D4 (0.750)       | 0.590 (#4) -> 0.674 (#4 of 8)  | 0.614 (#2) -> 0.716 (#2 of 8)  |
| styles       | SD 1.4          | D4 (0.599)    | D2b-name (0.711) | 0.588 (#3) -> 0.701 (#2 of 10) | –                              |

## Across model families

|                          |   ('mean_attacks', 'PixArt-α') |   ('mean_attacks', 'Playground v2.5') |   ('mean_attacks', 'SD 1.4') |   ('su_f1', 'PixArt-α') |   ('su_f1', 'Playground v2.5') |   ('su_f1', 'SD 1.4') |
|:-------------------------|-------------------------------:|--------------------------------------:|-----------------------------:|------------------------:|-------------------------------:|----------------------:|
| ('characters', 'D0')     |                          0.256 |                                 0.333 |                        0.365 |                   0.474 |                          0.553 |                 0.415 |
| ('characters', 'D1')     |                        nan     |                               nan     |                        0.321 |                 nan     |                        nan     |                 0.371 |
| ('characters', 'D2b-kw') |                          0.058 |                                 0.128 |                        0.09  |                   0.512 |                          0.613 |                 0.449 |
| ('characters', 'D4')     |                          0.143 |                                 0.11  |                        0.091 |                   0.644 |                          0.663 |                 0.576 |
| ('characters', 'D4+D2b') |                          0.026 |                                 0.046 |                        0.006 |                   0.611 |                          0.653 |                 0.428 |
| ('characters', 'G')      |                          0.026 |                                 0.071 |                        0.019 |                   0.59  |                          0.664 |                 0.494 |
| ('characters', 'G-rw')   |                          0.154 |                                 0.135 |                        0.128 |                   0.614 |                          0.674 |                 0.593 |
| ('characters', 'V0')     |                          0.141 |                                 0.16  |                        0.167 |                   0.413 |                          0.469 |                 0.374 |
| ('styles', 'D0')         |                        nan     |                               nan     |                        0.154 |                 nan     |                        nan     |                 0.591 |
| ('styles', 'D1')         |                        nan     |                               nan     |                        0.192 |                 nan     |                        nan     |                 0.475 |
| ('styles', 'D2b-name')   |                        nan     |                               nan     |                        0.062 |                 nan     |                        nan     |                 0.583 |
| ('styles', 'D4')         |                        nan     |                               nan     |                        0.138 |                 nan     |                        nan     |                 0.599 |
| ('styles', 'G')          |                        nan     |                               nan     |                        0.071 |                 nan     |                        nan     |                 0.588 |
| ('styles', 'V0')         |                        nan     |                               nan     |                        0.042 |                 nan     |                        nan     |                 0.383 |

## Figures

![fig_gate_flow.png](fig_gate_flow.png)
![fig_f1_exact_copies_SD14.png](fig_f1_exact_copies_SD14.png)
![fig_f1_characters_SD14.png](fig_f1_characters_SD14.png)
![fig_f1_characters_Playgroundv25.png](fig_f1_characters_Playgroundv25.png)
![fig_f1_characters_PixArtalpha.png](fig_f1_characters_PixArtalpha.png)
![fig_f1_styles_SD14.png](fig_f1_styles_SD14.png)
![fig_tradeoff_sd14.png](fig_tradeoff_sd14.png)
![fig_gate_signals.png](fig_gate_signals.png)
![fig_models_characters.png](fig_models_characters.png)
