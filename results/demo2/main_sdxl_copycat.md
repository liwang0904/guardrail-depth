### sdxl_copycat: character detected (lower = better protection)

| guardrail                        | plain request   | 60-word lure   | keyword lure   | misspelling   |
|:---------------------------------|:----------------|:---------------|:---------------|:--------------|
| D0 none                          | 56%             | 18%            | 41%            | 3%            |
| V0 blocklist (vanilla)           | 0%              | 18%            | 38%            | 3%            |
| D1 task instruction              | 69%             | 18%            | 33%            | 3%            |
| D2a negation in prompt           | 79%             | 21%            | 36%            | 3%            |
| D2b-kw negative prompt (name+kw) | 21%             | 3%             | 13%            | 0%            |
| D3 chain-of-thought              | 67%             | 23%            | 33%            | 3%            |
| D4 LLM rewrite                   | 21%             | 18%            | 3%             | 0%            |
| D4+D2b LLM rewrite + neg. prompt | 5%              | 0%             | 0%             | 0%            |
| C1 random words                  | 0%              | 8%             | 0%             | 0%            |
