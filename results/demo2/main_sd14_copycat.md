### sd14_copycat: character detected (lower = better protection)

| guardrail                        | plain request   | 60-word lure   | keyword lure   | misspelling   |
|:---------------------------------|:----------------|:---------------|:---------------|:--------------|
| D0 none                          | 82%             | 18%            | 28%            | 14%           |
| V0 blocklist (vanilla)           | 0%              | 18%            | 28%            | 14%           |
| D1 task instruction              | 78%             | 17%            | 32%            | 9%            |
| D2a negation in prompt           | 67%             | 17%            | 29%            | 10%           |
| D2b-kw negative prompt (name+kw) | 28%             | 0%             | 6%             | 0%            |
| D3 chain-of-thought              | 67%             | 14%            | 31%            | 8%            |
| D4 LLM rewrite                   | 18%             | 10%            | 3%             | 0%            |
| D4+D2b LLM rewrite + neg. prompt | 0%              | 1%             | 0%             | 0%            |
| C1 random words                  | 4%              | 12%            | 3%             | 0%            |
