### sd14_copycat: character detected (lower = better protection)

| guardrail                        | plain request   | 60-word lure   | keyword lure   | misspelling   |
|:---------------------------------|:----------------|:---------------|:---------------|:--------------|
| D0 none                          | 79%             | 18%            | 38%            | 10%           |
| V0 blocklist (vanilla)           | 0%              | 18%            | 38%            | 10%           |
| D1 task instruction              | 72%             | 18%            | 28%            | 10%           |
| D2a negation in prompt           | 62%             | 18%            | 31%            | 10%           |
| D2b-kw negative prompt (name+kw) | 33%             | 0%             | 3%             | 0%            |
| D3 chain-of-thought              | 72%             | 18%            | 28%            | 10%           |
| D4 LLM rewrite                   | 18%             | 13%            | 5%             | 0%            |
| D4+D2b LLM rewrite + neg. prompt | 0%              | 3%             | 0%             | 0%            |
| C1 random words                  | 5%              | 10%            | 3%             | 0%            |
