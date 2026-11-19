### sd14_style: style attributed (lower = better protection)

| guardrail                       | plain request   | misspelling   |
|:--------------------------------|:----------------|:--------------|
| D0 none                         | 31%             | 5%            |
| V0 blocklist (vanilla)          | 12%             | 5%            |
| D1 task instruction             | 32%             | 4%            |
| D2a negation in prompt          | 38%             | 4%            |
| D2b-name negative prompt (name) | 5%              | 2%            |
| D3 chain-of-thought             | 32%             | 4%            |
| D4 LLM rewrite                  | 22%             | 7%            |
| C1 random words                 | 12%             | 4%            |
