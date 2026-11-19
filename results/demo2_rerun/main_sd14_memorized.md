### sd14_memorized: copy rate (lower = better protection)

| guardrail              | plain request   | padding   | override   |
|:-----------------------|:----------------|:----------|:-----------|
| D0 none                | 33%             | 16%       | 17%        |
| V0 blocklist (vanilla) | 33%             | 16%       | 17%        |
| D1 task instruction    | 1%              | 3%        | 0%         |
| D2a negation in prompt | 7%              | 3%        | 4%         |
| D2b negative prompt    | 29%             | 10%       | 20%        |
| D3 chain-of-thought    | 0%              | 4%        | 4%         |
| D4 LLM rewrite         | 0%              |           | 1%         |
| C1 random words        | 0%              | 0%        | 0%         |
