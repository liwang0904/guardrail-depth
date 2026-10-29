### sd14_memorized: copy rate (lower = better protection)

| guardrail              | plain request   | padding   | override   |
|:-----------------------|:----------------|:----------|:-----------|
| D0 none                | 33%             | 12%       | 15%        |
| V0 blocklist (vanilla) | 33%             | 12%       | 15%        |
| D1 task instruction    | 1%              | 2%        | 1%         |
| D2a negation in prompt | 6%              | 3%        | 5%         |
| D2b negative prompt    | 29%             | 8%        | 16%        |
| D3 chain-of-thought    | 1%              | 3%        | 4%         |
| D4 LLM rewrite         | 0%              |           | 1%         |
| C1 random words        | 0%              | 0%        | 0%         |
