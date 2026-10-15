"""guardrail-depth: evaluating how deep prompt-level copyright guardrails go.

Subpackages are imported lazily by the scripts so that the lightweight parts
(defense templates, attacks, manifests) work without torch/diffusers installed.
"""

__version__ = "0.1.0"
