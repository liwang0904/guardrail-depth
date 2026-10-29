"""Check our value-neuron activations against NeMo's released statistics, and measure how guardrail
templates change the activation distribution.

    python scripts/validate_nemo.py   # -> outputs/neurons/validation.json

If the re-implementation matches NeMo, raw non-memorized prompts give z-scores with mean ~0 and
std ~1 under NeMo's statistics. Wrapping the same prompts in a template shows whether NeMo's raw-prompt
statistics remain valid for guarded prompts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardrail_depth.data import load_manifest  # noqa: E402
from guardrail_depth.defenses import apply_defense  # noqa: E402
from guardrail_depth.models import load_t2i  # noqa: E402
from guardrail_depth.paths import output_root, raw_dir  # noqa: E402
from guardrail_depth.probes import load_nemo_stats, neuron_activations  # noqa: E402


def main() -> None:
    stats = load_nemo_stats(raw_dir() / "nemo/statistics/statistics_additional_laion_prompts_v1_4.pt")
    pipe, _ = load_t2i("sd14")
    prompts = load_manifest("unmemorized_laion").query("split == 'test'")["prompt"].tolist()[:200]
    res = {}
    for cond in ["none", "task_instruction", "neg_inprompt", "cot", "rand_tokens"]:
        acts = neuron_activations(pipe, [apply_defense(cond, p, {}).prompt for p in prompts])
        z = np.concatenate([((a - m) / s).ravel() for a, (m, s) in zip(acts, stats)])
        res[cond] = {"z_mean": float(z.mean()), "z_std": float(z.std()), "frac_z_gt_5": float((z > 5).mean()),
                     "layer_mean_ratio": [float(a.mean() / m.mean()) for a, (m, _) in zip(acts, stats)]}
        print(f"{cond:18s} z mean {z.mean():+.3f}  std {z.std():.3f}  frac>5 {np.mean(z > 5):.1e}")
    out = output_root() / "neurons"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps({"n_prompts": len(prompts), **res}, indent=1))


if __name__ == "__main__":
    main()
