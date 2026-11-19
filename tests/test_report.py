"""Demo 3 report: the G-rw ablation is assembled from existing renders, and gate runtimes are charged correctly."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo3_report as R  # noqa: E402


def _run():
    rows = []
    # three character requests: screened with a rewrite (1), missed by the screen (2), screened without a rewrite (3)
    for item, applied in ((1, "llm_rewrite+negbuf_name_kw"), (2, "none"), (3, "negbuf_name_kw")):
        base = dict(dataset="copycat_characters", item_id=item, attack_code="A0", seed=0, refused=0, sec_per_image=5.0)
        rows.append({**base, "defense_code": "D0", "detected": 1, "on_prompt": 1, "model_prompt": f"plain {item}"})
        if item != 3:   # D4 has no row where no rewrite exists
            rows.append({**base, "defense_code": "D4", "detected": 0, "on_prompt": 1, "model_prompt": f"rewrite {item}",
                         "image": f"d4_{item}.png"})
        rows.append({**base, "defense_code": "G-pre", "detected": 0 if item != 2 else 1, "on_prompt": 0 if item != 2 else 1,
                     "model_prompt": f"rewrite {item}" if item == 1 else f"plain {item}", "gate_applied": applied,
                     "gate_final_applied": applied, "screen_seconds": 0.1, "mem_seconds": 0.0, "image": f"pre_{item}.png"})
        rows.append({**base, "defense_code": "G", "detected": 0, "on_prompt": 0, "gate_applied": applied,
                     "gate_final_applied": applied, "screen_seconds": 0.1, "mem_seconds": 1.0, "n_generations": 1,
                     "pre_check_seconds": 0.02, "check_seconds": 0.02})
    return pd.DataFrame(rows)


def test_rewrite_only_ablation_takes_the_d4_image_only_where_the_gate_mitigated_a_character():
    logs = []
    runs = R.add_rewrite_only({"sd14_copycat": _run()}, log=logs.append)
    rw = runs["sd14_copycat"].query("defense_code == 'G-rw'").set_index("item_id")
    assert len(rw) == 3
    # screened with a rewrite: D4's image and judgments, the gate's bookkeeping
    assert rw.loc[1, "image"] == "d4_1.png" and rw.loc[1, "detected"] == 0 and rw.loc[1, "on_prompt"] == 1
    assert rw.loc[1, "gate_applied"] == "llm_rewrite" and rw.loc[1, "screen_seconds"] == 0.1
    # missed by the screen, or no rewrite: G-pre's image (D0's, the D2b-kw fallback)
    assert rw.loc[2, "image"] == "pre_2.png" and rw.loc[3, "image"] == "pre_3.png"
    assert "same rewrite as D4: 100%" in logs[0]
    # other threats are left alone
    style = pd.DataFrame([{"dataset": "style_prompts", "defense_code": "G-pre", "gate_applied": "negbuf_name"}])
    assert "G-rw" not in set(R.add_rewrite_only({"sd14_style": style}, log=lambda *_: None)["sd14_style"]["defense_code"])


def test_gate_runtimes_charge_the_shared_memorization_screen_and_only_g_pays_for_the_check():
    runs = R.add_rewrite_only({"sd14_copycat": _run()}, log=lambda *_: None)
    rt = R.runtime(runs["sd14_copycat"], rewrite_seconds=3.0)
    # render 5 + screen 0.1 + memorization 1.0 (computed by the G row) + rewrite on 1 of 3 requests
    assert np.isclose(rt["G-pre"], 5 + 0.1 + 1.0 + 3.0 / 3)
    assert np.isclose(rt["G-rw"], rt["G-pre"])
    assert np.isclose(rt["G"], rt["G-pre"] + 0.02)
    iv = R.intervention_rate(runs["sd14_copycat"])
    assert np.isclose(iv["G-rw"], 2 / 3) and np.isclose(iv["G-pre"], 2 / 3)
