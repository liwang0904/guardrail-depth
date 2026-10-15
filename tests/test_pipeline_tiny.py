"""End-to-end path on tiny random checkpoints (CPU, a few MB downloaded once).

Checks wiring, shapes and bookkeeping - not image quality.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from guardrail_depth import probes as P
from guardrail_depth.generation import Job, run_job
from guardrail_depth.models import load_t2i

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def tiny():
    pipe, spec = load_t2i("tiny", torch.device("cpu"))
    return pipe, spec


def test_run_job_writes_images_records_and_probes(tiny, tmp_path):
    pipe, spec = tiny
    job = Job(name="t", model="tiny", dataset="memorized_sd1", ids=[2, 7], attacks=["direct", "override"],
              defenses=["none", "task_instruction", "negbuf_rules"], seeds=[0, 1],
              probes=["noise", "attention"], x0_steps=[0, 1], steps=2)
    rec_path = run_job(pipe, spec, job, tmp_path, log=lambda *_: None)
    recs = [json.loads(l) for l in open(rec_path)]
    assert len(recs) == 2 * 2 * 3 * 2
    r = recs[0]
    assert (tmp_path / r["image"]).exists() and len(r["x0"]) == 2
    for k in ["noise_first", "attn_entropy_t0", "attn_user_avg", "attn_guardrail_t0", "n_tokens"]:
        assert k in r and np.isfinite(r[k]), k
    # guardrail tokens only exist when a template wraps the prompt
    by = {(x["defense"], x["attack"]): x for x in recs if x["seed"] == 0 and x["item_id"] == 2}
    assert by[("none", "direct")]["attn_guardrail_t0"] == pytest.approx(0, abs=1e-6)
    assert by[("task_instruction", "direct")]["attn_guardrail_t0"] > 0
    npz = np.load(next((tmp_path / "probes").glob("*.npz")))
    assert npz["noise"].shape == (2, 2) and npz["attention"].shape[-1] == 77
    # resume: nothing new is generated
    n_before = len(recs)
    run_job(pipe, spec, job, tmp_path, log=lambda *_: None)
    assert len(open(rec_path).readlines()) == n_before


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="needs a GPU")
def test_attention_probe_is_finite_and_exact_on_gpu_fp16():
    """Regression: diffusers' baddbmm(beta=0) attention path produced NaN on Apple MPS."""
    from guardrail_depth.models import pick_device

    pipe, _ = load_t2i("tiny", pick_device(), torch.float16)
    kw = dict(num_inference_steps=3, output_type="np", height=64, width=64)
    ref = pipe("a cat", generator=torch.Generator("cpu").manual_seed(0), **kw).images
    with P.probes_attached(pipe, noise=True, attention=True) as probes:
        out = pipe("a cat", generator=torch.Generator("cpu").manual_seed(0), **kw).images
        att = probes["attention"].result()
    assert np.isfinite(att).all() and np.allclose(att.sum(-1), 1, atol=1e-3)
    assert np.isfinite(out).all() and np.abs(out - ref).max() < 2e-2  # probe does not change the output


def test_wen_scores_and_nemo_neurons(tiny):
    pipe, _ = tiny
    s = P.wen_detection_scores(pipe, ["a cat", "a dog", "a red car"], n_images=2, num_steps=5, batch_size=2)
    assert s.shape == (3, 2) and np.isfinite(s).all()
    # paired noise: identical prompts give identical scores
    s2 = P.wen_detection_scores(pipe, ["a cat", "a cat"], n_images=2, num_steps=5)
    assert np.allclose(s2[0], s2[1])
    acts = P.neuron_activations(pipe, ["a cat", "a dog", "a red car", "a tree"])
    stats = P.neuron_reference_stats(acts)
    out = P.neuron_outliers(acts, stats, theta=1.0)
    assert len(acts) == len(P.nemo_value_layers(pipe.unet)) and out["n_ood"].shape == (4,)


def test_attention_summary_math():
    mass = np.zeros(77)
    mass[0], mass[1:4], mass[4] = 0.7, 0.05, 0.15
    s = P.attention_summary(mass, n_prompt_tokens=3, user_tokens=(0, 2))
    assert s["attn_bos"] == pytest.approx(0.7) and s["attn_prompt"] == pytest.approx(0.15)
    assert s["attn_user"] == pytest.approx(0.10) and s["attn_guardrail"] == pytest.approx(0.05)
    assert s["attn_summary"] == pytest.approx(0.15) and s["attn_prompt_entropy"] == pytest.approx(1.0)


def test_rewriter_and_judge_api():
    from guardrail_depth.judge import VLMJudge, matches_target
    from guardrail_depth.rewriter import LLMRewriter, leakage

    rw = LLMRewriter("trl-internal-testing/tiny-Qwen3ForCausalLM", thinking=True, max_new_tokens=6)
    out = rw.rewrite("Mario")
    assert {"rewrite", "reasoning", "entropy_final_mean", "n_reason_tokens"} <= set(out)
    assert leakage("a plumber named mario in red", "Mario", ["plumber", "green"])["kw_hits"] == 1
    judge = VLMJudge("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration", device="cpu")
    img = Image.new("RGB", (64, 64), "red")
    assert {"character", "score", "raw"} <= set(judge.detect(img, max_new_tokens=4))
    p = judge.p_yes(img, "a mermaid")
    assert 0 <= p["p_yes_norm"] <= 1
    assert matches_target("Spider Man", "Spider-Man") and not matches_target("Man", "Spider-Man")
