"""Tests for batched execution: decode_batch, OpenAI per-request error capture,
and end-to-end determinism (batched results must not depend on execution)."""

import pandas as pd

from ccap.decoders import ModelDecoder
from ccap.grid import BENIGN, GridSpec
from ccap.mechanisms import get_mechanism
from ccap.models import MockBackend
from ccap.models.base import ModelBackend
from ccap.models.openai_api import OpenAIBackend
from ccap.runner import ExperimentRunner
from ccap.types import GenRequest, GenResult


class _FixedBitsBackend(ModelBackend):
    """Returns a fixed bit string regardless of prompt (stands in for a model decoder)."""
    name = "fixed"

    def __init__(self, bits_str):
        self.bits_str = bits_str

    def generate(self, request: GenRequest) -> GenResult:
        return GenResult(text=self.bits_str, backend=self.name)


def test_model_decoder_decode_batch():
    dec = ModelDecoder(_FixedBitsBackend("0110"), name="model")
    mech = get_mechanism("framing")
    out = dec.decode_batch(["cot a", "cot b", "cot c"], mech, 4)
    assert out == [(0, 1, 1, 0)] * 3


class _FlakyOpenAI(OpenAIBackend):
    """Overrides the HTTP call: raises for prompts containing BOOM, else echoes 'ok'."""
    def generate(self, request: GenRequest) -> GenResult:
        if "BOOM" in request.prompt:
            raise RuntimeError("simulated failure")
        return GenResult(text="ok", backend=self.name)


def test_openai_generate_batch_captures_errors_in_order():
    be = _FlakyOpenAI(model_id="x", api_key="k", max_workers=4)
    reqs = [GenRequest(prompt=p) for p in ["good1", "BOOM", "good2"]]
    res = be.generate_batch(reqs)
    assert [r.text for r in res] == ["ok", "", "ok"]      # order preserved
    assert res[0].error is None and res[2].error is None
    assert res[1].error is not None and "simulated failure" in res[1].error


def test_per_backend_max_tokens_override():
    from ccap.models import build_backend
    be = build_backend({"type": "openai", "model_id": "x", "api_key": "k", "max_tokens": 12345})
    assert be.max_tokens == 12345                       # spec key -> per-backend cap
    plain = build_backend({"type": "openai", "model_id": "x", "api_key": "k"})
    assert plain.max_tokens is None
    req = GenRequest(prompt="hi", max_tokens=8192)      # cap wins over the request budget
    assert (be.max_tokens or req.max_tokens) == 12345
    assert (plain.max_tokens or req.max_tokens) == 8192


def test_batched_run_is_deterministic():
    grid = GridSpec(models=["A", "B"], mechanisms=[BENIGN, "acrostic", "framing"],
                    tasks=["math"], slots=16, ladder=(1, 2, 4, 8, 16), n_items=40)
    backends = {"A": MockBackend(name="A", fidelity=0.97), "B": MockBackend(name="B", fidelity=0.9)}
    d1 = ExperimentRunner(backends, verbose=False).run(grid)
    d2 = ExperimentRunner(backends, verbose=False).run(grid)
    key = ["model", "mechanism", "task", "trial_index"]
    a = d1.sort_values(key).reset_index(drop=True)
    b = d2.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # benign cells decode with every programmatic decoder; encode cells with their own
    assert "decoded_prog_acrostic" in a.columns
    assert a[(a.mechanism == "acrostic")]["decoded_programmatic"].notna().any()
