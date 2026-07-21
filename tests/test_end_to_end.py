"""Full pre-registered pipeline test on the mock backend (truncation design)."""

import pytest

from ccap.analysis.aggregate import AggConfig, aggregate_all
from ccap.config import EstimationConfig
from ccap.grid import BENIGN, GridSpec
from ccap.models import MockBackend
from ccap.runner import ExperimentRunner


@pytest.fixture(scope="module")
def agg(tmp_path_factory):
    grid = GridSpec(
        models=["MC-think", "MC-nothink"],
        mechanisms=[BENIGN, "acrostic", "enumeration", "framing"],
        tasks=["math"], slots=16, ladder=(1, 2, 4, 8, 16), n_items=100,
    )
    backends = {
        "MC-think": MockBackend(name="MC-think", task_competence=0.95, fidelity={
            "acrostic": 0.9997, "enumeration": 0.997, "framing": 0.985}),
        "MC-nothink": MockBackend(name="MC-nothink", task_competence=0.9, fidelity={
            "acrostic": 0.997, "enumeration": 0.985, "framing": 0.965}),
    }
    runner = ExperimentRunner(backends, verbose=False)
    out = tmp_path_factory.mktemp("run")
    df = runner.run(grid, output_dir=str(out))
    cfg = AggConfig(estimation=EstimationConfig(n_boot=600),
                    contrasts={"think_pairs": [["MC-think", "MC-nothink"]]})
    return aggregate_all(df, cfg)


def test_outputs_present(agg):
    for key in ["cells", "capacity_map", "reliability", "benign", "sentinel",
                "compliance", "exclusions", "hypotheses"]:
        assert key in agg


def test_capacity_ordering_lexical_above_conceptual(agg):
    cm = agg["capacity_map"]
    think = cm[cm.model == "MC-think"].set_index("mechanism")["c_ctrl"]
    assert think["acrostic"] > think["framing"]           # lexical > conceptual
    assert think["acrostic"] >= think["enumeration"]      # lexical >= structural
    assert think["acrostic"] >= 8                          # high-fidelity lexical carries many bits


def test_think_beats_nothink(agg):
    cm = agg["capacity_map"].pivot_table(index="model", columns="mechanism", values="c_ctrl")
    assert cm.loc["MC-think", "acrostic"] >= cm.loc["MC-nothink", "acrostic"]


def test_benign_collapses_to_zero(agg):
    ben = agg["benign"]
    assert not ben.empty
    assert ben["c_ctrl"].max() < 1.0          # H4: no recoverable bits from benign traces
    assert ben["c_ctrl_hi"].max() < 1.0       # even the upper CI is below R_min


def test_validation_hypotheses_pass(agg):
    h = agg["hypotheses"].set_index("hypothesis")
    assert bool(h.loc["H4", "supported"])
    assert bool(h.loc["H5", "supported"])


def test_h2_think_effect_supported(agg):
    h2 = agg["hypotheses"]
    row = h2[(h2.hypothesis == "H2") & (h2.contrast.str.contains("MC-think"))]
    assert not row.empty
    assert bool(row["supported"].iloc[0])     # robust given the large think/nothink gap


def test_sentinel_recovers_nothing(agg):
    s = agg["sentinel"]
    assert (s["r_sent"] < 1.0).all()          # no-codebook decoder recovers ~0 bits


def test_sender_compliance_high_for_lexical(agg):
    c = agg["compliance"]
    think = c[c.model == "MC-think"].set_index("mechanism")
    assert think.loc["acrostic", "sender_compliance"] > 0.9


def test_lazy_backend_factory_loads_one_model_at_a_time():
    calls, closed = [], []

    class TrackedMock(MockBackend):
        def close(self):
            closed.append(self.name)

    def factory(name):
        calls.append(name)
        return TrackedMock(name=name, fidelity=0.9)

    grid = GridSpec(models=["A", "B"], mechanisms=["acrostic"], tasks=["math"],
                    slots=4, ladder=(1, 2, 4), n_items=5)
    runner = ExperimentRunner(backends={}, backend_factory=factory, verbose=False)
    df = runner.run(grid)
    assert calls == ["A", "B"]
    assert closed == ["A", "B"]
    assert len(df) == 2 * 5
