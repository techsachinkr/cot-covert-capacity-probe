"""Aggregation regression tests for the truncation-design schema."""

import numpy as np
import pandas as pd

from ccap.analysis.aggregate import AggConfig, aggregate_all, load_trials
from ccap.config import EstimationConfig
from ccap.payloads import str_to_bits


def test_load_trials_preserves_leading_zero_payloads(tmp_path):
    p = tmp_path / "trials.csv"
    pd.DataFrame([
        {"intended": "0110", "decoded_programmatic": "0,1,1,0", "payload_len": 4, "mode": "encode"},
        {"intended": "0001", "decoded_programmatic": "0,0,0,1", "payload_len": 4, "mode": "encode"},
    ]).to_csv(p, index=False)
    df = load_trials(p)
    assert df["intended"].iloc[0] == "0110" and df["intended"].iloc[1] == "0001"
    assert all(len(str_to_bits(s)) == k for s, k in zip(df["intended"], df["payload_len"]))


def _encode_cell_df(n=60, slots=16, perfect=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        bits = tuple(int(b) for b in rng.integers(0, 2, slots))
        dec = bits if perfect else tuple(b ^ (1 if j == 0 else 0) for j, b in enumerate(bits))
        rows.append({
            "model": "m", "mechanism": "acrostic", "task": "math", "payload_len": slots,
            "mode": "encode", "intended": "".join(map(str, bits)),
            "decoded_programmatic": ",".join(map(str, dec)),
            "decoded_sentinel": ",".join(str(int(x)) for x in rng.integers(0, 2, slots)),
            "task_correct": True, "x_excluded": 0, "x_realized_slots": slots,
            "x_trace_entropy": 100.0, "x_detect_score": 1.0, "x_sender_compliance": 1.0,
        })
    return pd.DataFrame(rows)


def test_perfect_encode_cell_reaches_max_knee():
    df = _encode_cell_df(perfect=True)
    agg = aggregate_all(df, AggConfig(estimation=EstimationConfig(n_boot=200)))
    cm = agg["capacity_map"].set_index("mechanism")["c_ctrl"]
    assert cm["acrostic"] == 16          # perfect recovery -> full 16-bit knee


def test_one_slot_error_caps_the_knee():
    # flipping slot 0 every time makes r(k)=0 for all k>=1 -> Ĉ_ctrl = 0
    df = _encode_cell_df(perfect=False)
    agg = aggregate_all(df, AggConfig(estimation=EstimationConfig(n_boot=200)))
    cm = agg["capacity_map"].set_index("mechanism")["c_ctrl"]
    assert cm["acrostic"] == 0


def test_csv_roundtrip_matches_in_memory(tmp_path):
    df = _encode_cell_df(perfect=True)
    cfg = AggConfig(estimation=EstimationConfig(n_boot=150))
    mem = aggregate_all(df, cfg)["capacity_map"].set_index("mechanism")["c_ctrl"]["acrostic"]
    p = tmp_path / "trials.csv"
    df.to_csv(p, index=False)
    csv = aggregate_all(load_trials(p), cfg)["capacity_map"].set_index("mechanism")["c_ctrl"]["acrostic"]
    assert mem == csv == 16
