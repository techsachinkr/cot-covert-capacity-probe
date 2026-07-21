import numpy as np

from ccap.capacity import (
    achieved_position_rate, estimate_capacity, mi_bits, trace_entropy_bits,
)
from ccap.text_utils import ERASURE


def make_channel(n, k, flip01, flip10, seed=0):
    """Generate (intended, decoded) for a per-position channel with given flip probs."""
    rng = np.random.default_rng(seed)
    intended, decoded = [], []
    for _ in range(n):
        b = tuple(int(x) for x in rng.integers(0, 2, k))
        d = []
        for bit in b:
            r = rng.random()
            if bit == 0:
                d.append(1 if r < flip01 else 0)
            else:
                d.append(0 if r < flip10 else 1)
        intended.append(b)
        decoded.append(tuple(d))
    return intended, decoded


def test_mi_perfect_and_independent():
    # perfect binary channel: joint is diagonal -> 1 bit
    perfect = np.array([[500, 0], [0, 500]])
    plugin, mm = mi_bits(perfect)
    assert abs(plugin - 1.0) < 1e-6
    # independent: uniform joint -> 0 bits
    indep = np.array([[250, 250], [250, 250]])
    plugin, mm = mi_bits(indep)
    assert abs(plugin) < 1e-9


def test_capacity_perfect_channel():
    intended, decoded = make_channel(600, 4, 0.0, 0.0, seed=1)
    res = estimate_capacity(intended, decoded)
    assert res.capacity_bits > 3.8
    assert res.achieved_bits > 3.8  # noiseless -> repetition factor 1
    assert res.erasure_rate == 0.0


def test_capacity_independent_collapses():
    rng = np.random.default_rng(2)
    n, k = 600, 4
    intended = [tuple(int(x) for x in rng.integers(0, 2, k)) for _ in range(n)]
    decoded = [tuple(int(x) for x in rng.integers(0, 2, k)) for _ in range(n)]  # independent
    res = estimate_capacity(intended, decoded)
    assert res.capacity_bits < 0.3


def test_channel_asymmetry_reported():
    intended, decoded = make_channel(2000, 3, flip01=0.30, flip10=0.0, seed=3)
    res = estimate_capacity(intended, decoded)
    assert res.per_position_p01[0] > 0.2
    assert res.per_position_p10[0] < 0.1
    assert res.asymmetry > 0.15  # not a binary-symmetric channel


def test_achieved_is_lower_bound_on_capacity():
    # BSC with q=0.1: capacity per position = 1 - H(0.1) ~ 0.531
    intended, decoded = make_channel(3000, 4, 0.1, 0.1, seed=4)
    res = estimate_capacity(intended, decoded, epsilon=0.01)
    assert res.achieved_bits <= res.capacity_bits + 0.05
    assert res.achieved_bits < res.capacity_bits  # repetition code is below capacity here


def test_block_mi_matches_position_sum_when_independent():
    intended, decoded = make_channel(2000, 3, 0.05, 0.05, seed=5)
    res = estimate_capacity(intended, decoded, block_max_k=8)
    assert res.block_capacity_bits is not None
    # positions are independent by construction -> block MI ~ sum of per-position MI
    assert abs(res.block_capacity_bits - res.capacity_bits) < 0.25


def test_erasures_counted_not_swept_under_bsc():
    intended, decoded = make_channel(400, 2, 0.0, 0.0, seed=6)
    # force half the second position to erasure
    decoded = [(d[0], ERASURE if i % 2 == 0 else d[1]) for i, d in enumerate(decoded)]
    res = estimate_capacity(intended, decoded)
    assert res.erasure_rate > 0.2
    assert res.per_position_erasure[1] > 0.4
    assert res.capacity_bits < 2.0  # erasures reduce recoverable bits


def test_achieved_position_rate_repetition():
    assert achieved_position_rate(0.0, 0.01, 21) == 1.0    # noiseless -> rate 1
    assert achieved_position_rate(0.5, 0.01, 21) == 0.0    # chance -> rate 0
    r = achieved_position_rate(0.1, 0.01, 21)
    assert 0.0 < r <= 0.5                                   # needs repetition


def test_trace_entropy_positive():
    assert trace_entropy_bits("") == 0.0
    assert trace_entropy_bits("aaaa") == 0.0               # zero per-char entropy
    assert trace_entropy_bits("abcd abcd") > 0.0
