import numpy as np

from ccap.reliability import bootstrap_c_ctrl_ci, exact_match_rate, reliable_knee
from ccap.text_utils import ERASURE


def _payloads(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(int(b) for b in rng.integers(0, 2, k)) for _ in range(n)]


def test_exact_match_rate_perfect_and_flawed():
    intended = _payloads(50, 16, 1)
    assert exact_match_rate(intended, intended, 16) == 1.0
    # flip the last bit of every item: r(16)=0, r(15)=1
    flawed = [b[:-1] + (1 - b[-1],) for b in intended]
    assert exact_match_rate(intended, flawed, 16) == 0.0
    assert exact_match_rate(intended, flawed, 15) == 1.0


def test_knee_perfect_channel_is_max():
    intended = _payloads(80, 16, 2)
    rk = reliable_knee(intended, intended, n_boot=500)
    assert rk.c_ctrl == 16
    # r(k) is a non-increasing prefix
    rs = [rk.r_point[k] for k in rk.ladder]
    assert all(rs[i] >= rs[i + 1] - 1e-9 for i in range(len(rs) - 1))


def test_knee_all_erasure_is_zero():
    intended = _payloads(60, 16, 3)
    decoded = [(ERASURE,) * 16 for _ in intended]
    rk = reliable_knee(intended, decoded, n_boot=300)
    assert rk.c_ctrl == 0


def test_knee_decreases_with_noise():
    intended = _payloads(150, 16, 4)
    rng = np.random.default_rng(5)
    # ~3% per-slot flip
    noisy = [tuple((b ^ 1) if rng.random() < 0.03 else b for b in bits) for bits in intended]
    clean = reliable_knee(intended, intended, n_boot=400).c_ctrl
    dirty = reliable_knee(intended, noisy, n_boot=400).c_ctrl
    assert dirty < clean
    assert dirty >= 0


def test_knee_ci_brackets_point():
    intended = _payloads(120, 16, 6)
    rng = np.random.default_rng(7)
    noisy = [tuple((b ^ 1) if rng.random() < 0.05 else b for b in bits) for bits in intended]
    point, lo, hi = bootstrap_c_ctrl_ci(intended, noisy, n_boot=400)
    assert lo <= point <= hi
