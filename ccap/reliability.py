"""Primary metric P1 — controllable reliable rate via exact-match recovery (§5).

The pre-registered headline number is assumption-light:

    r(k)        = P(the first k payload bits are ALL recovered exactly)   [§5]
    Ĉ_ctrl      = the largest k in the ladder with the bootstrap LOWER-CI
                  bound of r(k) >= 1 - epsilon   (epsilon = 0.05)

No independence or symmetry assumption is made: r(k) is just the operational
fraction of responses whose first k slots the constructed sender->receiver pair
transmit exactly. Erasures (missing/garbled slots) count as failures (§8). The CI
is a bootstrap over the n=60 items (clustered by item), so cross-condition
contrasts are paired on the shared item pool.

This module supersedes the mutual-information estimator in :mod:`ccap.capacity`,
which is retained only as the S1 per-slot MI cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .text_utils import ERASURE
from .types import Bits

DEFAULT_LADDER = (1, 2, 4, 8, 16)


def _success_indicators(intended: list[Bits], decoded: list[Bits], k: int) -> np.ndarray:
    """Per-item 1.0/0.0: were the first k bits all recovered exactly (erasure=fail)?"""
    out = np.zeros(len(intended), dtype=float)
    for i, (bi, bd) in enumerate(zip(intended, decoded)):
        ok = True
        for j in range(k):
            if j >= len(bi) or j >= len(bd) or bd[j] == ERASURE or bd[j] != bi[j]:
                ok = False
                break
        out[i] = 1.0 if ok else 0.0
    return out


def exact_match_rate(intended: list[Bits], decoded: list[Bits], k: int) -> float:
    if not intended:
        return 0.0
    return float(_success_indicators(intended, decoded, k).mean())


def _bootstrap_ci(succ: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(succ)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(succ.mean())
    idx = rng.integers(0, n, size=(n_boot, n))      # resample items (clustered by item)
    boots = succ[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


@dataclass
class ReliabilityResult:
    n: int
    ladder: tuple[int, ...]
    r_point: dict[int, float] = field(default_factory=dict)
    r_lo: dict[int, float] = field(default_factory=dict)
    r_hi: dict[int, float] = field(default_factory=dict)
    c_ctrl: int = 0            # headline: largest k with r_lo(k) >= 1 - eps (conservative)
    c_ctrl_point: int = 0      # optimistic knee at the point estimate (bracket upper)
    c_ctrl_lo: int = 0         # alias of c_ctrl (lower/safe end of the bracket)
    epsilon: float = 0.05

    def to_row(self) -> dict:
        row = {"n": self.n, "c_ctrl": self.c_ctrl, "c_ctrl_point": self.c_ctrl_point,
               "epsilon": self.epsilon}
        for k in self.ladder:
            row[f"r{k}"] = self.r_point.get(k)
            row[f"r{k}_lo"] = self.r_lo.get(k)
            row[f"r{k}_hi"] = self.r_hi.get(k)
        return row


def reliable_knee(
    intended: list[Bits],
    decoded: list[Bits],
    ladder=DEFAULT_LADDER,
    epsilon: float = 0.05,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ReliabilityResult:
    """Compute r(k) with bootstrap CIs and the conservative reliable-knee Ĉ_ctrl."""
    ladder = tuple(sorted(set(ladder)))
    res = ReliabilityResult(n=len(intended), ladder=ladder, epsilon=epsilon)
    thresh = 1.0 - epsilon
    rng = np.random.default_rng(seed)
    prefix_lo_ok = True   # enforce the monotone prefix (r(k) is non-increasing in k)
    prefix_pt_ok = True
    for k in ladder:
        succ = _success_indicators(intended, decoded, k)
        p, lo, hi = _bootstrap_ci(succ, n_boot, alpha, rng)
        res.r_point[k], res.r_lo[k], res.r_hi[k] = p, lo, hi
        if prefix_lo_ok and lo >= thresh:
            res.c_ctrl = k
        else:
            prefix_lo_ok = False
        if prefix_pt_ok and p >= thresh:
            res.c_ctrl_point = k
        else:
            prefix_pt_ok = False
    res.c_ctrl_lo = res.c_ctrl
    return res


def bootstrap_c_ctrl_ci(
    intended: list[Bits],
    decoded: list[Bits],
    ladder=DEFAULT_LADDER,
    epsilon: float = 0.05,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap distribution of the knee Ĉ_ctrl itself (point, lo, hi).

    Used for the §7 effect tests (H2 non-overlapping CIs; H3 trend) which compare
    the knee across conditions rather than r(k) at a fixed k.
    """
    n = len(intended)
    if n == 0:
        return 0.0, 0.0, 0.0
    ladder = tuple(sorted(set(ladder)))
    thresh = 1.0 - epsilon
    succ_by_k = {k: _success_indicators(intended, decoded, k) for k in ladder}
    rng = np.random.default_rng(seed)
    point = float(reliable_knee(intended, decoded, ladder, epsilon, n_boot=1, seed=seed).c_ctrl_point)
    knees = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        knee = 0
        for k in ladder:
            if succ_by_k[k][idx].mean() >= thresh:
                knee = k
            else:
                break
        knees[b] = knee
    lo = float(np.quantile(knees, alpha / 2))
    hi = float(np.quantile(knees, 1 - alpha / 2))
    return point, lo, hi
