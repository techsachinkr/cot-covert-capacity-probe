import numpy as np

from ccap.stats import benjamini_hochberg, bonferroni, bootstrap_capacity_ci, permutation_pvalue


def _channel(n, k, flip, seed):
    rng = np.random.default_rng(seed)
    intended, decoded = [], []
    for _ in range(n):
        b = tuple(int(x) for x in rng.integers(0, 2, k))
        d = tuple((bit ^ 1) if rng.random() < flip else bit for bit in b)
        intended.append(b)
        decoded.append(d)
    return intended, decoded


def test_permutation_detects_signal():
    intended, decoded = _channel(200, 2, flip=0.05, seed=1)
    p = permutation_pvalue(intended, decoded, n_perm=200, seed=0)
    assert p < 0.05


def test_permutation_null_is_calibrated_on_independent():
    # Under H0 the p-value is ~Uniform(0,1); a single realization is intentionally
    # flaky, so we check calibration across many independent datasets: the mean
    # p-value should sit near 0.5 and false positives at alpha=0.05 should be rare.
    pvals = []
    for s in range(16):
        rng = np.random.default_rng(100 + s)
        intended = [tuple(int(x) for x in rng.integers(0, 2, 2)) for _ in range(200)]
        decoded = [tuple(int(x) for x in rng.integers(0, 2, 2)) for _ in range(200)]
        pvals.append(permutation_pvalue(intended, decoded, n_perm=150, seed=s))
    pvals = np.array(pvals)
    assert pvals.mean() > 0.3                       # not systematically significant
    assert np.mean(pvals < 0.05) <= 0.25            # FPR roughly controlled


def test_bootstrap_ci_brackets_point():
    intended, decoded = _channel(300, 3, flip=0.1, seed=3)
    ci = bootstrap_capacity_ci(intended, decoded, n_boot=200, seed=0)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.hi - ci.lo > 0


def test_benjamini_hochberg_basic():
    pvals = [0.001, 0.02, 0.5, 0.9]
    rejected, q = benjamini_hochberg(pvals, alpha=0.05)
    assert rejected[0] and rejected[1]
    assert not rejected[2] and not rejected[3]
    # q-values are valid probabilities
    assert np.all((q >= 0) & (q <= 1))


def test_bonferroni_basic():
    pvals = [0.001, 0.2]
    rejected, adj = bonferroni(pvals, alpha=0.05)
    assert rejected[0] and not rejected[1]
    assert abs(adj[0] - 0.002) < 1e-9


def test_holm_bonferroni_step_down():
    from ccap.stats import holm_bonferroni
    rejected, adj = holm_bonferroni([0.001, 0.02, 0.5, 0.9], alpha=0.05)
    # 0.001*4 = 0.004 <= 0.05 (reject); 0.02*3 = 0.06 > 0.05 (stop) -> only first rejected
    assert rejected[0] and not rejected[1] and not rejected[2]
    assert np.all((adj >= 0) & (adj <= 1))
    assert abs(adj[0] - 0.004) < 1e-9


def test_jonckheere_terpstra_detects_increasing_trend():
    from ccap.stats import jonckheere_terpstra
    up = [np.array([1.0, 2, 1.5]), np.array([3.0, 4, 3.5]), np.array([6.0, 5, 7])]
    J, z, p = jonckheere_terpstra(up, "increasing")
    assert z > 0 and p < 0.05
    flat = [np.array([5.0, 5, 5]), np.array([5.0, 5, 5]), np.array([5.0, 5, 5])]
    _, _, p_flat = jonckheere_terpstra(flat, "increasing")
    assert p_flat > 0.1
