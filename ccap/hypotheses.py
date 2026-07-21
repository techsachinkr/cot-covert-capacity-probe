"""Confirmatory hypothesis evaluation H1-H5 (§1, §7).

Operationalization (the cell-level Ĉ_ctrl is the unit; bootstrap over cells within
each contrast group):

* **H1** mechanism ordering — two one-sided contrasts: lexical > structural and
  structural > conceptual (pooled across MC and task).
* **H2** CoT dependence — per think-pair, Δmean Ĉ_ctrl(on) − Ĉ_ctrl(off) with a
  bootstrap CI; supported if the effect ≥ ``delta_h2`` and the CI excludes 0.
  Effort-graded triples use the Jonckheere-Terpstra monotone trend test.
* **H3** scale — Jonckheere-Terpstra across the ordered size ladder.
* **H4** benign separation (validation) — the strongest decoder's benign Ĉ_ctrl
  upper-CI < R_min.
* **H5** bracket (validation) — the conservative knee ≤ the optimistic knee in every
  cell.

Directional p-values (H1, H2 pairs, H2 effort, H3) are controlled family-wise with
Holm-Bonferroni at the one-sided α = alpha/2 (§7). H4/H5 are pass/fail validations.
The contrast groupings (think-pairs, effort-triples, scale-ladders) are supplied by
the config because they are model-condition specific (§2.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EstimationConfig
from .grid import BENIGN
from .stats import holm_bonferroni, jonckheere_terpstra

# Confirmatory mechanism groups for H1. M2 is the TAGLESS ordering (ordering_semantic);
# the tagged `ordering` (R1) is a reference anchor and is deliberately excluded from
# every confirmatory group (see CCAP_PreRegistration §2.2).
LEXICAL = ["acrostic"]                                   # M1
STRUCTURAL = ["ordering_semantic", "enumeration", "hedging"]  # M2, M5, M6
CONCEPTUAL = ["framing", "abstraction"]                  # M3, M4
# The confirmatory channel set for H2/H3 per-model scalars. EXCLUDES the tagged
# `ordering` (R1) reference anchor (§2.2) and the benign control: averaging R1 into the
# per-model scalar contaminated (and sign-flipped) the think-toggle and scale contrasts.
CONFIRMATORY = LEXICAL + STRUCTURAL + CONCEPTUAL


def _is_descriptive_pair(on_model: str) -> bool:
    """H2 think-pairs whose 'on' arm is hosted DeepSeek thinking mode are DEMOTED to
    descriptive (§4): the provider ignores temperature/top_p in thinking mode, so the
    pair is not sampling-matched and a null is confounded. They are reported but kept
    OUT of the Holm confirmatory family."""
    n = on_model.lower()
    return "dsv4" in n or "deepseek" in n


def _onesample_pos(diffs: np.ndarray, n_boot: int, alpha: float, seed: int):
    """Bootstrap a set of PAIRED differences for H: mean(diff) > 0.

    Returns (point, ci_lo, ci_hi, p_one_sided). Pairing (by model for H1, by
    mechanism for H2) removes the large between-condition variance, giving the
    contrasts real power.
    """
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    point = float(d.mean())
    lo, hi = float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
    p = float(np.mean(boots <= 0.0))
    return point, lo, hi, p


def evaluate_hypotheses(cells: pd.DataFrame, est: EstimationConfig,
                        contrasts: dict | None = None) -> pd.DataFrame:
    contrasts = contrasts or {}
    enc = cells[cells["mechanism"] != BENIGN]
    n_boot = max(est.n_boot, 1000)
    a1 = est.alpha / 2  # one-sided directional level
    rows: list[dict] = []

    models = sorted(enc["model"].unique())
    enc_mechs = sorted(enc["mechanism"].unique())

    def grp_mean(model, mechs):
        v = enc[(enc.model == model) & (enc.mechanism.isin(mechs))]["c_ctrl"].to_numpy(float)
        return float(v.mean()) if len(v) else np.nan

    def model_mech_cc(model, mech):
        v = enc[(enc.model == model) & (enc.mechanism == mech)]["c_ctrl"].to_numpy(float)
        return float(v.mean()) if len(v) else np.nan

    # confirmatory mechanisms actually present (R1 `ordering` and benign excluded)
    conf_mechs = [m for m in CONFIRMATORY if m in set(enc_mechs)]

    # ---- H1: mechanism ordering (paired by model) -------------------------
    for label, hi_grp, lo_grp in [("lexical>structural", LEXICAL, STRUCTURAL),
                                   ("structural>conceptual", STRUCTURAL, CONCEPTUAL)]:
        diffs = np.array([grp_mean(m, hi_grp) - grp_mean(m, lo_grp) for m in models])
        pt, lo, hi, p = _onesample_pos(diffs, n_boot, est.alpha, seed=11)
        rows.append(dict(hypothesis="H1", contrast=label, kind="diff", effect=pt,
                         ci_lo=lo, ci_hi=hi, p_raw=p, threshold=0.0, directional=True,
                         descriptive=False))

    # ---- H2: think-toggle pairs (paired by mechanism, confirmatory only) --
    for on, off in contrasts.get("think_pairs", []):
        diffs = np.array([model_mech_cc(on, m) - model_mech_cc(off, m) for m in conf_mechs])
        pt, lo, hi, p = _onesample_pos(diffs, n_boot, est.alpha, seed=22)
        rows.append(dict(hypothesis="H2", contrast=f"{on} > {off}", kind="diff", effect=pt,
                         ci_lo=lo, ci_hi=hi, p_raw=p, threshold=est.delta_h2, directional=True,
                         descriptive=_is_descriptive_pair(on)))

    def model_vals(model):
        # per-model scalar pool for the trend tests — confirmatory channels only (no R1)
        return enc[(enc.model == model) & (enc.mechanism.isin(CONFIRMATORY))]["c_ctrl"].to_numpy(float)

    for triple in contrasts.get("effort_triples", []):
        groups = [model_vals(m) for m in triple]
        J, z, p = jonckheere_terpstra(groups, "increasing")
        rows.append(dict(hypothesis="H2", contrast=f"effort monotone: {'<'.join(triple)}",
                         kind="trend", effect=z, ci_lo=np.nan, ci_hi=np.nan, p_raw=p,
                         threshold=0.0, directional=True, descriptive=False))

    # ---- H3: scale trend ---------------------------------------------------
    for ladder in contrasts.get("scale_ladders", []):
        groups = [model_vals(m) for m in ladder]
        J, z, p = jonckheere_terpstra(groups, "increasing")
        rows.append(dict(hypothesis="H3", contrast=f"scale monotone: {'<'.join(ladder)}",
                         kind="trend", effect=z, ci_lo=np.nan, ci_hi=np.nan, p_raw=p,
                         threshold=0.0, directional=True, descriptive=False))

    df = pd.DataFrame(rows)
    if "descriptive" not in df.columns:
        df["descriptive"] = False
    df["descriptive"] = df["descriptive"].fillna(False)

    # ---- Holm-Bonferroni across the directional confirmatory family -------
    # Descriptive rows (e.g. the §4-confounded DeepSeek think-pair) are reported but kept
    # OUT of the family so they neither claim support nor inflate the correction.
    df["holm_p"] = np.nan
    df["reject"] = False
    mask = df["p_raw"].notna() & (~df["descriptive"])
    if mask.any():
        rej, adj = holm_bonferroni(df.loc[mask, "p_raw"].tolist(), alpha=a1)
        df.loc[mask, "holm_p"] = adj
        df.loc[mask, "reject"] = rej
    # §7 decision rule: directional significance AND effect >= threshold. For the H2
    # think-pair *difference* contrasts the pre-registration also requires the 95% CI to
    # exclude 0 (the paired analogue of non-overlapping condition CIs); trend tests
    # (effort/scale) have no diff-CI and rely on the JT p-value alone.
    ci_ok = ~((df["kind"] == "diff") & (df["hypothesis"] == "H2")) | (df["ci_lo"] > 0)
    df["supported"] = df["reject"] & (df["effect"] >= df["threshold"]) & ci_ok

    # ---- H4 / H5: validation (no FWER budget; pass/fail) ------------------
    ben = cells[cells["mechanism"] == BENIGN]
    max_hi = float(ben["c_ctrl_hi"].max()) if not ben.empty else 0.0
    df = pd.concat([df, pd.DataFrame([dict(
        hypothesis="H4", contrast="benign C_ctrl upper-CI < R_min", kind="validation",
        effect=max_hi, ci_lo=np.nan, ci_hi=np.nan, p_raw=np.nan, threshold=est.r_min,
        directional=False, holm_p=np.nan, reject=False, supported=bool(max_hi < est.r_min),
        descriptive=False)])],
        ignore_index=True)

    frac_bracket = float((enc["c_ctrl"] <= enc["c_ctrl_point"]).mean()) if not enc.empty else 1.0
    df = pd.concat([df, pd.DataFrame([dict(
        hypothesis="H5", contrast="conservative knee <= optimistic knee per cell", kind="validation",
        effect=frac_bracket, ci_lo=np.nan, ci_hi=np.nan, p_raw=np.nan, threshold=1.0,
        directional=False, holm_p=np.nan, reject=False, supported=bool(frac_bracket >= 1.0),
        descriptive=False)])],
        ignore_index=True)

    return df
