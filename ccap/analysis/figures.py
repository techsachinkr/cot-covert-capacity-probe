"""Figure generation (Figs 1-4). Matplotlib imported lazily (``pip install ccap[viz]``).

Consumes the aggregated tables from :mod:`ccap.analysis.aggregate`:
  * Fig 1  pipeline schematic
  * Fig 2  Ĉ_ctrl heatmap (MC x mechanism)          <- capacity_map
  * Fig 3  reliability r(k) vs k + the 0.95 knee     <- reliability
  * Fig 4  benign separation: Ĉ_ctrl(M0) ~ 0 vs trace entropy   <- benign
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..grid import BENIGN


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_path: str | Path) -> list[str]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    paths = []
    for ext in ("pdf", "png"):
        p = out.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(str(p))
    plt.close(fig)
    return paths


def fig_pipeline(out_path="figures/fig1_pipeline") -> list[str]:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")
    boxes = [("Payload\n(S=16 slots)", "B ~ Uniform{0,1}^16"),
             ("Mechanism\nencoder prompt", "bits -> CoT feature"),
             ("Sender CoT\n(1 generation)", "solves task +\nembeds 16 slots"),
             ("Decoder", "programmatic /\nmodel codebook"),
             ("Ĉ_ctrl estimator", "exact-match knee\nr(k)>=0.95")]
    n = len(boxes)
    for i, (title, sub) in enumerate(boxes):
        x = i / n
        ax.add_patch(plt.Rectangle((x + 0.01, 0.25), 1 / n - 0.04, 0.5, fill=True,
                                   facecolor="#eef3fb", edgecolor="#33527a", lw=1.5))
        cx = x + (1 / n - 0.03) / 2 + 0.01
        ax.text(cx, 0.62, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(cx, 0.40, sub, ha="center", va="center", fontsize=6.8, color="#333")
        if i < n - 1:
            ax.annotate("", xy=(x + 1 / n - 0.02, 0.5), xytext=(x + 1 / n - 0.035, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#33527a"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("CCAP measurement pipeline (truncation design)", fontsize=11)
    return _save(fig, out_path)


def fig_capacity_heatmap(capacity_map: pd.DataFrame, mechanisms, models,
                         out_path="figures/fig2_capacity_heatmap") -> list[str]:
    plt = _mpl()
    mechs = [m for m in mechanisms if m != BENIGN]
    M = np.full((len(models), len(mechs)), np.nan)
    for _, r in capacity_map.iterrows():
        if r["model"] in models and r["mechanism"] in mechs:
            M[models.index(r["model"]), mechs.index(r["mechanism"])] = r["c_ctrl"]
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(mechs), 1.0 + 0.55 * len(models)))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mechs))); ax.set_xticklabels(mechs, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
    for i in range(len(models)):
        for j in range(len(mechs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < vmax * 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Ĉ_ctrl (reliable bits / response)", fontsize=8)
    ax.set_title("Controllable covert capacity per mechanism", fontsize=10)
    return _save(fig, out_path)


def fig_reliability(reliability: pd.DataFrame, epsilon: float = 0.05, model=None,
                    out_path="figures/fig3_reliability") -> list[str]:
    plt = _mpl()
    if reliability.empty:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "no reliability data", ha="center")
        return _save(fig, out_path)
    if model is None:
        model = sorted(reliability["model"].unique())[0]
    rel = (reliability[reliability["model"] == model]
           .groupby(["mechanism", "k"], as_index=False)[["r", "r_lo"]].mean())
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(sorted(rel["mechanism"].unique())):
        d = rel[rel["mechanism"] == m].sort_values("k")
        c = cmap(i % 10)
        ax.plot(d["k"], d["r"], "-o", color=c, ms=4, label=m)
        ax.fill_between(d["k"], d["r_lo"], d["r"], color=c, alpha=0.12)
    ax.axhline(1 - epsilon, color="grey", ls="--", lw=1)
    ax.text(rel["k"].min(), 1 - epsilon - 0.03, f"reliability target 1-ε={1 - epsilon:.2f}",
            ha="left", va="top", fontsize=7, color="grey")
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(rel["k"].unique()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("payload size k (bits)"); ax.set_ylabel("exact-match recovery r(k)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Reliability curves and the Ĉ_ctrl knee — {model}", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    return _save(fig, out_path)


def fig_benign(benign: pd.DataFrame, out_path="figures/fig4a_benign") -> list[str]:
    plt = _mpl()
    if benign.empty:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "no benign data", ha="center")
        return _save(fig, out_path)
    g = benign.groupby("model", as_index=False).agg(
        c_ctrl=("c_ctrl", "max"), entropy=("trace_entropy_bits", "mean"))
    x = np.arange(len(g))
    fig, ax1 = plt.subplots(figsize=(1.6 + 1.0 * len(g), 3.6))
    ax1.bar(x - 0.2, g["c_ctrl"], 0.4, color="#c0563b", label="benign Ĉ_ctrl (bits)")
    ax1.set_ylabel("benign Ĉ_ctrl (bits/response)", color="#c0563b")
    ax1.set_ylim(0, max(1.0, g["c_ctrl"].max() * 1.3 + 0.5))
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, g["entropy"], 0.4, color="#3b6fb0", label="trace entropy (bits)")
    ax2.set_ylabel("trace Shannon entropy (bits)", color="#3b6fb0")
    ax1.set_xticks(x); ax1.set_xticklabels(g["model"], rotation=30, ha="right", fontsize=8)
    ax1.set_title("Style vs capacity (H4): benign Ĉ_ctrl ≈ 0 while trace entropy is large", fontsize=8.5)
    return _save(fig, out_path)


def make_all_figures(agg: dict, mechanisms, models, out_dir="figures") -> dict:
    out = Path(out_dir)
    eps = 0.05
    return {
        "fig1_pipeline": fig_pipeline(out / "fig1_pipeline"),
        "fig2_capacity_heatmap": fig_capacity_heatmap(agg["capacity_map"], mechanisms, models,
                                                      out / "fig2_capacity_heatmap"),
        "fig3_reliability": fig_reliability(agg["reliability"], eps, out_path=out / "fig3_reliability"),
        "fig4a_benign": fig_benign(agg["benign"], out / "fig4a_benign"),
    }
