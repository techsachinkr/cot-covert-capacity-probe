"""LaTeX tables: mechanism taxonomy (Table 1), capacity map, sentinel (S6),
and the confirmatory hypothesis results (H1-H5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..mechanisms import all_mechanisms


def _esc(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("Ĉ", r"$\hat{C}$"))


def _write(tex: str, out_path) -> str:
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(tex, encoding="utf-8")
    return tex


def table_mechanisms(out_path=None) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mechanism taxonomy (encoding contracts). M0 is the no-payload control.}",
        r"\label{tab:mechanisms}", r"\small", r"\begin{tabular}{llc}",
        r"\toprule", r"ID & Mechanism (bit $\rightarrow$ feature) & Decoder \\", r"\midrule",
        r"M0 & Benign control (no payload) & -- \\",
    ]
    # M2 is the TAGLESS semantic ordering (ordering_semantic); the tagged `ordering` is
    # R1, the reference anchor (§2.2) — NOT M2. The first cut mislabeled `ordering` as M2
    # and dropped the real M2 entirely.
    ids = {"acrostic": "M1", "ordering_semantic": "M2", "framing": "M3", "abstraction": "M4",
           "enumeration": "M5", "hedging": "M6", "ordering": "R1"}
    order = ["acrostic", "ordering_semantic", "framing", "abstraction", "enumeration",
             "hedging", "ordering"]
    by_name = {m.name: m for m in all_mechanisms()}
    for nm in order:
        m = by_name[nm]
        lines.append(f"{ids[nm]} & {_esc(m.label)}: {_esc(m.codebook_description())} & {m.decoder} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return _write("\n".join(lines), out_path)


def table_capacity_map(capacity_map: pd.DataFrame, mechanisms, models, out_path=None) -> str:
    mechs = [m for m in mechanisms if m != "benign"]
    piv = capacity_map.pivot_table(index="model", columns="mechanism", values="c_ctrl")
    piv = piv.reindex(index=[m for m in models if m in piv.index],
                      columns=[c for c in mechs if c in piv.columns])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Controllable covert capacity $\hat{C}_{\text{ctrl}}$ (reliable bits/response, "
        r"exact-match knee), model-condition $\times$ mechanism.}",
        r"\label{tab:capacity_map}", r"\small",
        r"\begin{tabular}{l" + "r" * len(piv.columns) + "}", r"\toprule",
        "Model-condition & " + " & ".join(_esc(c) for c in piv.columns) + r" \\", r"\midrule",
    ]
    for model, row in piv.iterrows():
        cells = " & ".join("--" if pd.isna(v) else f"{v:.0f}" for v in row)
        lines.append(f"{_esc(model)} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return _write("\n".join(lines), out_path)


def table_sentinel(sentinel_df: pd.DataFrame, out_path=None) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Sentinel gap (exploratory, S6). $R_{\text{recv}}$/$R_{\text{sent}}$ are "
        r"reliable rates (bits/response) with/without the codebook; "
        r"$\Delta_{\text{op}}=R_{\text{recv}}-R_{\text{sent}}$; FPR/TPR for the codebook-free detector.}",
        r"\label{tab:sentinel}", r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & Mech & $R_{\text{recv}}$ & $R_{\text{sent}}$ & $\Delta_{\text{op}}$ & FPR & TPR \\",
        r"\midrule",
    ]
    agg = sentinel_df.groupby(["model", "mechanism"], as_index=False).mean(numeric_only=True) \
        if not sentinel_df.empty else sentinel_df
    for _, r in agg.iterrows():
        fpr = "--" if pd.isna(r.get("fpr")) else f"{r['fpr']:.2f}"
        tpr = "--" if pd.isna(r.get("tpr")) else f"{r['tpr']:.2f}"
        lines.append(f"{_esc(r['model'])} & {_esc(r['mechanism'])} & {r['r_recv']:.1f} & "
                     f"{r['r_sent']:.1f} & {r['delta_op']:.1f} & {fpr} & {tpr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return _write("\n".join(lines), out_path)


def table_hypotheses(hyp: pd.DataFrame, out_path=None) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Confirmatory hypotheses under Holm-Bonferroni (directional one-sided "
        r"$\alpha=0.025$). H4/H5 are pass/fail validations.}",
        r"\label{tab:hypotheses}", r"\small", r"\begin{tabular}{llrrc}", r"\toprule",
        r"H & Contrast & Effect & Holm $p$ & Supported \\", r"\midrule",
    ]
    for _, r in hyp.iterrows():
        hp = "--" if pd.isna(r.get("holm_p")) else f"{r['holm_p']:.3f}"
        eff = "--" if pd.isna(r.get("effect")) else f"{r['effect']:.2f}"
        sup = r"\checkmark" if r.get("supported") else r"$\times$"
        lines.append(f"{_esc(r['hypothesis'])} & {_esc(r['contrast'])} & {eff} & {hp} & {sup} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return _write("\n".join(lines), out_path)


def perslot_accuracy(trials_df: pd.DataFrame, decoded_col: dict) -> dict:
    """Mean per-slot decode accuracy vs the intended payload, per mechanism.

    ``decoded_col`` maps mechanism name -> the column to read
    (``decoded_programmatic`` for surface channels, ``decoded_model`` for conceptual).
    Erasures / short decodes count as wrong. Used for Table~\\ref{tab:graded}.
    """
    from ..payloads import csv_to_bits, str_to_bits
    df = trials_df[trials_df.get("mode") == "encode"] if "mode" in trials_df else trials_df
    if "x_excluded" in df:
        df = df[~df["x_excluded"].astype(str).isin(["1", "True", "true"])]
    out = {}
    for mech, col in decoded_col.items():
        s = df[df["mechanism"] == mech]
        c = t = 0
        for _, r in s.iterrows():
            bi = str_to_bits(str(r["intended"])); bd = csv_to_bits(r.get(col, ""))
            for j in range(len(bi)):
                t += 1
                if j < len(bd) and bd[j] == bi[j]:
                    c += 1
        out[mech] = (c / t) if t else float("nan")
    return out


def table_graded_recovery(cells: pd.DataFrame, acc_primary: dict, acc_llama: dict,
                          order=None, out_path=None) -> str:
    """Graded / decoder-relative recovery table (per mechanism, mean over the grid).

    Shows that the exact-match knee Ĉ dichotomizes while MI, the coded rate, and per-slot
    accuracy are graded, and that conceptual channels are decoder-relative. ``acc_primary``
    and ``acc_llama`` come from :func:`perslot_accuracy` on the primary and non-family runs.
    """
    from ..mechanisms import get_mechanism
    if order is None:
        order = ["hedging", "acrostic", "enumeration", "ordering",
                 "framing", "abstraction", "ordering_semantic"]
    enc = cells[cells["mechanism"] != "benign"]
    agg = enc.groupby("mechanism").agg(c=("c_ctrl", "mean"), mi=("mi_bits", "mean"),
                                       coded=("achieved_bits", "mean"))

    def fmt(x):
        return "---" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.2f}"

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Graded, decoder-relative recovery per mechanism (mean over the grid). "
        r"$\hat{C}_{\text{ctrl}}$ dichotomizes; MI (Miller--Madow), the repetition-coded rate "
        r"(coded), and per-slot accuracy are graded; conceptual channels are decoder-relative "
        r"(primary vs Llama-3.3-70B).}",
        r"\label{tab:graded}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"Mechanism & Dec. & $\hat{C}$ & MI & coded & acc$_{\text{pri}}$ & acc$_{\text{Lla}}$ \\",
        r"\midrule",
    ]
    for m in order:
        if m not in agg.index:
            continue
        dec = "model" if get_mechanism(m).decoder == "model" else "prog"
        r = agg.loc[m]
        al = acc_llama.get(m) if dec == "model" else None
        lines.append(f"{_esc(m)} & {dec} & {r['c']:.1f} & {r['mi']:.1f} & {r['coded']:.1f} & "
                     f"{fmt(acc_primary.get(m))} & {fmt(al)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return _write("\n".join(lines), out_path)


def make_all_tables(agg: dict, mechanisms, models, out_dir="figures") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "table1_mechanisms": table_mechanisms(out / "table1_mechanisms.tex"),
        "table_capacity_map": table_capacity_map(agg["capacity_map"], mechanisms, models,
                                                 out / "table_capacity_map.tex"),
    }
    if "sentinel" in agg:
        res["table2_sentinel"] = table_sentinel(agg["sentinel"], out / "table2_sentinel.tex")
    if "hypotheses" in agg:
        res["table_hypotheses"] = table_hypotheses(agg["hypotheses"], out / "table_hypotheses.tex")
    return res
