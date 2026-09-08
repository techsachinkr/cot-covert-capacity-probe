#!/usr/bin/env python
"""Decoder-robustness check: compare two model-decoder re-decodes of the SAME CoTs.

The conceptual channels (M2 ordering_semantic, M3 framing, M4 abstraction) are read by a
single hosted "model decoder". Because the grid is Qwen-heavy (MC4/MC5 ARE the primary
decoder qwen3-32b), a same-family decoder might self-decode and inflate those channels for
Qwen senders. This compares the primary (Qwen) re-decode against an independent non-Qwen
re-decode of the IDENTICAL saved CoTs and reports, per (model, mechanism):

  * bit-level agreement between the two decoders (the robustness metric),
  * each decoder's erasure rate (empty/failed decode),
  * the resulting C_ctrl side by side (from each run's capacity_map.csv).

High agreement + similar C_ctrl => the single-decoder result is robust (no self-decoding
artifact). Qwen senders scoring high under Qwen but low under the non-Qwen decoder => a
home-field bias to flag.

    python scripts/decoder_agreement.py runs/ccap-main-fixed runs/ccap-main-fixed-llama
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

MODEL_MECHS = {"ordering_semantic", "framing", "abstraction"}


def _bits(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in (s.split(",") if "," in s else list(s)):
        tok = tok.strip()
        out.append(int(tok) if tok in ("0", "1") else -1)
    return out


def _load_decodes(run: str) -> dict:
    """key (model,mechanism,task,trial_index) -> decoded_model bit list (encode, non-excluded)."""
    d = {}
    with open(f"{run}/trials.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mechanism"] not in MODEL_MECHS or r.get("mode") != "encode":
                continue
            if str(r.get("x_excluded", "0")) in ("1", "True", "true"):
                continue
            key = (r["model"], r["mechanism"], r["task"], r["trial_index"])
            d[key] = _bits(r.get("decoded_model", ""))
    return d


def _load_capacity(run: str) -> dict:
    d = {}
    try:
        with open(f"{run}/analysis/capacity_map.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d[(r["model"], r["mechanism"])] = float(r["c_ctrl"])
    except FileNotFoundError:
        pass
    return d


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/decoder_agreement.py <qwen_run_dir> <nonqwen_run_dir>")
    run_a, run_b = sys.argv[1], sys.argv[2]
    A, B = _load_decodes(run_a), _load_decodes(run_b)
    capA, capB = _load_capacity(run_a), _load_capacity(run_b)
    keys = A.keys() & B.keys()
    if not keys:
        sys.exit("no overlapping (model,mechanism,task,trial) keys — were both --decode-model runs done?")

    # per (model, mechanism): bit agreement over slots both decoders parsed; erasure rates
    agree = defaultdict(lambda: [0, 0])   # [matching_bits, comparable_bits]
    erasA = defaultdict(lambda: [0, 0]); erasB = defaultdict(lambda: [0, 0])
    for k in keys:
        model, mech = k[0], k[1]
        a, b = A[k], B[k]
        for i in range(min(len(a), len(b))):
            if a[i] == -1:
                erasA[(model, mech)][0] += 1
            if b[i] == -1:
                erasB[(model, mech)][0] += 1
            erasA[(model, mech)][1] += 1; erasB[(model, mech)][1] += 1
            if a[i] in (0, 1) and b[i] in (0, 1):
                agree[(model, mech)][1] += 1
                if a[i] == b[i]:
                    agree[(model, mech)][0] += 1

    models = sorted({k[0] for k in keys})
    print(f"Decoder agreement: A={run_a}  vs  B={run_b}   ({len(keys)} shared trials)\n")
    hdr = f"{'model':22s} {'mech':18s} {'agree':>7s} {'A_eras':>7s} {'B_eras':>7s} {'C_A':>5s} {'C_B':>5s}"
    print(hdr); print("-" * len(hdr))
    tot_m = tot_c = 0
    for model in models:
        for mech in sorted(MODEL_MECHS):
            m, c = agree[(model, mech)]
            if c == 0 and (model, mech) not in capA:
                continue
            ag = f"{m / c:.2f}" if c else "  -- "
            ea = erasA[(model, mech)]; eb = erasB[(model, mech)]
            ear = f"{ea[0] / ea[1]:.2f}" if ea[1] else "  -- "
            ebr = f"{eb[0] / eb[1]:.2f}" if eb[1] else "  -- "
            ca = capA.get((model, mech)); cb = capB.get((model, mech))
            print(f"{model:22s} {mech:18s} {ag:>7s} {ear:>7s} {ebr:>7s} "
                  f"{('%.1f'%ca) if ca is not None else '  --':>5s} {('%.1f'%cb) if cb is not None else '  --':>5s}")
            tot_m += m; tot_c += c
    print("-" * len(hdr))
    print(f"overall bit-agreement (where both decoders parsed): {tot_m}/{tot_c} = "
          f"{(tot_m / tot_c if tot_c else 0):.3f}")
    if tot_c:
        from scipy.stats import binomtest
        bt = binomtest(tot_m, tot_c, 0.5, alternative="two-sided")
        try:
            lo, hi = bt.proportion_ci(0.95)
            ci, verdict = f"95% CI [{lo:.3f}, {hi:.3f}]; ", \
                ("consistent with chance" if lo <= 0.5 <= hi else "ABOVE chance")
        except Exception:
            ci, verdict = "", ""
        print(f"  n={tot_c} compared bits; {ci}p(vs 0.5)={bt.pvalue:.3g}  {verdict}")
    # flag Qwen-family senders whose capacity drops a lot under the non-Qwen decoder
    print("\nQwen-family senders — C_ctrl drop under the non-Qwen decoder (possible self-decoding):")
    for model in models:
        if not any(t in model.lower() for t in ("qwen", "distill")):
            continue
        for mech in sorted(MODEL_MECHS):
            ca, cb = capA.get((model, mech)), capB.get((model, mech))
            if ca is not None and cb is not None and ca - cb >= 1.0:
                print(f"  {model:22s} {mech:18s} {ca:.1f} (Qwen) -> {cb:.1f} (non-Qwen)  Δ={ca - cb:.1f}")


if __name__ == "__main__":
    main()
