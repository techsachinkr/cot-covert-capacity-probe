#!/usr/bin/env python
"""Regenerate the graded / decoder-relative table (tab:graded) and figure (fig5_graded).

Combines a primary run (strong decoder, e.g. Qwen) with a non-family-decoder re-decode
(e.g. Llama) to produce the per-mechanism exact-match Ĉ vs MI vs coded vs per-slot
accuracy artifacts. Offline, no API.

    python scripts/make_graded.py --primary runs/ccap-main-fixed --llama runs/ccap-main-fixed-llama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import pandas as pd

from ccap.analysis.figures import fig_graded_recovery
from ccap.analysis.tables import perslot_accuracy, table_graded_recovery
from ccap.mechanisms import all_mechanisms


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate tab:graded + fig5_graded.")
    ap.add_argument("--primary", required=True,
                    help="primary run dir (needs trials.csv + analysis/cells.csv)")
    ap.add_argument("--llama", required=True,
                    help="non-family-decoder re-decode run dir (needs trials.csv)")
    ap.add_argument("--out", default=None, help="output dir (default: <primary>/figures)")
    args = ap.parse_args()

    primary, llama = Path(args.primary), Path(args.llama)
    out = Path(args.out) if args.out else primary / "figures"
    cells = pd.read_csv(primary / "analysis" / "cells.csv")
    colmap = {m.name: ("decoded_model" if m.decoder == "model" else "decoded_programmatic")
              for m in all_mechanisms()}
    accp = perslot_accuracy(pd.read_csv(primary / "trials.csv", dtype=str, low_memory=False), colmap)
    accl = perslot_accuracy(pd.read_csv(llama / "trials.csv", dtype=str, low_memory=False),
                            {m.name: "decoded_model" for m in all_mechanisms() if m.decoder == "model"})
    tex = table_graded_recovery(cells, accp, accl, out_path=out / "table_graded.tex")
    figs = fig_graded_recovery(cells, accp, accl, out_path=out / "fig5_graded")
    print(f"[make_graded] wrote {out / 'table_graded.tex'}")
    print(f"[make_graded] wrote {figs}")
    print(tex)


if __name__ == "__main__":
    main()
