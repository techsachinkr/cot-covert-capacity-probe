#!/usr/bin/env python
"""End-to-end CCAP demo on the mock backend (no GPU, ~1 min).

Runs the pre-registered pipeline (truncation design, exact-match Ĉ_ctrl knee,
H1-H5 under Holm-Bonferroni) and writes trials, analysis, figures, and tables.
Equivalent to ``ccap demo`` but spelled out as a usage example.

    python scripts/run_demo.py
"""

from __future__ import annotations

from ccap.analysis.aggregate import AggConfig, aggregate_all
from ccap.analysis.figures import make_all_figures
from ccap.analysis.tables import make_all_tables
from ccap.config import RunConfig
from ccap.decoders import ModelDecoder
from ccap.models import build_backend
from ccap.runner import ExperimentRunner


def main() -> None:
    cfg = RunConfig.demo()
    cfg.output_dir = "runs/demo"

    # Senders built lazily, one model at a time (frees GPU/memory between models).
    def factory(name):
        return build_backend({**cfg.backends[name], "name": name})

    pinned, model_decoder = {}, None
    if cfg.model_decoder_backend:
        be = build_backend({**cfg.backends[cfg.model_decoder_backend], "name": cfg.model_decoder_backend})
        pinned[cfg.model_decoder_backend] = be
        model_decoder = ModelDecoder(be, name="model")

    runner = ExperimentRunner(backends=pinned, backend_factory=factory, model_decoder=model_decoder,
                              sender_temperature=cfg.sender_temperature, sender_top_p=cfg.sender_top_p)
    print(f"Grid: {cfg.grid.n_cells()} cells, ~{cfg.grid.n_generations()} generations")
    df = runner.run(cfg.grid, output_dir=cfg.output_dir)

    agg = aggregate_all(df, AggConfig(estimation=cfg.estimation, contrasts=cfg.contrasts),
                        output_dir=f"{cfg.output_dir}/analysis")
    make_all_figures(agg, cfg.grid.mechanisms, cfg.grid.models, out_dir=f"{cfg.output_dir}/figures")
    make_all_tables(agg, cfg.grid.mechanisms, cfg.grid.models, out_dir=f"{cfg.output_dir}/figures")

    print("\nC_ctrl (reliable bits/response):")
    print(agg["capacity_map"].pivot_table(index="model", columns="mechanism",
                                          values="c_ctrl").round(0).to_string())
    print("\nConfirmatory hypotheses:")
    print(agg["hypotheses"][["hypothesis", "contrast", "effect", "holm_p", "supported"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
