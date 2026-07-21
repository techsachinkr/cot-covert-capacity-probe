"""CCAP command-line interface.

Subcommands::

    ccap demo  [--out runs/demo]        # end-to-end on the mock backend (no GPU)
    ccap run   --config cfg.yaml        # run the pre-registered grid
    ccap aggregate --run runs/demo      # raw trials -> capacity estimates + tables
    ccap figures   --run runs/demo      # aggregated CSVs -> figures + LaTeX tables
    ccap all   --config cfg.yaml        # run + aggregate + figures
    ccap info                           # list mechanisms / tasks / decoders
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis.aggregate import AggConfig, aggregate_all, load_trials
from .analysis.figures import make_all_figures
from .analysis.tables import make_all_tables
from .config import EstimationConfig, RunConfig, load_config
from .decoders import ModelDecoder
from .mechanisms import list_mechanisms
from .models import build_backend
from .runner import ExperimentRunner
from .tasks import list_tasks


def _grid_meta(run_dir: Path) -> dict:
    gp = run_dir / "grid.json"
    return json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}


def _models_mechs(run_dir: Path, agg) -> tuple[list, list]:
    meta = _grid_meta(run_dir)
    cells = agg["cells"]
    models = meta.get("models") or sorted(cells["model"].unique().tolist())
    mechs = meta.get("mechanisms") or sorted(cells["mechanism"].unique().tolist())
    return models, mechs


def _run_grid(cfg: RunConfig) -> Path:
    missing = [m for m in cfg.grid.models if m not in cfg.backends]
    if missing:
        raise KeyError(f"grid references models with no backend defined: {missing}")

    def _spec(name: str) -> dict:
        # API backends get the run-level concurrency unless they set their own.
        spec = {**cfg.backends[name], "name": name}
        if spec.get("type") == "openai":
            spec.setdefault("max_workers", cfg.max_workers)
        return spec

    # Senders are built lazily, one model at a time (the runner frees the previous
    # before loading the next) so a multi-model grid does not hold every model at once.
    def factory(name: str):
        return build_backend(_spec(name))

    # The model decoder (M2/M3/M4 codebook holder) is pinned — built once, kept resident.
    pinned: dict = {}
    model_decoder = None
    if cfg.model_decoder_backend:
        if cfg.model_decoder_backend not in cfg.backends:
            raise KeyError(f"model_decoder_backend {cfg.model_decoder_backend!r} has no backend spec")
        be = build_backend(_spec(cfg.model_decoder_backend))
        pinned[cfg.model_decoder_backend] = be
        model_decoder = ModelDecoder(be, name="model")

    runner = ExperimentRunner(backends=pinned, backend_factory=factory, model_decoder=model_decoder,
                              sender_max_tokens=cfg.sender_max_tokens,
                              sender_temperature=cfg.sender_temperature,
                              sender_top_p=cfg.sender_top_p, run_sentinel=cfg.grid.run_sentinel)
    out = Path(cfg.output_dir)
    print(f"[ccap] grid: {cfg.grid.n_cells()} cells, ~{cfg.grid.n_generations()} generations")
    runner.run(cfg.grid, output_dir=str(out))
    return out


def _aggregate(run_dir: Path, est: EstimationConfig, contrasts: dict | None = None) -> dict:
    df = load_trials(run_dir / "trials.csv")
    agg = aggregate_all(df, AggConfig(estimation=est, contrasts=contrasts),
                        output_dir=str(run_dir / "analysis"))
    print(f"[ccap] aggregated -> {run_dir / 'analysis'} ({len(agg['cells'])} cells)")
    return agg


_ANALYSIS_TABLES = ["cells", "capacity_map", "reliability", "benign", "sentinel",
                    "compliance", "exclusions", "hypotheses"]


def _figures(run_dir: Path, agg: dict) -> None:
    models, mechs = _models_mechs(run_dir, agg)
    fig_dir = run_dir / "figures"
    figs = make_all_figures(agg, mechs, models, out_dir=fig_dir)
    tabs = make_all_tables(agg, mechs, models, out_dir=fig_dir)
    print(f"[ccap] wrote {len(figs)} figures + {len(tabs)} tables -> {fig_dir}")


# --------------------------------------------------------------------------- #
def cmd_demo(args: argparse.Namespace) -> None:
    cfg = RunConfig.demo()
    if args.out:
        cfg.output_dir = args.out
    run_dir = _run_grid(cfg)
    agg = _aggregate(run_dir, cfg.estimation, cfg.contrasts)
    _figures(run_dir, agg)
    _print_headline(agg)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.models:                       # shard the grid (resumability / one model at a time)
        missing = [m for m in args.models if m not in cfg.backends]
        if missing:
            raise KeyError(f"--models not in config backends: {missing}")
        cfg.grid.models = list(args.models)
    if args.out:
        cfg.output_dir = args.out
    _run_grid(cfg)


def cmd_aggregate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config) if args.config else None
    est = cfg.estimation if cfg else EstimationConfig()
    contrasts = cfg.contrasts if cfg else None
    _print_headline(_aggregate(Path(args.run), est, contrasts))


def cmd_figures(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    analysis = run_dir / "analysis"
    if (analysis / "cells.csv").exists():
        import pandas as pd
        agg = {name: pd.read_csv(analysis / f"{name}.csv")
               for name in _ANALYSIS_TABLES if (analysis / f"{name}.csv").exists()}
    else:
        cfg = load_config(args.config) if args.config else None
        agg = _aggregate(run_dir, cfg.estimation if cfg else EstimationConfig(),
                         cfg.contrasts if cfg else None)
    _figures(run_dir, agg)


def cmd_all(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_dir = _run_grid(cfg)
    agg = _aggregate(run_dir, cfg.estimation, cfg.contrasts)
    _figures(run_dir, agg)
    _print_headline(agg)


def cmd_info(args: argparse.Namespace) -> None:
    print("Mechanisms:", ", ".join(list_mechanisms()), "(+ benign control M0)")
    print("Tasks:     ", ", ".join(list_tasks()))
    print("Decoders:  ", "programmatic (M1/M2/M5/M6), model codebook (M3/M4), sentinel (no-codebook)")
    print("Backends:  ", "mock, hf, vllm, openai")


def _print_headline(agg: dict) -> None:
    cm = agg.get("capacity_map")
    if cm is not None and not cm.empty:
        print("\n=== Headline: C_ctrl (reliable bits/response, exact-match knee) ===")
        print(cm.pivot_table(index="model", columns="mechanism", values="c_ctrl").round(1).to_string())
    ben = agg.get("benign")
    if ben is not None and not ben.empty:
        print(f"\nBenign separation (H4): max benign C_ctrl={ben['c_ctrl'].max():.2f} bits, "
              f"mean trace entropy={ben['trace_entropy_bits'].mean():,.0f} bits")
    hyp = agg.get("hypotheses")
    if hyp is not None and not hyp.empty:
        print("\n=== Confirmatory hypotheses (Holm-Bonferroni) ===")
        cols = [c for c in ["hypothesis", "contrast", "effect", "holm_p", "supported"] if c in hyp.columns]
        print(hyp[cols].round(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccap", description="CCAP — Controllable-Capacity Auditing Probe")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end mock run (no GPU)")
    d.add_argument("--out", default="runs/demo")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run the grid from a YAML config")
    r.add_argument("--config", required=True)
    r.add_argument("--models", nargs="+", default=None,
                   help="run only these model-conditions (shard the grid for resumability)")
    r.add_argument("--out", default=None, help="override output_dir")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("aggregate", help="raw trials -> capacity estimates")
    a.add_argument("--run", required=True)
    a.add_argument("--config", default=None)
    a.set_defaults(func=cmd_aggregate)

    f = sub.add_parser("figures", help="aggregated CSVs -> figures + tables")
    f.add_argument("--run", required=True)
    f.add_argument("--config", default=None)
    f.set_defaults(func=cmd_figures)

    al = sub.add_parser("all", help="run + aggregate + figures")
    al.add_argument("--config", required=True)
    al.set_defaults(func=cmd_all)

    i = sub.add_parser("info", help="list mechanisms / tasks / decoders")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> None:
    import sys
    for stream in (sys.stdout, sys.stderr):   # robust unicode on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
