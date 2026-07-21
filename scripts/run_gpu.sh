#!/usr/bin/env bash
# Full CCAP confirmatory run on a single GPU, sharded by model-condition for
# resumability (the runner writes per shard; re-running skips completed shards).
# Then merge -> aggregate -> figures/tables.
#
#   export OPENROUTER_API_KEY=sk-or-...     # hosted model decoder
#   export DEEPSEEK_API_KEY=...             # MC12/MC13 (DeepSeek-V4-Flash)
#   bash scripts/run_gpu.sh [config] [output_dir]
set -euo pipefail

CFG="${1:-configs/preregistration.yaml}"
ROOT="${2:-runs/ccap-main}"
mkdir -p "$ROOT/shards"

MODELS=$(python -c "from ccap.config import load_config; print(' '.join(load_config('$CFG').grid.models))")
echo "[run_gpu] models: $MODELS"

for M in $MODELS; do
  if [ -f "$ROOT/shards/$M/trials.csv" ]; then
    echo "[run_gpu] skip $M (shard already complete)"
    continue
  fi
  echo "[run_gpu] === $M ==="
  ccap run --config "$CFG" --models "$M" --out "$ROOT/shards/$M"
done

echo "[run_gpu] merging shards -> aggregate -> figures"
python - "$CFG" "$ROOT" <<'PY'
import sys, glob, pandas as pd
from ccap.config import load_config
from ccap.analysis.aggregate import AggConfig, aggregate_all, load_trials
from ccap.analysis.figures import make_all_figures
from ccap.analysis.tables import make_all_tables

cfg, root = load_config(sys.argv[1]), sys.argv[2]
shards = sorted(glob.glob(f"{root}/shards/*/trials.csv"))
df = pd.concat([load_trials(f) for f in shards], ignore_index=True)   # load_trials keeps bit cols as str
df.to_csv(f"{root}/trials.csv", index=False)
print(f"[run_gpu] merged {len(shards)} shards -> {len(df)} trials")

agg = aggregate_all(df, AggConfig(estimation=cfg.estimation, contrasts=cfg.contrasts),
                    output_dir=f"{root}/analysis")
make_all_figures(agg, cfg.grid.mechanisms, cfg.grid.models, out_dir=f"{root}/figures")
make_all_tables(agg, cfg.grid.mechanisms, cfg.grid.models, out_dir=f"{root}/figures")
print(f"[run_gpu] DONE -> {root}/  (analysis/, figures/)")
PY
