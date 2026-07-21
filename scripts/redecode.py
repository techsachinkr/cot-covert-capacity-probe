#!/usr/bin/env python
"""Offline re-decode / re-parse of a completed run — NO sender regeneration.

The expensive sender generations (every CoT) are preserved verbatim in each shard's
``trials.jsonl`` (``gen.text``). The two critical first-run bugs are both DOWNSTREAM of
generation, so they are fixable from the saved text without touching the GPU:

  * C2 (gpt-oss harmony): re-apply the fixed ``strip_reasoning`` to ``gen.text`` so only
    the ``final`` channel is parsed -> correct n_steps / slots / entropy / exclusions and
    correct PROGRAMMATIC decodes. Fully offline (0 LLM calls).
  * C1 (model decoder for M2/M3/M4): with ``--decode-model`` set, re-run the (now fixed,
    reasoning-off, 4096-token) model decoder on the saved CoTs. Hosted, ~7k cheap calls;
    without the flag the existing ``decoded_model`` is preserved and only the offline
    fixes apply.

It reuses ``ExperimentRunner._assemble`` so every derived column matches a real run.
Writes a fresh ``trials.csv`` (+ ``grid.json``) to ``--out`` and never mutates the input.

    python scripts/redecode.py --run runs/ccap-main --out runs/ccap-main-fixed \
        --config configs/preregistration.yaml [--decode-model]
    python -m ccap aggregate --run runs/ccap-main-fixed --config configs/preregistration.yaml
    python -m ccap figures   --run runs/ccap-main-fixed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from ccap.config import load_config
from ccap.decoders import ModelDecoder, NoCodebookDecoder, ProgrammaticDecoder
from ccap.grid import BENIGN
from ccap.mechanisms import all_mechanisms, get_mechanism
from ccap.models import build_backend
from ccap.payloads import derive_seed
from ccap.runner import ExperimentRunner
from ccap.tasks import get_task
from ccap.text_utils import strip_reasoning
from ccap.types import GenResult, Mode, TrialSpec


def _find_jsonl(run: Path) -> list[Path]:
    shard = sorted((run / "shards").glob("*/trials.jsonl"))
    if shard:
        return shard
    if (run / "trials.jsonl").exists():
        return [run / "trials.jsonl"]
    raise SystemExit(f"no trials.jsonl under {run} (looked in shards/*/ and run root)")


def _grid_meta(run: Path) -> dict:
    for p in [run / "grid.json", *sorted((run / "shards").glob("*/grid.json"))]:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _item_map(task_name: str, n_items: int, master_seed: int, cache: dict) -> dict:
    if task_name not in cache:
        seed = derive_seed(master_seed, "task-items", task_name)
        cache[task_name] = {it.id: it for it in get_task(task_name).items(n_items, seed)}
    return cache[task_name]


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline re-decode of a completed CCAP run.")
    ap.add_argument("--run", required=True, help="input run dir (with shards/*/trials.jsonl)")
    ap.add_argument("--out", required=True, help="output dir for the corrected trials.csv")
    ap.add_argument("--config", default=None, help="YAML (for n_items + the model-decoder backend)")
    ap.add_argument("--decode-model", action="store_true",
                    help="re-run the hosted model decoder on M2/M3/M4 (needs the API key); "
                         "otherwise the existing decoded_model is preserved")
    ap.add_argument("--decoder-backend", default=None,
                    help="which config backend to use as the model decoder (default: the "
                         "config's model_decoder_backend). Use e.g. 'decoder_llama' for the "
                         "non-Qwen decoder-robustness check.")
    ap.add_argument("--limit", type=int, default=0, help="probe: only this many cells (0 = all)")
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    cfg = load_config(args.config) if args.config else None
    meta = _grid_meta(run)
    master_seed = int(meta.get("master_seed", cfg.master_seed if cfg else 0))
    n_items = int(meta.get("n_items", cfg.grid.n_items if cfg else 60))
    slots = int(meta.get("slots", cfg.grid.slots if cfg else 16))

    model_decoder = None
    dec_backend = args.decoder_backend or (cfg.model_decoder_backend if cfg else None)
    if args.decode_model:
        if not (cfg and dec_backend):
            raise SystemExit("--decode-model needs --config and a decoder backend "
                             "(model_decoder_backend in the YAML, or --decoder-backend)")
        if dec_backend not in cfg.backends:
            raise SystemExit(f"--decoder-backend {dec_backend!r} not in config backends: "
                             f"{sorted(cfg.backends)}")
        spec = {**cfg.backends[dec_backend], "name": dec_backend}
        spec.setdefault("max_workers", cfg.max_workers)
        model_decoder = ModelDecoder(build_backend(spec), name="model",
                                     max_tokens=int(spec.get("max_tokens", 4096)))
        print(f"[redecode] model decoder backend: {dec_backend} "
              f"(model_id={spec.get('model_id')}, max_tokens={model_decoder.max_tokens})", flush=True)

    runner = ExperimentRunner(verbose=False)          # only for _assemble + _exclusion
    prog, sentinel = ProgrammaticDecoder(), NoCodebookDecoder()
    items_cache: dict = {}

    # group every preserved trial by cell so the model decoder can batch per cell
    cells: dict = defaultdict(list)
    for jp in _find_jsonl(run):
        for line in jp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            s = d["spec"]
            cells[(s["model"], s["mechanism"], s["task"])].append(d)

    keys = list(cells)
    if args.limit:
        keys = keys[: args.limit]
    rows, stat = [], dict(n=0, model_erasure=0, model_total=0, no_cot=0)
    for ci, key in enumerate(keys):
        model, mech_name, task_name = key
        recs = cells[key]
        mech = None if mech_name == BENIGN else get_mechanism(mech_name)
        task = get_task(task_name)
        imap = _item_map(task_name, n_items, master_seed, items_cache)

        # re-strip every CoT from the preserved raw generation (the C2 fix)
        cots, errs = [], []
        for d in recs:
            err = d.get("error")
            raw = (d.get("gen") or {}).get("text", "") or ""
            cot = strip_reasoning(raw) if err is None else ""
            cots.append(cot)
            errs.append(err)

        # re-run the model decoder once per cell (batched) if requested
        model_bits = None
        if mech is not None and mech.decoder == "model" and model_decoder is not None:
            ok_idx = [i for i, e in enumerate(errs) if e is None]
            batched = model_decoder.decode_batch([cots[i] for i in ok_idx], mech, slots)
            model_bits = {}
            for j, i in enumerate(ok_idx):
                model_bits[i] = batched[j]

        for i, d in enumerate(recs):
            s = d["spec"]
            spec = TrialSpec(model=model, mechanism=mech_name, task=task_name,
                             payload_len=s["payload_len"], mode=Mode(s["mode"]),
                             trial_index=s["trial_index"], seed=s["seed"],
                             task_item_id=s.get("task_item_id", ""))
            bits = tuple(d.get("intended", []))
            cot, err = cots[i], errs[i]
            g = GenResult(**d["gen"]) if d.get("gen") else None
            item = imap.get(spec.task_item_id)

            decoded: dict = {}
            if err is None:
                if mech is None:                                   # benign: every proxy (H4)
                    for m in all_mechanisms():
                        decoded[f"prog_{m.name}"] = m.decode_programmatic(cot, slots)
                elif mech.decoder == "model":
                    if model_bits is not None:
                        decoded["model"] = model_bits[i]
                    else:                                          # preserve prior decode
                        prev = (d.get("decoded") or {}).get("model")
                        decoded["model"] = tuple(prev) if prev is not None else (-1,) * slots
                    decoded["sentinel"] = sentinel.decode(cot, mech, slots)
                else:
                    decoded["programmatic"] = prog.decode(cot, mech, slots)
                    decoded["sentinel"] = sentinel.decode(cot, mech, slots)

            tr = runner._assemble(spec, bits, cot, err, decoded, task, item, mech, slots, g)
            if item is None and err is None:        # couldn't map item -> don't fabricate a verdict
                tr.task_correct = d.get("task_correct")
            rows.append(tr.to_row())

            stat["n"] += 1
            if mech is not None and mech.decoder == "model" and err is None:
                stat["model_total"] += 1
                bb = decoded.get("model", ())
                if not bb or bb[0] == -1:
                    stat["model_erasure"] += 1
            if err is None and tr.n_steps == 0:
                stat["no_cot"] += 1
        if (ci + 1) % 25 == 0 or ci + 1 == len(keys):
            print(f"[redecode] cell {ci + 1}/{len(keys)}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)
    if meta:
        (out / "grid.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    me, mt = stat["model_erasure"], stat["model_total"]
    print(f"[redecode] wrote {stat['n']} trials -> {out/'trials.csv'}")
    print(f"[redecode] no-parseable-CoT after re-strip: {stat['no_cot']}/{stat['n']} "
          f"({stat['no_cot']/max(stat['n'],1):.1%})")
    if mt:
        print(f"[redecode] model-decoder erasure: {me}/{mt} ({me/mt:.1%}) "
              f"{'(preserved from prior run; pass --decode-model to refresh)' if model_decoder is None else ''}")
    print(f"[redecode] next:  python -m ccap aggregate --run {out} "
          f"--config {args.config or '<cfg>'}  &&  python -m ccap figures --run {out}")


if __name__ == "__main__":
    main()
