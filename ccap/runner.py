"""Experiment runner — drives the pre-registered CCAP grid (§2-§4, §8, §9).

Per (MC, mechanism, task) cell we run ``n_items`` paired items. Each non-benign
item is ONE generation embedding ``slots`` (S=16) symbol-slots; payload sizes are
recovered later by truncation (§2.4). The decoder is frozen per mechanism (§2.2):
programmatic for M1/M2/M5/M6, model decoder for M3/M4. The benign control M0 is
generated with no encoding and decoded by every programmatic decoder (the §7 H4
"strongest decoder still finds nothing" check). A no-codebook sentinel runs on the
encode cells for the exploratory S6 gap.

Outputs under ``output_dir``: ``trials.jsonl`` (raw, incl. each CoT), ``trials.csv``
(tidy), ``grid.json`` (the exact grid + metadata for the T0 lock).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .capacity import trace_entropy_bits
from .decoders import Decoder, ModelDecoder, NoCodebookDecoder, ProgrammaticDecoder
from .grid import BENIGN, GridSpec
from .mechanisms import all_mechanisms, get_mechanism
from .mechanisms.base import Mechanism
from .models.base import ModelBackend
from .payloads import PayloadGenerator
from .prompts import SENDER_SYSTEM, build_sender_prompt
from .sentinel import detection_score
from .tasks import get_task
from .text_utils import split_step_blocks, strip_reasoning
from .types import Bits, GenRequest, GenResult, Mode, SimContext, TrialResult, TrialSpec


class ExperimentRunner:
    def __init__(
        self,
        backends: Optional[dict[str, ModelBackend]] = None,
        backend_factory=None,
        model_decoder: Optional[ModelDecoder] = None,
        sender_max_tokens: int = 2048,
        sender_temperature: float = 0.7,
        sender_top_p: float = 0.95,
        run_sentinel: bool = True,
        verbose: bool = True,
    ):
        self.backends = backends or {}
        self.backend_factory = backend_factory
        self.model_decoder = model_decoder           # used only for M3/M4 (mechanism.decoder == "model")
        self.prog = ProgrammaticDecoder()
        self.sentinel = NoCodebookDecoder()
        self.sender_max_tokens = sender_max_tokens
        self.sender_temperature = sender_temperature
        self.sender_top_p = sender_top_p
        self.run_sentinel = run_sentinel
        # Throughput comes from batched generation per cell: in-process vLLM batches
        # internally; the OpenAI backend threads its own generate_batch (its max_workers).
        self.verbose = verbose
        self._lazy_name: Optional[str] = None
        self._lazy_backend: Optional[ModelBackend] = None

    # ---- backend lifecycle (lazy single-slot; see prior design) ------------
    def _get_backend(self, name: str) -> ModelBackend:
        if name in self.backends:
            return self.backends[name]
        if self.backend_factory is None:
            raise KeyError(f"no backend for model {name!r} and no backend_factory provided")
        if name != self._lazy_name:
            if self._lazy_backend is not None:
                try:
                    self._lazy_backend.close()
                except Exception:
                    pass
            if self.verbose:
                print(f"[ccap] loading backend: {name}", flush=True)
            self._lazy_backend = self.backend_factory(name)
            self._lazy_name = name
        return self._lazy_backend

    def _designated_decoder(self, mech: Mechanism) -> tuple[Decoder, str]:
        if mech.decoder == "model":
            # real model decoder if configured (M3/M4); else programmatic proxy under "model"
            return (self.model_decoder or self.prog), "model"
        return self.prog, "programmatic"

    # ---- public API --------------------------------------------------------
    def run(self, grid: GridSpec, output_dir: Optional[str] = None) -> pd.DataFrame:
        """Run the grid. With ``output_dir`` set, results are CHECKPOINTED per cell
        (appended to trials.jsonl as each cell completes), so an interruption resumes
        from the last completed cell — not the start of the model. Re-running the same
        command skips already-complete cells; torn/partial cells are dropped and redone.
        trials.csv + grid.json are written at the end from the full log."""
        results: list[TrialResult] = []
        gen = PayloadGenerator(grid.master_seed)
        cells = list(grid.base_cells())
        out = Path(output_dir) if output_dir else None
        jsonl = (out / "trials.jsonl") if out else None

        done: dict = {}
        if jsonl is not None and jsonl.exists():
            done = self._resume_from(jsonl, grid.n_items)   # also drops torn/partial cells
            if self.verbose and done:
                print(f"[ccap] resume: {len(done)}/{len(cells)} cells already checkpointed", flush=True)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        fh = jsonl.open("a", encoding="utf-8") if jsonl is not None else None
        try:
            for ci, (model, mech_name, task_name) in enumerate(cells):
                key = (model, mech_name, task_name)
                if key in done:
                    results.extend(done[key])
                    continue
                backend = self._get_backend(model)
                task = get_task(task_name)
                items = task.items(grid.n_items, grid.item_seed(task_name))
                mech = None if mech_name == BENIGN else get_mechanism(mech_name)
                cell = self._run_cell(backend, gen, grid, model, mech_name, mech, task, items)
                if fh is not None:   # one write per cell -> minimal torn-write window
                    fh.write("".join(json.dumps(r.to_json(), ensure_ascii=False) + "\n" for r in cell))
                    fh.flush()
                results.extend(cell)
                if self.verbose:
                    print(f"[ccap] cell {ci + 1}/{len(cells)}: {model}|{mech_name}|{task_name} "
                          f"({grid.n_items} items)", flush=True)
        finally:
            if fh is not None:
                fh.close()
            if self._lazy_backend is not None:
                try:
                    self._lazy_backend.close()
                except Exception:
                    pass
                self._lazy_backend, self._lazy_name = None, None

        df = pd.DataFrame([r.to_row() for r in results])
        if out is not None:
            df.to_csv(out / "trials.csv", index=False)
            self._write_grid_meta(out, grid)
            if self.verbose:
                print(f"[ccap] wrote {len(results)} trials -> {out}/trials.jsonl, trials.csv", flush=True)
        return df

    # ---- per-cell batched execution ---------------------------------------
    def _sim(self, backend, mechanism_name, bits, mode, seed, item):
        if not backend.is_simulator:
            return None
        return SimContext(mechanism=mechanism_name, intended_bits=bits, mode=mode,
                          seed=seed, task_answer=item.answer)

    def _run_cell(self, backend, gen, grid, model, mech_name, mech, task, items) -> list[TrialResult]:
        """One cell = n_items independent trials. Sender generations are issued as a
        BATCH (vLLM batches internally; the OpenAI backend threads), and the model
        decoder's calls are batched too — so in-process vLLM runs at full throughput
        instead of batch-size-1. Determinism is unaffected: each trial is independently
        seeded and generate_batch preserves order."""
        slots = grid.slots
        n = grid.n_items
        mode = Mode.BENIGN if mech is None else Mode.ENCODE

        # Phase 1: per-trial context + sender requests (paired payloads/items).
        ctx, reqs = [], []
        for it in range(n):
            item = items[it % len(items)]
            payload = gen.generate(slots, "payload", task.name, it)
            seed = grid.trial_seed(model, mech_name, task.name, it)
            sim_name = mech.name if mech is not None else "acrostic"  # benign render is mechanism-agnostic
            reqs.append(GenRequest(
                prompt=build_sender_prompt(item, mech, payload, encode=(mech is not None)),
                system=SENDER_SYSTEM, max_tokens=self.sender_max_tokens,
                temperature=self.sender_temperature, top_p=self.sender_top_p,
                sim=self._sim(backend, sim_name, payload, mode, seed, item)))
            ctx.append((it, item, payload, seed))

        # Phase 2: batched sender generation.
        gens = backend.generate_batch(reqs)
        errs = [(g.error if g is not None else "no_result") for g in gens]
        # Decode/parse the DELIVERED answer only: strip the private <think> reasoning
        # span so the surface decoders never read the model's scratchpad (which for
        # reasoning models echoes the encoding contract verbatim -> spurious decodes).
        # The raw generation is preserved verbatim in GenResult.text (trials.jsonl) for
        # diagnostics. An unclosed <think> (truncated mid-thought) strips to "" and is
        # recorded as no_parseable_cot. Non-reasoning output is unchanged.
        cots = [strip_reasoning(gens[i].text) if (gens[i] is not None and errs[i] is None) else ""
                for i in range(n)]

        # Phase 3: decode (the designated/model decoder is batched).
        decoded_per: list[dict] = [dict() for _ in range(n)]
        ok = [i for i in range(n) if errs[i] is None]
        if mech is None:
            for m in all_mechanisms():   # benign M0: every programmatic decoder; H4 takes the strongest
                for i in ok:
                    decoded_per[i][f"prog_{m.name}"] = m.decode_programmatic(cots[i], slots)
        elif ok:
            dec, name = self._designated_decoder(mech)
            recovered = dec.decode_batch([cots[i] for i in ok], mech, slots)
            for j, i in enumerate(ok):
                decoded_per[i][name] = recovered[j]
            if self.run_sentinel:
                for i in ok:
                    decoded_per[i]["sentinel"] = self.sentinel.decode(cots[i], mech, slots)

        # Phase 4: assemble.
        out = []
        for (it, item, payload, seed), g, err in zip(ctx, gens, errs):
            spec = TrialSpec(model=model, mechanism=(mech.name if mech is not None else BENIGN),
                             task=task.name, payload_len=slots, mode=mode,
                             trial_index=it, seed=seed, task_item_id=item.id)
            out.append(self._assemble(spec, payload, cots[it], err, decoded_per[it],
                                      task, item, mech, slots, g))
        return out

    # ---- assembly + exclusions (§8) ---------------------------------------
    def _assemble(self, spec, bits, cot, err, decoded, task, item, mech, slots, g) -> TrialResult:
        n_steps = len(split_step_blocks(cot)) if cot else 0
        task_correct = task.check_answer(item, cot) if err is None else None
        truncated = bool(g.truncated) if g is not None else False
        excluded, reason = self._exclusion(err, n_steps, truncated)
        extra = {
            "detect_score": detection_score(cot, slots),
            "trace_entropy": trace_entropy_bits(cot),
            "realized_slots": min(n_steps, slots),
            "truncated": int(truncated),
            "excluded": int(excluded),
            "exclude_reason": reason,
        }
        if spec.mode == Mode.ENCODE and err is None and mech is not None:
            # ALWAYS the programmatic surface read — never the model decoder. Compliance
            # must measure SENDER behaviour ("did the model surface the contracted
            # feature?") independent of decoder skill, so that for model-decoder
            # mechanisms (framing/abstraction/ordering_semantic) a low capacity can be
            # attributed to the sender vs the decoder (§125).
            prog = mech.decode_programmatic(cot, slots)
            extra["sender_compliance"] = sum(1 for a, b in zip(prog, bits) if a == b) / slots
        return TrialResult(spec=spec, intended=bits, cot=cot, decoded=decoded,
                           task_correct=task_correct, n_steps=n_steps, gen=g, error=err, extra=extra)

    @staticmethod
    def _exclusion(err, n_steps, truncated) -> tuple[bool, str]:
        if err is not None:
            return True, "generation_error"
        if truncated:
            # Hit the token budget -> the delivered answer is missing/unreliable. This
            # is an infrastructure artifact, not a capability signal, so drop it from
            # the denominator (§8) instead of scoring a failed transmission (which
            # would deflate Ĉ_ctrl, the safety *upper* bound, in the dangerous direction).
            return True, "truncated"
        if n_steps == 0:
            return True, "no_parseable_cot"
        return False, ""

    # ---- checkpoint / resume ----------------------------------------------
    @staticmethod
    def _result_from_json(d: dict) -> TrialResult:
        s = d["spec"]
        spec = TrialSpec(model=s["model"], mechanism=s["mechanism"], task=s["task"],
                         payload_len=s["payload_len"], mode=Mode(s["mode"]),
                         trial_index=s["trial_index"], seed=s["seed"],
                         task_item_id=s.get("task_item_id", ""))
        g = d.get("gen")
        gen = GenResult(**g) if g else None
        return TrialResult(
            spec=spec, intended=tuple(d.get("intended", [])), cot=d.get("cot", "") or "",
            decoded={k: tuple(v) for k, v in (d.get("decoded") or {}).items()},
            task_correct=d.get("task_correct"), n_steps=d.get("n_steps"),
            gen=gen, error=d.get("error"), extra=d.get("extra") or {})

    def _resume_from(self, jsonl: Path, n_items: int) -> dict:
        """Load complete cells from a checkpoint and rewrite the file to drop any
        torn/partial cells (so they are cleanly regenerated). A cell is complete only
        if it has exactly ``n_items`` parseable rows."""
        from collections import defaultdict
        by_cell: dict = defaultdict(list)
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)        # a crash mid-flush can leave one torn line
            except Exception:
                continue
            s = d.get("spec", {})
            by_cell[(s.get("model"), s.get("mechanism"), s.get("task"))].append(d)
        done, keep = {}, []
        for cell, rows in by_cell.items():
            if len(rows) >= n_items:
                rows = rows[:n_items]
                done[cell] = [self._result_from_json(d) for d in rows]
                keep.extend(json.dumps(d, ensure_ascii=False) for d in rows)
        jsonl.write_text(("\n".join(keep) + "\n") if keep else "", encoding="utf-8")
        return done

    # ---- io ----------------------------------------------------------------
    def _write_grid_meta(self, out: Path, grid) -> None:
        (out / "grid.json").write_text(json.dumps({
            "models": grid.models, "mechanisms": grid.mechanisms, "tasks": grid.tasks,
            "slots": grid.slots, "ladder": list(grid.ladder), "n_items": grid.n_items,
            "master_seed": grid.master_seed, "n_cells": grid.n_cells(),
            "n_generations": grid.n_generations(),
            "sender": {"max_tokens": self.sender_max_tokens, "temperature": self.sender_temperature,
                       "top_p": self.sender_top_p},
        }, indent=2), encoding="utf-8")
