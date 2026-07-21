"""Checkpoint/resume: an interrupted run must resume from the last completed cell
and produce results identical to an uninterrupted run."""

import pandas as pd

from ccap.grid import BENIGN, GridSpec
from ccap.models import MockBackend
from ccap.runner import ExperimentRunner


def _grid():
    return GridSpec(models=["A", "B"], mechanisms=[BENIGN, "acrostic", "framing"],
                    tasks=["math"], slots=16, ladder=(1, 2, 4, 8, 16), n_items=30)


def _backends():
    return {"A": MockBackend(name="A", fidelity=0.97), "B": MockBackend(name="B", fidelity=0.9)}


def test_resume_matches_uninterrupted(tmp_path):
    grid = _grid()
    # 1) full clean run -> reference
    ref_dir = tmp_path / "ref"
    ref = ExperimentRunner(_backends(), verbose=False).run(grid, output_dir=str(ref_dir))

    # 2) simulate a crash: truncate the checkpoint to the first ~2.5 cells of trials.jsonl
    crash_dir = tmp_path / "crash"
    ExperimentRunner(_backends(), verbose=False).run(grid, output_dir=str(crash_dir))
    lines = (crash_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    # keep 2 full cells (2*30) + a partial 3rd cell (15 rows) to exercise torn-cell drop
    (crash_dir / "trials.jsonl").write_text("\n".join(lines[: 2 * 30 + 15]) + "\n", encoding="utf-8")
    (crash_dir / "trials.csv").unlink()   # csv never existed at crash time

    # 3) resume -> must complete and match the reference exactly
    resumed = ExperimentRunner(_backends(), verbose=False).run(grid, output_dir=str(crash_dir))

    key = ["model", "mechanism", "task", "trial_index"]
    a = ref.sort_values(key).reset_index(drop=True)
    b = resumed.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) == grid.n_cells() * grid.n_items
    pd.testing.assert_frame_equal(a, b)


def test_rerun_is_full_skip(tmp_path):
    grid = _grid()
    out = tmp_path / "run"
    ExperimentRunner(_backends(), verbose=False).run(grid, output_dir=str(out))
    n_lines_1 = len((out / "trials.jsonl").read_text(encoding="utf-8").splitlines())
    # re-running a completed run regenerates nothing and the log doesn't grow
    ExperimentRunner(_backends(), verbose=False).run(grid, output_dir=str(out))
    n_lines_2 = len((out / "trials.jsonl").read_text(encoding="utf-8").splitlines())
    assert n_lines_1 == n_lines_2 == grid.n_cells() * grid.n_items
