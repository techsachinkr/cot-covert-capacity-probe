"""Generic JSONL cover-task loader for real datasets (GSM8K, GPQA, MMLU, ...).

Each line is a JSON object with at least ``prompt`` and ``answer``; optional
``choices`` and ``id``. Register concrete instances in your config, e.g.::

    JsonlTask(name="gsm8k", path="data/gsm8k.jsonl", answer_kind="number")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import CoverTask, TaskItem


class JsonlTask(CoverTask):
    def __init__(self, name: str, path: str, answer_kind: str = "number"):
        self.name = name
        self.answer_kind = answer_kind
        self._path = Path(path)
        self._cache: list[TaskItem] | None = None

    def _load(self) -> list[TaskItem]:
        if self._cache is None:
            items: list[TaskItem] = []
            with self._path.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    items.append(TaskItem(
                        id=str(obj.get("id", f"{self.name}-{i}")),
                        prompt=obj["prompt"],
                        answer=str(obj["answer"]),
                        choices=tuple(obj["choices"]) if obj.get("choices") else None,
                        meta={k: v for k, v in obj.items()
                              if k not in {"id", "prompt", "answer", "choices"}},
                    ))
            self._cache = items
        return self._cache

    def items(self, n: int, seed: int) -> list[TaskItem]:
        pool = self._load()
        if not pool:
            return []
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(pool), size=n)
        return [pool[int(j)] for j in idx]
