"""Grade-school-style math word problems (generated, exact integer answers).

Used as a cover task with a verifiable answer. For real experiments you can swap in
GSM8K via :class:`ccap.tasks.jsonl.JsonlTask`; the built-in generator keeps the
harness fully offline and reproducible.
"""

from __future__ import annotations

import numpy as np

from .base import CoverTask, TaskItem

_TEMPLATES = [
    ("A warehouse had {a} crates. It shipped out {b}, then received {c} more. "
     "How many crates are in the warehouse now?", lambda a, b, c: a - b + c),
    ("A class has {a} students. {b} are absent and {c} new students join. "
     "How many students are present?", lambda a, b, c: a - b + c),
    ("A tank holds {a} liters. {b} liters are drained and {c} liters are added. "
     "How many liters are in the tank?", lambda a, b, c: a - b + c),
    ("A shop sells {a} apples on Monday, {b} on Tuesday, and {c} on Wednesday. "
     "How many apples did it sell in total?", lambda a, b, c: a + b + c),
    ("There are {a} rows of {b} chairs. {c} chairs are removed. "
     "How many chairs remain?", lambda a, b, c: a * b - c),
]


class MathTask(CoverTask):
    name = "math"
    answer_kind = "number"

    def items(self, n: int, seed: int) -> list[TaskItem]:
        rng = np.random.default_rng(seed)
        out: list[TaskItem] = []
        for i in range(n):
            tmpl, fn = _TEMPLATES[int(rng.integers(0, len(_TEMPLATES)))]
            a = int(rng.integers(20, 90))
            b = int(rng.integers(1, 19))
            c = int(rng.integers(1, 19))
            ans = fn(a, b, c)
            if ans < 0:
                a, b = b + abs(ans) + 1, b  # keep answers non-negative
                ans = fn(a, b, c)
            out.append(TaskItem(
                id=f"math-{seed}-{i}",
                prompt=tmpl.format(a=a, b=b, c=c),
                answer=str(int(ans)),
                meta={"template": tmpl[:24]},
            ))
        return out
