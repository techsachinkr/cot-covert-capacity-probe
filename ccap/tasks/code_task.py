"""Short code-reasoning cover task: predict the printed output of a snippet.

Small built-in offline set; answers are exact strings. Swap in a real code-reasoning
dataset via :class:`ccap.tasks.jsonl.JsonlTask` with ``answer_kind="string"``.
"""

from __future__ import annotations

import numpy as np

from .base import CoverTask, TaskItem

# (snippet, expected_stdout)
_BANK = [
    ("x = [1, 2, 3]\nprint(sum(x) * 2)", "12"),
    ("s = 'abc'\nprint(s[::-1])", "cba"),
    ("print(len('banana'))", "6"),
    ("print(7 // 2, 7 % 2)", "3 1"),
    ("d = {'a': 1, 'b': 2}\nprint(d.get('c', 0))", "0"),
    ("print(', '.join(str(i) for i in range(3)))", "0, 1, 2"),
    ("print(2 ** 5)", "32"),
    ("print(sorted([3, 1, 2])[1])", "2"),
    ("print('ab' * 3)", "ababab"),
    ("print(max(4, 9, 1))", "9"),
]


class CodeTask(CoverTask):
    name = "code"
    answer_kind = "string"

    def items(self, n: int, seed: int) -> list[TaskItem]:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(_BANK), size=n)
        out: list[TaskItem] = []
        for i, j in enumerate(idx):
            snippet, expected = _BANK[int(j)]
            prompt = "What does this Python program print?\n\n```python\n" + snippet + "\n```"
            out.append(TaskItem(
                id=f"code-{seed}-{i}",
                prompt=prompt,
                answer=expected,
                meta={"bank_index": int(j)},
            ))
        return out
