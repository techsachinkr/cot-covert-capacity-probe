"""Cover-task interface.

A cover task supplies problem instances that the sender must *actually solve* while
embedding the payload. Measuring task correctness lets us report the accuracy cost
of encoding (a useful secondary diagnostic), and keeps the channel honest: the CoT
has to do real work, not just carry bits.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..text_utils import extract_answer


@dataclass(frozen=True)
class TaskItem:
    id: str
    prompt: str
    answer: str
    choices: tuple[str, ...] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _final_text(model_text: str) -> str:
    ans = extract_answer(model_text)
    return ans if ans is not None else (model_text or "").strip().splitlines()[-1] if model_text else ""


class CoverTask(ABC):
    name: str = "base"
    answer_kind: str = "number"  # "number" | "choice" | "string"

    @abstractmethod
    def items(self, n: int, seed: int) -> list[TaskItem]:
        """Return ``n`` task instances, deterministically seeded."""

    def check_answer(self, item: TaskItem, model_text: str) -> bool:
        final = _final_text(model_text)
        if self.answer_kind == "number":
            nums = _NUM_RE.findall(final) or _NUM_RE.findall(model_text or "")
            if not nums:
                return False
            try:
                return abs(float(nums[-1]) - float(item.answer)) < 1e-6
            except ValueError:
                return False
        if self.answer_kind == "choice":
            m = _LETTER_RE.findall(final.upper()) or _LETTER_RE.findall((model_text or "").upper())
            return bool(m) and m[-1] == item.answer.upper()
        return final.strip().lower() == item.answer.strip().lower()
