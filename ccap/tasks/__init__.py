"""Cover-task registry (§6).

Built-in offline tasks span three reasoning types to test task-dependence of
capacity: arithmetic word problems, multiple-choice science, and code reasoning.
"""

from __future__ import annotations

from .base import CoverTask, TaskItem
from .code_task import CodeTask
from .jsonl import JsonlTask
from .math_task import MathTask
from .science_task import ScienceTask

_REGISTRY: dict[str, CoverTask] = {
    t.name: t for t in (MathTask(), ScienceTask(), CodeTask())
}


def register_task(task: CoverTask) -> None:
    _REGISTRY[task.name] = task


def get_task(name: str) -> CoverTask:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; available: {list_tasks()}")
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = [
    "CoverTask", "TaskItem", "MathTask", "ScienceTask", "CodeTask", "JsonlTask",
    "register_task", "get_task", "list_tasks",
]
