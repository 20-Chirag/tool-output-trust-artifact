from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from trustbench.data.schemas import AgentEvent, Scenario

T = TypeVar("T")


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_scenarios(path: str | Path, scenarios: Iterable[Scenario]) -> None:
    write_jsonl(path, (scenario.to_dict() for scenario in scenarios))


def read_scenarios(path: str | Path) -> list[Scenario]:
    return [Scenario.from_dict(row) for row in read_jsonl(path)]


def write_events(path: str | Path, events: Iterable[AgentEvent]) -> None:
    write_jsonl(path, (event.to_dict() for event in events))


def read_events(path: str | Path) -> list[AgentEvent]:
    return [AgentEvent.from_dict(row) for row in read_jsonl(path)]
