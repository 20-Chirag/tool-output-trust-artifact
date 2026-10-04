"""Canonical paper-facing interpretation of tool-output trust decisions."""
from __future__ import annotations

from typing import Any, Mapping


def accepted_initial_output(row: Mapping[str, Any]) -> bool:
    """Return whether the visible tool output was accepted for direct use.

    Newer run rows record this construct explicitly. Historical rows fall back
    to the four-way action because some early policies did not emit the
    ``used_tool_output`` field.
    """
    if "accepted_initial_output" in row:
        return bool(row["accepted_initial_output"])
    return str(row.get("agent_action", "")) == "accept"


def deferred_or_verified_initial_output(row: Mapping[str, Any]) -> bool:
    return not accepted_initial_output(row)
