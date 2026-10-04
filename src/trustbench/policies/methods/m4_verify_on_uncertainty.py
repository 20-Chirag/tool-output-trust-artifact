"""M4 — Verify-on-uncertainty.

Post-hoc rule: take a baseline event's trust_score; if it sits in an
uncertainty band (default [0.3, 0.7]), override the action to `verify`.
Otherwise leave the baseline action unchanged.

This costs no API calls — it is a pure function of an existing event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from trustbench.data.schemas import TrustAction


@dataclass(frozen=True)
class VerifyOnUncertaintyConfig:
    low: float = 0.3
    high: float = 0.7


def apply(event: dict, cfg: VerifyOnUncertaintyConfig = VerifyOnUncertaintyConfig()) -> dict:
    """Return a new event row with the M4 action applied.

    The input is one row from runs.jsonl with at minimum `trust_score`
    and `agent_action`. The output is a copy with `agent_action`
    possibly overridden to `verify`, and a marker `m4_overridden`
    saying whether the rule fired.
    """
    out = dict(event)
    score = float(event["trust_score"])
    overridden = cfg.low <= score <= cfg.high
    if overridden:
        out["agent_action"] = TrustAction.VERIFY.value
    out["m4_overridden"] = overridden
    out["policy_name"] = "m4_verify_on_uncertainty"
    return out


def apply_many(events: Iterable[dict], cfg: VerifyOnUncertaintyConfig = VerifyOnUncertaintyConfig()) -> list[dict]:
    return [apply(e, cfg) for e in events]
