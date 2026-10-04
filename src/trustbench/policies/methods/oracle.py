"""Oracle policy — uses the ground-truth gold_action from each scenario.

This establishes the theoretical upper bound: a policy that always knows
whether the output is reliable and acts on that knowledge perfectly. It
costs no API calls; it reads gold_action directly from the scenario.

Use this in every phase's evaluation to anchor the performance ceiling.
"""
from __future__ import annotations

from typing import Iterable

from trustbench.data.schemas import ReliabilityLabel, Scenario, TrustAction


def event_from_scenario(scenario: Scenario, model_name: str) -> dict:
    """Produce an oracle event row for one scenario."""
    gold = scenario.gold_action
    task_success = (
        gold in {TrustAction.ACCEPT, TrustAction.VERIFY}
        if scenario.reliability_label == ReliabilityLabel.RELIABLE
        else gold in {TrustAction.VERIFY, TrustAction.ABSTAIN, TrustAction.ESCALATE}
    )
    return {
        "scenario_id": scenario.scenario_id,
        "model_name": model_name,
        "model_tier": "oracle",
        "policy_name": "oracle",
        "tool_class": scenario.tool_class.value,
        "reliability_label": scenario.reliability_label.value,
        "corruption_mode": scenario.corruption_mode.value,
        "gold_action": gold.value,
        "agent_action": gold.value,
        "trust_score": 1.0 if scenario.reliability_label == ReliabilityLabel.RELIABLE else 0.0,
        "task_success": task_success,
        "second_verification_call": gold == TrustAction.VERIFY,
        "accepted_initial_output": gold == TrustAction.ACCEPT,
        "cost_tokens": 0,
        "latency_ms": 0.0,
        "reason": "oracle",
    }


def events_from_scenarios(scenarios: Iterable[Scenario], model_names: list[str]) -> list[dict]:
    """One oracle event per (scenario, model_name) — model is a label only."""
    out = []
    for s in scenarios:
        for m in model_names:
            out.append(event_from_scenario(s, m))
    return out
