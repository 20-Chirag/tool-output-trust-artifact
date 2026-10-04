from __future__ import annotations

import random
from dataclasses import dataclass

from trustbench.agents.backends import BackendDecision, ModelBackend
from trustbench.data.schemas import AgentEvent, ReliabilityLabel, Scenario, TrustAction
from trustbench.policies.base import PolicyContext, TrustPolicy
from trustbench.policies.trust_features import action_from_score, apparent_answer_matches_clean, heuristic_score


class BackendPolicy(TrustPolicy):
    def __init__(self, name: str):
        self.name = name

    def run(self, scenario: Scenario, backend: ModelBackend, context: PolicyContext) -> AgentEvent:
        decision = backend.decide(scenario, self.name, context.seed)
        return _event_from_decision(scenario, backend, self.name, decision)


class HeuristicTrustPolicy(TrustPolicy):
    name = "heuristic_trust"

    def run(self, scenario: Scenario, backend: ModelBackend, context: PolicyContext) -> AgentEvent:
        score, reasons = heuristic_score(scenario)
        action = action_from_score(score, context.enable_second_verification)
        second = action == TrustAction.VERIFY and context.enable_second_verification
        success = _success_from_action(scenario, action, second)
        decision = BackendDecision(
            action=action,
            trust_score=score,
            reason="heuristic trust layer: " + "; ".join(reasons),
            tokens=180 + len(str(scenario.tool_output)) // 5 + (110 if second else 0),
            latency_ms=240.0 + (180.0 if second else 0.0),
        )
        return _event_from_decision(scenario, backend, self.name, decision, success_override=success, second=second)


@dataclass
class LearnedTrustModel:
    threshold_accept: float = 0.68
    threshold_verify: float = 0.42
    reliable_mean: float = 0.75
    unreliable_mean: float = 0.42

    def fit(self, scenarios: list[Scenario]) -> "LearnedTrustModel":
        reliable_scores = []
        unreliable_scores = []
        for scenario in scenarios:
            score, _ = heuristic_score(scenario)
            if scenario.reliability_label == ReliabilityLabel.RELIABLE:
                reliable_scores.append(score)
            else:
                unreliable_scores.append(score)
        if reliable_scores:
            self.reliable_mean = sum(reliable_scores) / len(reliable_scores)
        if unreliable_scores:
            self.unreliable_mean = sum(unreliable_scores) / len(unreliable_scores)
        midpoint = (self.reliable_mean + self.unreliable_mean) / 2
        self.threshold_accept = min(0.96, max(0.72, midpoint + 0.04))
        self.threshold_verify = min(0.78, max(0.45, midpoint - 0.12))
        return self

    def score(self, scenario: Scenario) -> float:
        base, _ = heuristic_score(scenario)
        return max(0.0, min(1.0, base))


class LearnedTrustPolicy(TrustPolicy):
    name = "learned_trust"

    def __init__(self, model: LearnedTrustModel):
        self.model = model

    def run(self, scenario: Scenario, backend: ModelBackend, context: PolicyContext) -> AgentEvent:
        score = self.model.score(scenario)
        if score >= self.model.threshold_accept:
            action = TrustAction.ACCEPT
        elif score >= self.model.threshold_verify and context.enable_second_verification:
            action = TrustAction.VERIFY
        else:
            action = TrustAction.ABSTAIN
        second = action == TrustAction.VERIFY and context.enable_second_verification
        success = _success_from_action(scenario, action, second)
        decision = BackendDecision(
            action=action,
            trust_score=score,
            reason=(
                "learned trust classifier: "
                f"score={score:.2f}; accept_at={self.model.threshold_accept:.2f}; "
                f"verify_at={self.model.threshold_verify:.2f}"
            ),
            tokens=210 + len(str(scenario.tool_output)) // 5 + (110 if second else 0),
            latency_ms=260.0 + (180.0 if second else 0.0),
        )
        return _event_from_decision(scenario, backend, self.name, decision, success_override=success, second=second)


class OraclePolicy(TrustPolicy):
    name = "oracle"

    def run(self, scenario: Scenario, backend: ModelBackend, context: PolicyContext) -> AgentEvent:
        action = scenario.gold_action
        score = 0.98 if scenario.reliability_label == ReliabilityLabel.RELIABLE else 0.05
        second = action == TrustAction.VERIFY and context.enable_second_verification
        success = _success_from_action(scenario, action, second)
        decision = BackendDecision(
            action=action,
            trust_score=score,
            reason="oracle upper bound using gold reliability label",
            tokens=80 + (90 if second else 0),
            latency_ms=100.0 + (120.0 if second else 0.0),
        )
        return _event_from_decision(scenario, backend, self.name, decision, success_override=success, second=second)


def _event_from_decision(
    scenario: Scenario,
    backend: ModelBackend,
    policy_name: str,
    decision: BackendDecision,
    success_override: bool | None = None,
    second: bool | None = None,
) -> AgentEvent:
    second_call = bool(second) if second is not None else decision.action == TrustAction.VERIFY
    success = _success_from_action(scenario, decision.action, second_call) if success_override is None else success_override
    return AgentEvent(
        scenario_id=scenario.scenario_id,
        model_name=backend.name,
        model_tier=backend.tier,
        policy_name=policy_name,
        tool_class=scenario.tool_class,
        reliability_label=scenario.reliability_label,
        corruption_mode=scenario.corruption_mode,
        gold_action=scenario.gold_action,
        agent_action=decision.action,
        trust_score=decision.trust_score,
        task_success=success,
        second_verification_call=second_call,
        accepted_initial_output=decision.action == TrustAction.ACCEPT,
        cost_tokens=decision.tokens,
        latency_ms=decision.latency_ms,
        reason=decision.reason,
    )


def _success_from_action(scenario: Scenario, action: TrustAction, second_call: bool) -> bool:
    if scenario.reliability_label == ReliabilityLabel.RELIABLE:
        return action in {TrustAction.ACCEPT, TrustAction.VERIFY}
    if action == TrustAction.ACCEPT:
        return apparent_answer_matches_clean(scenario)
    if action == TrustAction.VERIFY and second_call:
        return True
    if action in {TrustAction.ABSTAIN, TrustAction.ESCALATE}:
        return True
    return False
