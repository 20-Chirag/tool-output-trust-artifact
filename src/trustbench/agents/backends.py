from __future__ import annotations

import random
from dataclasses import dataclass

from trustbench.data.schemas import ReliabilityLabel, Scenario, ToolClass, TrustAction


@dataclass(frozen=True)
class BackendDecision:
    action: TrustAction
    trust_score: float
    reason: str
    tokens: int
    latency_ms: float


class ModelBackend:
    name: str
    tier: str

    def decide(self, scenario: Scenario, policy_name: str, seed: int) -> BackendDecision:
        raise NotImplementedError


class SimulatedModelBackend(ModelBackend):
    """Deterministic model-tier simulator for cheap pilot validation.

    It intentionally does not use gold labels directly. It samples from
    tier- and policy-specific behavioral tendencies, with subtle corruption
    modes harder to catch than schema or entity mismatches.
    """

    def __init__(self, name: str, tier: str):
        self.name = name
        self.tier = tier

    def decide(self, scenario: Scenario, policy_name: str, seed: int) -> BackendDecision:
        rng = random.Random(f"{seed}:{self.name}:{policy_name}:{scenario.scenario_id}")
        detectable = _detectability(scenario)
        tier_skill = {"cheap": 0.20, "mid": 0.34, "strong": 0.48}.get(self.tier, 0.30)
        skepticism = 0.0
        if policy_name == "skeptical":
            skepticism = 0.16
        if policy_name == "baseline":
            skepticism = -0.06

        if scenario.reliability_label == ReliabilityLabel.RELIABLE:
            false_alarm = max(0.03, 0.10 + skepticism + rng.uniform(-0.04, 0.04))
            action = TrustAction.VERIFY if rng.random() < false_alarm else TrustAction.ACCEPT
            trust = 0.78 - false_alarm / 2 + rng.uniform(-0.08, 0.08)
            success = action == TrustAction.ACCEPT or rng.random() < 0.98
            reason = "accepted plausible clean output" if action == TrustAction.ACCEPT else "unnecessary verification"
        else:
            catch_prob = min(0.92, tier_skill + detectable + skepticism + rng.uniform(-0.08, 0.08))
            if rng.random() < catch_prob:
                action = TrustAction.VERIFY if rng.random() < 0.72 else TrustAction.ABSTAIN
                trust = 0.32 + rng.uniform(-0.13, 0.12)
                success = rng.random() < (0.80 if action == TrustAction.VERIFY else 0.55)
                reason = "noticed inconsistency and challenged tool output"
            else:
                action = TrustAction.ACCEPT
                trust = 0.67 + rng.uniform(-0.12, 0.12)
                success = False
                reason = "silently accepted plausible unreliable output"

        tokens = _token_estimate(scenario) + {"baseline": 120, "skeptical": 150}.get(policy_name, 140)
        latency = tokens * {"cheap": 0.9, "mid": 1.5, "strong": 2.5}.get(self.tier, 1.5)
        return BackendDecision(
            action=action,
            trust_score=max(0.0, min(1.0, trust)),
            reason=f"{self.tier} simulator: {reason}",
            tokens=tokens,
            latency_ms=latency,
        )


def make_backends(config_models: list[dict]) -> list[ModelBackend]:
    return [SimulatedModelBackend(model["name"], model.get("tier", "mid")) for model in config_models]


def _detectability(scenario: Scenario) -> float:
    by_corruption = {
        "stale_timestamp": 0.12,
        "partial_field_omission": 0.18,
        "off_by_one_or_arithmetic_bug": 0.08,
        "schema_drift": 0.24,
        "wrong_user_or_wrong_entity": 0.23,
        "plausible_hallucinated_content": 0.04,
        "hidden_edge_case_failure": 0.06,
    }
    by_tool = {
        ToolClass.WEB_SEARCH: -0.02,
        ToolClass.CALCULATOR: 0.02,
        ToolClass.STRUCTURED_API: 0.05,
        ToolClass.FILE_MEMORY: 0.00,
    }
    return by_corruption.get(scenario.corruption_mode.value, 0.0) + by_tool[scenario.tool_class]


def _token_estimate(scenario: Scenario) -> int:
    chars = len(scenario.user_task) + len(str(scenario.tool_output))
    return max(90, chars // 4)
