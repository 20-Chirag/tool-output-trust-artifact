from __future__ import annotations

from trustbench.data.schemas import Scenario
from trustbench.policies.base import TrustPolicy
from trustbench.policies.implemented import (
    BackendPolicy,
    HeuristicTrustPolicy,
    LearnedTrustModel,
    LearnedTrustPolicy,
    OraclePolicy,
)


def build_policies(names: list[str], training_scenarios: list[Scenario] | None = None) -> list[TrustPolicy]:
    learned = LearnedTrustModel().fit(training_scenarios or [])
    registry = {
        "baseline": BackendPolicy("baseline"),
        "skeptical": BackendPolicy("skeptical"),
        "heuristic_trust": HeuristicTrustPolicy(),
        "learned_trust": LearnedTrustPolicy(learned),
        "oracle": OraclePolicy(),
    }
    return [registry[name] for name in names]
