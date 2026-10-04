from __future__ import annotations

from dataclasses import dataclass

from trustbench.agents.backends import ModelBackend
from trustbench.data.schemas import AgentEvent, Scenario


@dataclass(frozen=True)
class PolicyContext:
    seed: int
    enable_second_verification: bool = True


class TrustPolicy:
    name: str

    def run(self, scenario: Scenario, backend: ModelBackend, context: PolicyContext) -> AgentEvent:
        raise NotImplementedError
