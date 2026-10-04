from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ToolClass(str, Enum):
    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    STRUCTURED_API = "structured_api"
    FILE_MEMORY = "file_memory"


class ReliabilityLabel(str, Enum):
    RELIABLE = "reliable"
    UNRELIABLE = "unreliable"
    CONTEXT_DEPENDENT = "context_dependent"


class TrustAction(str, Enum):
    ACCEPT = "accept"
    VERIFY = "verify"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


class CorruptionMode(str, Enum):
    NONE = "none"
    STALE = "stale_timestamp"
    PARTIAL_OMISSION = "partial_field_omission"
    OFF_BY_ONE = "off_by_one_or_arithmetic_bug"
    SCHEMA_DRIFT = "schema_drift"
    WRONG_ENTITY = "wrong_user_or_wrong_entity"
    PLAUSIBLE_FALSE = "plausible_hallucinated_content"
    EDGE_CASE = "hidden_edge_case_failure"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    tool_class: ToolClass
    user_task: str
    tool_invocation: dict[str, Any]
    tool_output: dict[str, Any]
    clean_output: dict[str, Any]
    reliability_label: ReliabilityLabel
    corruption_mode: CorruptionMode
    gold_action: TrustAction
    expected_answer: str
    difficulty: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ["tool_class", "reliability_label", "corruption_mode", "gold_action"]:
            data[key] = data[key].value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=data["scenario_id"],
            tool_class=ToolClass(data["tool_class"]),
            user_task=data["user_task"],
            tool_invocation=data["tool_invocation"],
            tool_output=data["tool_output"],
            clean_output=data["clean_output"],
            reliability_label=ReliabilityLabel(data["reliability_label"]),
            corruption_mode=CorruptionMode(data["corruption_mode"]),
            gold_action=TrustAction(data["gold_action"]),
            expected_answer=data["expected_answer"],
            difficulty=data["difficulty"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Annotation:
    scenario_id: str
    annotator_id: str
    trust_action: TrustAction
    reliability_label: ReliabilityLabel
    rationale: str
    adjudicated: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trust_action"] = self.trust_action.value
        data["reliability_label"] = self.reliability_label.value
        return data


@dataclass(frozen=True)
class PolicySpec:
    name: str
    description: str
    emits_trust_score: bool
    can_verify: bool
    abstention_enabled: bool


@dataclass(frozen=True)
class AgentEvent:
    scenario_id: str
    model_name: str
    model_tier: str
    policy_name: str
    tool_class: ToolClass
    reliability_label: ReliabilityLabel
    corruption_mode: CorruptionMode
    gold_action: TrustAction
    agent_action: TrustAction
    trust_score: float
    task_success: bool
    second_verification_call: bool
    accepted_initial_output: bool
    cost_tokens: int
    latency_ms: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ["tool_class", "reliability_label", "corruption_mode", "gold_action", "agent_action"]:
            data[key] = data[key].value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvent":
        return cls(
            scenario_id=data["scenario_id"],
            model_name=data["model_name"],
            model_tier=data["model_tier"],
            policy_name=data["policy_name"],
            tool_class=ToolClass(data["tool_class"]),
            reliability_label=ReliabilityLabel(data["reliability_label"]),
            corruption_mode=CorruptionMode(data["corruption_mode"]),
            gold_action=TrustAction(data["gold_action"]),
            agent_action=TrustAction(data["agent_action"]),
            trust_score=float(data["trust_score"]),
            task_success=bool(data["task_success"]),
            second_verification_call=bool(data["second_verification_call"]),
            accepted_initial_output=bool(data["accepted_initial_output"]),
            cost_tokens=int(data["cost_tokens"]),
            latency_ms=float(data["latency_ms"]),
            reason=data["reason"],
        )
