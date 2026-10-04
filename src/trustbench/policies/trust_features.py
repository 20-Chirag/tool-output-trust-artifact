from __future__ import annotations

from datetime import date, datetime
from typing import Any

from trustbench.data.schemas import CorruptionMode, Scenario, ToolClass, TrustAction


REQUIRED_FIELDS = {
    ToolClass.WEB_SEARCH: ["title", "snippet", "url", "published_at", "retrieved_at", "entity", "answer"],
    ToolClass.CALCULATOR: ["expression", "result", "engine", "checked_at", "answer"],
    ToolClass.STRUCTURED_API: ["requested_user_id", "user_id", "balance_cents", "currency", "updated_at", "schema_version"],
    ToolClass.FILE_MEMORY: ["requested_entity", "retrieved_entity", "owner_id", "document_id", "snippet", "last_updated"],
}


def extract_features(scenario: Scenario) -> dict[str, float]:
    output = scenario.tool_output
    features = {
        "freshness": _freshness_score(scenario),
        "schema": _schema_score(scenario),
        "entity_match": _entity_score(scenario),
        "numeric_consistency": _numeric_score(scenario),
        "completeness": _completeness_score(scenario),
        "lexical_plausibility": _lexical_score(output),
    }
    features["aggregate"] = sum(features.values()) / len(features)
    return features


def heuristic_score(scenario: Scenario) -> tuple[float, list[str]]:
    features = extract_features(scenario)
    weights = {
        "freshness": 0.18,
        "schema": 0.20,
        "entity_match": 0.22,
        "numeric_consistency": 0.18,
        "completeness": 0.16,
        "lexical_plausibility": 0.06,
    }
    score = sum(features[name] * weight for name, weight in weights.items())
    reasons = [f"{name}={features[name]:.2f}" for name in weights]
    return max(0.0, min(1.0, score)), reasons


def action_from_score(score: float, enable_second_verification: bool) -> TrustAction:
    if score >= 0.94:
        return TrustAction.ACCEPT
    if score >= 0.70 and enable_second_verification:
        return TrustAction.VERIFY
    return TrustAction.ABSTAIN


def _completeness_score(scenario: Scenario) -> float:
    required = REQUIRED_FIELDS[scenario.tool_class]
    present = sum(1 for field in required if field in scenario.tool_output and scenario.tool_output[field] not in [None, ""])
    return present / len(required)


def _schema_score(scenario: Scenario) -> float:
    output = scenario.tool_output
    if scenario.tool_class == ToolClass.STRUCTURED_API:
        if "balance" in output and "balance_cents" not in output:
            return 0.35
        if output.get("schema_version") not in {"2.1", "2.2"}:
            return 0.55
    return _completeness_score(scenario)


def _entity_score(scenario: Scenario) -> float:
    output = scenario.tool_output
    if scenario.tool_class == ToolClass.STRUCTURED_API:
        return 1.0 if output.get("requested_user_id") == output.get("user_id") else 0.15
    if scenario.tool_class == ToolClass.FILE_MEMORY:
        return 1.0 if output.get("requested_entity") == output.get("retrieved_entity") else 0.20
    return 1.0


def _freshness_score(scenario: Scenario) -> float:
    output = scenario.tool_output
    today = date(2026, 4, 25)
    if scenario.tool_class == ToolClass.CALCULATOR:
        return 1.0 if "checked_at" in output else 0.65
    if scenario.tool_class == ToolClass.STRUCTURED_API:
        field = "updated_at"
    elif scenario.tool_class == ToolClass.WEB_SEARCH:
        field = "published_at"
    else:
        field = "last_updated"
    if field not in output:
        return 0.65
    try:
        parsed = datetime.fromisoformat(str(output[field]).replace("Z", "+00:00")).date()
    except ValueError:
        return 0.50
    age_days = (today - parsed).days
    if age_days <= 30:
        return 1.0
    if age_days <= 180:
        return 0.72
    if age_days <= 365:
        return 0.45
    return 0.18


def _numeric_score(scenario: Scenario) -> float:
    if scenario.tool_class != ToolClass.CALCULATOR:
        return 1.0
    output = scenario.tool_output
    try:
        expected = eval(str(output["expression"]), {"__builtins__": {}}, {})
        observed = float(output["result"])
    except Exception:
        return 0.40
    return 1.0 if abs(expected - observed) < 1e-9 else max(0.10, 1.0 - min(0.9, abs(expected - observed) / 15.0))


def _lexical_score(output: dict[str, Any]) -> float:
    text = " ".join(str(value).lower() for value in output.values())
    penalties = ["reported", "archive", "beta", "not shown", "exception"]
    penalty = sum(0.08 for marker in penalties if marker in text)
    return max(0.35, 1.0 - penalty)


def apparent_answer_matches_clean(scenario: Scenario) -> bool:
    return str(scenario.tool_output.get("answer")) == str(scenario.clean_output.get("answer"))
