"""M3 — Feature-gated trust.

Post-hoc rule. Computes a small set of tool-class-specific features over
the *visible* tool output (no leak fields), turns them into a "concern
score" in [0,1], and uses the concern score to downgrade the baseline
trust score and possibly override the action.

The features are intentionally lightweight static checks a real defender
would write — field completeness, recency, schema-version match, cross-
field consistency. They are not a synthetic-bench fingerprint.

This costs no API calls; it is a pure function of (baseline event,
scenario tool output).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from trustbench.data.schemas import TrustAction


KNOWN_SCHEMA_VERSIONS = {"2.1"}
STALE_THRESHOLD_DAYS = 90


@dataclass(frozen=True)
class FeatureGatedConfig:
    # Concern score weight applied multiplicatively to baseline trust.
    concern_weight: float = 0.7
    # If new_trust below this, override action to verify.
    verify_below: float = 0.55
    # If new_trust below this, override action to abstain.
    abstain_below: float = 0.25


def _has_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"not shown", "n/a"}
    return False


def _empty_field_fraction(output: dict) -> float:
    if not output:
        return 1.0
    n = len(output)
    empty = sum(1 for v in output.values() if _has_empty_value(v))
    return empty / n


def _recency_concern(output: dict, today: date | None = None) -> float:
    today = today or date(2026, 4, 28)
    keys = ("published_at", "last_updated", "updated_at", "checked_at")
    for key in keys:
        raw = output.get(key)
        if not isinstance(raw, str):
            continue
        try:
            stamp = date.fromisoformat(raw[:10])
        except ValueError:
            continue
        age = (today - stamp).days
        if age > STALE_THRESHOLD_DAYS:
            # Smooth ramp: 90 days -> 0, 1 year -> ~1.
            return min(1.0, (age - STALE_THRESHOLD_DAYS) / 275)
    return 0.0


def _schema_concern(tool_class: str, output: dict) -> float:
    if tool_class != "structured_api":
        return 0.0
    score = 0.0
    if "schema_version" in output and output["schema_version"] not in KNOWN_SCHEMA_VERSIONS:
        score += 0.5
    expected = {"user_id", "balance_cents", "currency", "account_status"}
    missing = expected - set(output.keys())
    if missing:
        score += min(1.0, 0.25 * len(missing))
    return min(1.0, score)


def _entity_consistency_concern(tool_class: str, output: dict) -> float:
    if tool_class == "file_memory":
        req = str(output.get("requested_entity", "")).strip().lower()
        ret = str(output.get("retrieved_entity", "")).strip().lower()
        if not req or not ret:
            return 0.5
        if req == ret:
            return 0.0
        # If retrieved looks like a near-variant (suffix added) treat as suspicious.
        if req in ret or ret in req:
            return 0.6
        return 1.0
    if tool_class == "structured_api":
        req = str(output.get("requested_user_id", ""))
        got = str(output.get("user_id", "") or output.get("user", ""))
        if req and got and req != got:
            return 1.0
    return 0.0


def _missing_provenance_concern(tool_class: str, output: dict) -> float:
    if tool_class == "web_search":
        if not output.get("url"):
            return 0.6
    return 0.0


def concern_score(tool_class: str, tool_output: dict) -> dict:
    """Return per-feature components and a single combined concern score in [0,1]."""
    if not isinstance(tool_output, dict):
        return {"combined": 0.0, "features": {}}
    features = {
        "empty_fraction": _empty_field_fraction(tool_output),
        "recency": _recency_concern(tool_output),
        "schema": _schema_concern(tool_class, tool_output),
        "entity_consistency": _entity_consistency_concern(tool_class, tool_output),
        "provenance": _missing_provenance_concern(tool_class, tool_output),
    }
    # Combine via noisy-OR so any one strong signal dominates.
    keep = 1.0
    for v in features.values():
        keep *= 1.0 - max(0.0, min(1.0, float(v)))
    combined = 1.0 - keep
    return {"combined": combined, "features": features}


def apply(event: dict, scenario_tool_output: dict, cfg: FeatureGatedConfig = FeatureGatedConfig()) -> dict:
    """Apply M3 to a baseline event using its scenario's visible tool output."""
    out = dict(event)
    cs = concern_score(event["tool_class"], scenario_tool_output)
    base_trust = float(event["trust_score"])
    new_trust = max(0.0, base_trust * (1.0 - cfg.concern_weight * cs["combined"]))
    if new_trust < cfg.abstain_below:
        new_action = TrustAction.ABSTAIN.value
    elif new_trust < cfg.verify_below:
        new_action = TrustAction.VERIFY.value
    else:
        new_action = event["agent_action"]
    out["trust_score"] = new_trust
    out["agent_action"] = new_action
    out["m3_concern_score"] = cs["combined"]
    out["m3_features"] = cs["features"]
    out["policy_name"] = "m3_feature_gated"
    return out


def apply_many(events: Iterable[dict], scenario_outputs: dict[str, dict], cfg: FeatureGatedConfig = FeatureGatedConfig()) -> list[dict]:
    """events: iterable of run rows. scenario_outputs: scenario_id -> tool_output dict (sanitized)."""
    out: list[dict] = []
    for e in events:
        scn = scenario_outputs.get(e["scenario_id"])
        if scn is None:
            continue
        out.append(apply(e, scn, cfg))
    return out
