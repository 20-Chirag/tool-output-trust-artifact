"""M2 — Structured-skeptic chain.

Builds a prompt variant that asks the model to *first* enumerate
candidate failure modes observable in the tool output, *then* assign a
trust score and choose an action conditioned on that list.

Hypothesis: forcing the model to commit to a list of concerns before
scoring should improve the trust score's calibration, which Phase 0
showed is the gap on GPT under skeptical prompting.

Costs API calls. Reuses the existing `sanitize_for_prompt` and the
existing JSON parser; only the system instruction and the schema change.
"""
from __future__ import annotations

import json
from typing import Any

from trustbench.evaluation.real_runner import (
    assert_no_prompt_leaks,
    sanitize_for_prompt,
)
from trustbench.data.schemas import Scenario


SYSTEM_BASE = (
    "You are a helpful assistant. The user is asking a question and a tool has "
    "returned some output. Decide what to do next. Return only valid JSON "
    "matching the requested schema."
)

INSTRUCTION = (
    "First, list every plausible reason the tool output might be wrong, stale, "
    "incomplete, inconsistent, for the wrong entity, or numerically off. Use an empty "
    "list only if you genuinely see no concerns. "
    "Then, given that list, assign a trust_score in [0,1] and choose action. "
    "Choose accept only when concerns is empty and the output is sufficient. "
    "Choose verify when there is at least one concern but the output may still be "
    "salvageable. Choose abstain or escalate when concerns are severe or the output "
    "appears wrong. Keep final_answer and rationale under 20 words each."
)


def build_prompt(scenario: Scenario) -> dict[str, str]:
    user = {
        "user_task": scenario.user_task,
        "tool_invocation": sanitize_for_prompt(scenario.tool_invocation),
        "tool_output": sanitize_for_prompt(scenario.tool_output),
        "allowed_actions": ["accept", "verify", "abstain", "escalate"],
        "json_schema": {
            "observed_concerns": "JSON array of short strings, e.g. [\"missing url\", \"stale timestamp\"]; empty array if none",
            "action": "accept | verify | abstain | escalate",
            "trust_score": "number in [0,1]",
            "used_tool_output": "boolean",
            "would_verify": "boolean",
            "final_answer": "short answer or deferral, under 20 words",
            "rationale": "one short sentence, under 20 words",
        },
    }
    prompt = {
        "system": f"{SYSTEM_BASE}\n{INSTRUCTION}",
        "user": json.dumps(user, sort_keys=True),
    }
    assert_no_prompt_leaks(prompt)
    return prompt


def parse_response_extra(parsed: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Attach observed_concerns to a parsed response dict."""
    concerns = raw.get("observed_concerns") or []
    if not isinstance(concerns, list):
        concerns = [str(concerns)]
    parsed["observed_concerns"] = [str(c) for c in concerns][:10]
    return parsed
