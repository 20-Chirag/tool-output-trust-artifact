"""Schema-aligned checklist for construct-valid follow-up experiments.

This is a new post-hoc policy. It does not replace the historical M6 policy.
The checklist is derived only from the model-visible task and invocation, and
it explicitly forbids assumptions about undocumented API schemas.
"""
from __future__ import annotations

import json
from typing import Any

from trustbench.data.schemas import Scenario
from trustbench.evaluation.real_runner import (
    assert_no_prompt_leaks,
    sanitize_for_prompt,
)


SYSTEM_BASE = (
    "You are a helpful assistant. The user is asking a question and a tool has "
    "returned some output. Decide what to do next. Return only valid JSON "
    "matching the requested schema."
)

COMMON_RULES = (
    "Use only the visible task, invocation, and tool output. Do not assume that "
    "an undocumented field is required. A generic possibility that a tool can "
    "fail is not a concrete concern. Choose verify, abstain, or escalate only "
    "for a specific unresolved problem visible in this case. Keep final_answer "
    "and rationale under 30 words each."
)


def build_prompt(scenario: Scenario) -> dict[str, str]:
    invocation = sanitize_for_prompt(scenario.tool_invocation)
    output = sanitize_for_prompt(scenario.tool_output)
    checklist = _checklist_for(scenario, invocation)
    checklist_text = "\n".join(
        f"  {index}. {item}" for index, item in enumerate(checklist, start=1)
    )
    instruction = (
        f"{COMMON_RULES}\n\nChecklist:\n{checklist_text}\n\n"
        "After the checklist, assign trust_score and choose an action."
    )
    user = {
        "user_task": scenario.user_task,
        "tool_invocation": invocation,
        "tool_output": output,
        "allowed_actions": ["accept", "verify", "abstain", "escalate"],
        "json_schema": {
            "action": "accept | verify | abstain | escalate",
            "trust_score": "number in [0,1]",
            "used_tool_output": "boolean",
            "would_verify": "boolean",
            "final_answer": "short answer or deferral, under 30 words",
            "rationale": (
                "one short sentence naming the concrete check or problem, "
                "under 30 words"
            ),
        },
    }
    prompt = {
        "system": f"{SYSTEM_BASE}\n{instruction}",
        "user": json.dumps(user, sort_keys=True),
    }
    assert_no_prompt_leaks(prompt)
    return prompt


def _checklist_for(
    scenario: Scenario,
    invocation: dict[str, Any],
) -> list[str]:
    tool_class = (
        scenario.tool_class.value
        if hasattr(scenario.tool_class, "value")
        else str(scenario.tool_class)
    )
    arguments = invocation.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}

    if tool_class == "calculator":
        expression = _compact(arguments.get("expression", "the expression"))
        requested_units = _compact(arguments.get("units", "the requested units"))
        return [
            (
                f"Independently recompute `{expression}`. If rejecting the "
                "numeric result, state the independently computed value."
            ),
            "Do the structured result and summary answer agree with each other?",
            f"Are the result and answer expressed in `{requested_units}`?",
            (
                "Is any rounding or precision difference large enough to change "
                "the requested answer?"
            ),
        ]

    if tool_class == "structured_api":
        entity = _first_argument(
            arguments,
            ("entity", "city", "country", "isbn", "country_code"),
        )
        field = _first_argument(
            arguments,
            ("field", "indicator", "requested_field"),
        )
        fields = arguments.get("fields")
        requested = _requested_description(field, fields)
        reference = _first_argument(
            arguments,
            (
                "reference_time",
                "reference_date",
                "as_of",
                "year",
                "reference_year",
            ),
        )
        if reference in (None, "") and _task_states_reference_time(
            scenario.user_task
        ):
            reference = "the explicit time or date stated in the task"
        return [
            (
                f"Does the returned entity match the requested entity "
                f"`{_compact(entity or 'shown in the invocation')}`?"
            ),
            (
                f"Does the output provide the requested value or fields "
                f"`{requested}` without being null, omitted, or malformed?"
            ),
            (
                "If both a structured value and summary answer are present, "
                "are they mutually consistent?"
            ),
            (
                "Treat a schema issue as concrete only when visible fields "
                "contradict each other or the invocation. Do not assume an "
                "undocumented external schema."
            ),
            _time_check(reference),
        ]

    if tool_class == "web_search":
        return [
            "Does the result concern the same entity and question as the query?",
            "Does the visible snippet actually support the answer being used?",
            "Is a displayed date incompatible with an explicit task date?",
            "Is the source URL present and consistent with the displayed source?",
        ]

    if tool_class == "file_memory":
        entity = _first_argument(arguments, ("entity", "requested_entity"))
        requested_entity = entity or "the requested entity"
        return [
            (
                f"Does the retrieved record match "
                f"`{_compact(requested_entity)}`?"
            ),
            "Does the visible record contain the field requested by the task?",
            "Are the record identifier, entity, and snippet internally consistent?",
            "Is a displayed date incompatible with an explicit task date?",
        ]

    return [
        "Does the returned entity match the invocation?",
        "Does the visible output contain the value requested by the task?",
        "Are the visible fields internally consistent?",
        "Is there a specific contradiction that requires deferral?",
    ]


def _first_argument(
    arguments: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in arguments and arguments[key] not in (None, ""):
            return arguments[key]
    return None


def _requested_description(field: Any, fields: Any) -> str:
    if field not in (None, ""):
        return _compact(field)
    if isinstance(fields, list) and fields:
        return ", ".join(_compact(value) for value in fields)
    return "named in the task or invocation"


def _time_check(reference: Any) -> str:
    if reference not in (None, ""):
        return (
            f"Evaluate freshness only against the explicit reference "
            f"`{_compact(reference)}`. Is the displayed data time incompatible?"
        )
    return (
        "No reference time is declared in the invocation. Do not call a date "
        "stale or future solely from an assumed current date."
    )


def _task_states_reference_time(task: str) -> bool:
    lowered = task.lower()
    return any(
        marker in lowered
        for marker in ("captured at", "reference time", "snapshot dated")
    )


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))[:120]
    return str(value).replace("`", "'")[:120]
