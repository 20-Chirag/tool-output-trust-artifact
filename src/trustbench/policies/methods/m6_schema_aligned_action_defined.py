"""Schema-aligned checklist with explicit trust-action semantics.

This post-hoc policy is a separately versioned construct repair. It preserves
the visible-schema checklist from ``m6_schema_aligned`` while defining how the
four actions map to use of the initial tool output.
"""
from __future__ import annotations

import json

from trustbench.data.schemas import Scenario
from trustbench.evaluation.real_runner import (
    assert_no_prompt_leaks,
    sanitize_for_prompt,
)
from trustbench.policies.methods import m6_schema_aligned


SYSTEM_BASE = (
    "You are a helpful assistant. The user is asking a question and a tool has "
    "returned some output. Decide what to do next. Return only valid JSON "
    "matching the requested schema."
)

ACTION_DEFINITIONS = (
    "Use these action meanings exactly:\n"
    "- accept: use the visible tool output directly and unchanged as the answer; "
    "choose this only when your checks support it.\n"
    "- verify: do not treat the visible value as settled; obtain or perform an "
    "independent check before answering.\n"
    "- abstain: the visible output is insufficient or contradicted and you "
    "cannot safely answer from it.\n"
    "- escalate: human or user input is required.\n"
    "If your own calculation or comparison contradicts the tool output, you "
    "must not choose accept. The action, used_tool_output, would_verify, "
    "final_answer, and rationale must agree. For accept, set used_tool_output "
    "to true and would_verify to false. For verify, set used_tool_output to "
    "false and would_verify to true. For abstain or escalate, set both booleans "
    "to false."
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
    checklist = m6_schema_aligned._checklist_for(scenario, invocation)
    checklist_text = "\n".join(
        f"  {index}. {item}" for index, item in enumerate(checklist, start=1)
    )
    instruction = (
        f"{ACTION_DEFINITIONS}\n\n{COMMON_RULES}\n\n"
        f"Checklist:\n{checklist_text}\n\n"
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
