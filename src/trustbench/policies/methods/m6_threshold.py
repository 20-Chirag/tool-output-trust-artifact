"""M6-threshold repair prompt for Phase 2.

This keeps the M6 checklist intervention but adds a specificity threshold:
the model should verify only when it can name at least two concrete concerns
supported by the visible output. Generic concerns about tool fallibility,
schema drift, stale timestamps, or unreliable web sources are not enough.
"""
from __future__ import annotations

import json

from trustbench.data.schemas import Scenario
from trustbench.evaluation.real_runner import assert_no_prompt_leaks, sanitize_for_prompt
from trustbench.policies.methods.m6_checklist import CHECKLISTS, SYSTEM_BASE


THRESHOLD_RULE = (
    "After working through the checklist, list only concerns that are SPECIFIC "
    "to this exact output - for example, a value that contradicts the requested "
    "entity, a timestamp older than reasonable for the task, or a field that "
    "contradicts another field in the same response. Do NOT list generic "
    "concerns such as 'the schema could be wrong,' 'timestamps could be stale,' "
    "or 'web sources may be inaccurate' unless you can point to a specific "
    "contradiction. Choose verify only if you can list at least two such "
    "specific concerns. Otherwise choose accept unless the output directly "
    "contradicts the task."
)

INSTRUCTION_TMPL = (
    "Work through the following checklist of possible failure modes for this tool output. "
    "For each item, consider whether it applies. Then decide whether any concern is "
    "specific enough to affect this exact output.\n\n"
    "{threshold_rule}\n\n"
    "Keep final_answer and rationale under 20 words each.\n\n"
    "Checklist:\n{checklist}"
)


def build_prompt(scenario: Scenario) -> dict[str, str]:
    tool_class_str = (
        scenario.tool_class.value
        if hasattr(scenario.tool_class, "value")
        else str(scenario.tool_class)
    )
    items = CHECKLISTS.get(
        tool_class_str,
        [
            "Is this output potentially stale, incomplete, or incorrect?",
            "Is the output from the expected source or entity?",
            "Are any required fields missing or suspicious?",
        ],
    )
    checklist_text = "\n".join(f"  {i + 1}. {item}" for i, item in enumerate(items))
    instruction = INSTRUCTION_TMPL.format(
        checklist=checklist_text,
        threshold_rule=THRESHOLD_RULE,
    )

    user = {
        "user_task": scenario.user_task,
        "tool_invocation": sanitize_for_prompt(scenario.tool_invocation),
        "tool_output": sanitize_for_prompt(scenario.tool_output),
        "allowed_actions": ["accept", "verify", "abstain", "escalate"],
        "json_schema": {
            "action": "accept | verify | abstain | escalate",
            "trust_score": "number in [0,1]",
            "used_tool_output": "boolean",
            "would_verify": "boolean",
            "final_answer": "short answer or deferral, under 20 words",
            "rationale": "one short sentence, under 20 words",
        },
    }
    prompt = {
        "system": f"{SYSTEM_BASE}\n{instruction}",
        "user": json.dumps(user, sort_keys=True),
    }
    assert_no_prompt_leaks(prompt)
    return prompt
