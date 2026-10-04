"""M6 — Pre-specified checklist (Tools Fail replication).

Provides a fixed, pre-specified checklist of the most common benign
tool-output failure modes. The model works through each item, then
decides. Unlike M2 (which asks the model to generate its own concerns),
this gives the model the failure taxonomy upfront.

This directly replicates the "Checklist" intervention from Sun et al.
2024 (Tools Fail, ACL 2025), which was their best-performing method.
Running this enables a direct paper comparison: our benchmark, their
best prompt, side-by-side with our M2 structured-skeptic chain.

The checklist is tool-class-aware: items irrelevant to the tool class
are omitted to reduce prompt bloat and confusion.

Costs API calls (~same as M2).
"""
from __future__ import annotations

import json

from trustbench.evaluation.real_runner import assert_no_prompt_leaks, sanitize_for_prompt
from trustbench.data.schemas import Scenario, ToolClass


# Failure modes from the Tools Fail taxonomy, adapted for each class.
CHECKLISTS: dict[str, list[str]] = {
    "web_search": [
        "Is the result possibly stale or outdated (published date is old)?",
        "Is the source URL missing or unverifiable?",
        "Is the snippet inconsistent with the query (wrong entity or topic)?",
        "Does the snippet contain a plausible but unverified number or fact?",
        "Could this be a near-miss result for a different, similarly-named entity?",
    ],
    "calculator": [
        "Does the numeric result match a quick mental check of the expression?",
        "Is the result off by one or by a small round number (off-by-one error)?",
        "Is the engine name unfamiliar or different from the expected engine?",
        "Could the expression have an edge case (integer overflow, precision loss)?",
        "Does the result seem implausibly large, small, or negative?",
    ],
    "structured_api": [
        "Are any required fields (user_id, balance, currency, status) missing?",
        "Is the schema_version different from the expected version?",
        "Does the returned user_id match the requested user_id?",
        "Are any field values null, empty, or malformed?",
        "Is the timestamp suspiciously old or in the future?",
    ],
    "file_memory": [
        "Does the retrieved_entity exactly match the requested_entity?",
        "Is the last_updated date very old (possible stale memory)?",
        "Does the snippet mention a different owner or project than requested?",
        "Does the document_id or snippet look like it belongs to a different record?",
        "Could this be a near-duplicate match for a different but similar entry?",
    ],
}

SYSTEM_BASE = (
    "You are a helpful assistant. The user is asking a question and a tool has "
    "returned some output. Decide what to do next. Return only valid JSON "
    "matching the requested schema."
)

INSTRUCTION_TMPL = (
    "Work through the following checklist of possible failure modes for this tool output. "
    "For each item, consider whether it applies. Then, based on your checklist assessment, "
    "assign trust_score and choose action. "
    "Do not accept if any checklist item applies and is unresolved. "
    "Keep final_answer and rationale under 20 words each.\n\n"
    "Checklist:\n{checklist}"
)


def build_prompt(scenario: Scenario) -> dict[str, str]:
    tool_class_str = (
        scenario.tool_class.value
        if hasattr(scenario.tool_class, "value")
        else str(scenario.tool_class)
    )
    items = CHECKLISTS.get(tool_class_str, [
        "Is this output potentially stale, incomplete, or incorrect?",
        "Is the output from the expected source or entity?",
        "Are any required fields missing or suspicious?",
    ])
    checklist_text = "\n".join(f"  {i + 1}. {item}" for i, item in enumerate(items))
    instruction = INSTRUCTION_TMPL.format(checklist=checklist_text)

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
