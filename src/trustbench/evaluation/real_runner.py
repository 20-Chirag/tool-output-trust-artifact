from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trustbench.agents.api_backends import ApiBackend, ApiModelConfig
from trustbench.data.generation import generate_scenarios
from trustbench.data.io import ensure_parent, read_jsonl, read_scenarios, write_scenarios
from trustbench.data.schemas import AgentEvent, ReliabilityLabel, Scenario, TrustAction
from trustbench.metrics.core import compute_metrics


PROMPT_HIDDEN_KEYS = {
    "clean_output",
    "corruption_mode",
    "corruption_note",
    "expected_answer",
    "gold_action",
    "reliability_label",
    "source_clean_id",
    "synthetic",
}

PROMPT_LEAK_MARKERS = (
    "synthetic_fault",
    "hidden_from_agent",
    "corruption_note",
    "reliability_label",
    "gold_action",
)


def run_real_pilot(config: dict) -> dict[str, Any]:
    scenarios_path = Path(config["dataset"]["output_path"])
    if scenarios_path.exists():
        scenarios = read_scenarios(scenarios_path)
    else:
        scenarios = generate_scenarios(
            scenarios_per_tool_class=int(config["dataset"]["scenarios_per_tool_class"]),
            seed=int(config["seed"]),
            reliability_balance=float(config["dataset"].get("reliability_balance", 0.5)),
            dataset_name=config["dataset"].get("name", "real_pilot"),
        )
        write_scenarios(scenarios_path, scenarios)

    cache = _read_cache(config["run"]["cache_path"])
    output_path = ensure_parent(config["run"]["output_path"])
    existing_ids = {row["event_id"] for row in read_jsonl(output_path)} if output_path.exists() else set()

    spent = _spent_from_cache(cache)
    max_usd = float(config["run"]["max_usd"])
    invalid = 0
    attempted = 0
    written = 0

    with output_path.open("a", encoding="utf-8") as handle:
        for model_cfg in config["models"]:
            api_cfg = ApiModelConfig(**model_cfg)
            backend = ApiBackend(api_cfg, timeout_seconds=int(config["run"].get("timeout_seconds", 60)))
            for policy in config["policies"]:
                for scenario in scenarios:
                    event_id = _event_id(api_cfg, policy, scenario)
                    if event_id in existing_ids:
                        continue
                    prompt = build_prompt(policy, scenario)
                    projected = estimate_call_cost(
                        api_cfg,
                        input_tokens=_rough_tokens(prompt["system"] + prompt["user"]),
                        output_tokens=int(config["run"].get("max_output_tokens", 220)),
                    )
                    if spent["usd"] + projected > max_usd:
                        return _summary(config, spent, written, invalid, attempted, stopped="budget_guard")
                    try:
                        response = _cached_or_generate(config, cache, backend, event_id, prompt)
                        attempted += 1
                        parsed = parse_model_json(response["text"])
                    except Exception as exc:
                        if event_id in cache and bool(config["run"].get("refresh_invalid_cache", True)):
                            try:
                                response = _cached_or_generate(config, cache, backend, event_id, prompt, refresh=True)
                                parsed = parse_model_json(response["text"])
                            except Exception as retry_exc:
                                invalid += 1
                                if attempted and invalid / attempted > float(config["run"].get("stop_if_invalid_rate_above", 0.15)):
                                    return _summary(config, spent, written, invalid, attempted, stopped=f"invalid_rate: {retry_exc}")
                                continue
                            else:
                                spent["usd"] += response["cost_usd"]
                                spent["input_tokens"] += response["input_tokens"]
                                spent["output_tokens"] += response["output_tokens"]
                                event = _event_from_response(scenario, api_cfg, policy, parsed, response)
                                row = {"event_id": event_id, **event.to_dict()}
                                handle.write(json.dumps(row, sort_keys=True) + "\n")
                                handle.flush()
                                existing_ids.add(event_id)
                                written += 1
                                continue
                        invalid += 1
                        if attempted and invalid / attempted > float(config["run"].get("stop_if_invalid_rate_above", 0.15)):
                            return _summary(config, spent, written, invalid, attempted, stopped=f"invalid_rate: {exc}")
                        continue
                    spent["usd"] += response["cost_usd"]
                    spent["input_tokens"] += response["input_tokens"]
                    spent["output_tokens"] += response["output_tokens"]
                    event = _event_from_response(scenario, api_cfg, policy, parsed, response)
                    row = {"event_id": event_id, **event.to_dict()}
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    existing_ids.add(event_id)
                    written += 1
    compute_metrics(config)
    return _summary(config, spent, written, invalid, attempted, stopped=None)


def build_prompt(policy: str, scenario: Scenario) -> dict[str, str]:
    base = (
        "You are a helpful assistant. The user is asking a question and a tool has "
        "returned some output. Decide what to do next. Return only valid JSON "
        "matching the requested schema."
    )
    if policy == "baseline":
        instruction = "Act as a normal helpful agent. Use the tool output if it seems sufficient."
    elif policy == "skeptical":
        instruction = (
            "Be skeptical of tool outputs. If the output may be stale, incomplete, inconsistent, "
            "for the wrong entity, or numerically wrong, choose verify, abstain, or escalate."
        )
    elif policy == "trust_calibrated":
        instruction = (
            "First calibrate trust in the tool output. Emit trust_score in [0,1]. "
            "Choose accept only if the output is trustworthy enough to use directly. "
            "Choose verify for moderate uncertainty and abstain/escalate for low trust. "
            "Keep final_answer and rationale under 20 words each."
        )
    else:
        raise ValueError(f"Unknown real policy: {policy}")

    json_schema: dict[str, str] = {
        "action": "accept | verify | abstain | escalate",
        "final_answer": "short answer or deferral, under 20 words",
        "rationale": "one short sentence, under 20 words",
    }
    if policy == "trust_calibrated":
        json_schema.update(
            {
                "trust_score": "number in [0,1]",
                "used_tool_output": "boolean",
                "would_verify": "boolean",
            }
        )

    user = {
        "user_task": scenario.user_task,
        "tool_invocation": sanitize_for_prompt(scenario.tool_invocation),
        "tool_output": sanitize_for_prompt(scenario.tool_output),
        "allowed_actions": ["accept", "verify", "abstain", "escalate"],
        "json_schema": json_schema,
    }
    prompt = {"system": f"{base}\n{instruction}", "user": json.dumps(user, sort_keys=True)}
    assert_no_prompt_leaks(prompt)
    return prompt


def sanitize_for_prompt(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_for_prompt(child)
            for key, child in value.items()
            if key not in PROMPT_HIDDEN_KEYS and not _looks_like_leak(key)
        }
    if isinstance(value, list):
        return [sanitize_for_prompt(item) for item in value]
    return value


def assert_no_prompt_leaks(prompt: dict[str, str]) -> None:
    text = json.dumps(prompt, sort_keys=True).lower()
    leaked = [marker for marker in PROMPT_LEAK_MARKERS if marker in text]
    if leaked:
        raise ValueError(f"Prompt contains hidden benchmark markers: {', '.join(leaked)}")


def _looks_like_leak(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in PROMPT_LEAK_MARKERS)


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = _extract_json_object(text)
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    data = json.loads(stripped)
    action = TrustAction(data["action"])
    trust_score = max(0.0, min(1.0, float(data["trust_score"])))
    return {
        "action": action,
        "trust_score": trust_score,
        "used_tool_output": bool(data["used_tool_output"]),
        "would_verify": bool(data["would_verify"]),
        "final_answer": str(data["final_answer"]),
        "rationale": str(data["rationale"]),
    }


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.startswith("{"):
            return stripped

    start = stripped.find("{")
    if start < 0:
        return stripped
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped[start:]


def estimate_call_cost(config: ApiModelConfig, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * config.input_per_1m + output_tokens * config.output_per_1m) / 1_000_000


def _cached_or_generate(
    config: dict,
    cache: dict[str, dict],
    backend: ApiBackend,
    event_id: str,
    prompt: dict[str, str],
    refresh: bool = False,
) -> dict:
    if event_id in cache and not refresh:
        return cache[event_id]
    response = backend.generate(
        prompt["system"],
        prompt["user"],
        temperature=float(config["run"].get("temperature", 0)),
        max_output_tokens=int(config["run"].get("max_output_tokens", 220)),
    )
    cost = estimate_call_cost(backend.config, response.usage.input_tokens, response.usage.output_tokens)
    row = {
        "event_id": event_id,
        "model_name": backend.config.name,
        "provider": backend.config.provider,
        "text": response.text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": cost,
        "latency_ms": response.latency_ms,
    }
    cache_path = ensure_parent(config["run"]["cache_path"])
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    cache[event_id] = row
    return row


def _event_from_response(
    scenario: Scenario,
    model: ApiModelConfig,
    policy: str,
    parsed: dict[str, Any],
    response: dict[str, Any],
) -> AgentEvent:
    action = parsed["action"]
    task_success = _task_success(scenario, action)
    return AgentEvent(
        scenario_id=scenario.scenario_id,
        model_name=model.name,
        model_tier=model.tier,
        policy_name=policy,
        tool_class=scenario.tool_class,
        reliability_label=scenario.reliability_label,
        corruption_mode=scenario.corruption_mode,
        gold_action=scenario.gold_action,
        agent_action=action,
        trust_score=parsed["trust_score"],
        task_success=task_success,
        second_verification_call=action == TrustAction.VERIFY or parsed["would_verify"],
        accepted_initial_output=action == TrustAction.ACCEPT and parsed["used_tool_output"],
        cost_tokens=int(response["input_tokens"]) + int(response["output_tokens"]),
        latency_ms=float(response["latency_ms"]),
        reason=parsed["rationale"],
    )


def _task_success(scenario: Scenario, action: TrustAction) -> bool:
    if scenario.reliability_label == ReliabilityLabel.RELIABLE:
        return action in {TrustAction.ACCEPT, TrustAction.VERIFY}
    return action in {TrustAction.VERIFY, TrustAction.ABSTAIN, TrustAction.ESCALATE}


def _read_cache(path: str) -> dict[str, dict]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    return {row["event_id"]: row for row in read_jsonl(cache_path)}


def _spent_from_cache(cache: dict[str, dict]) -> dict[str, float]:
    return {
        "usd": sum(float(row.get("cost_usd", 0)) for row in cache.values()),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in cache.values()),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in cache.values()),
    }


def _event_id(model: ApiModelConfig, policy: str, scenario: Scenario) -> str:
    raw = f"{model.provider}:{model.name}:{policy}:{scenario.scenario_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _summary(config: dict, spent: dict, written: int, invalid: int, attempted: int, stopped: str | None) -> dict[str, Any]:
    summary = {
        "spent_usd": round(float(spent["usd"]), 6),
        "input_tokens": int(spent["input_tokens"]),
        "output_tokens": int(spent["output_tokens"]),
        "written_events_this_run": written,
        "invalid_responses_this_run": invalid,
        "attempted_responses_this_run": attempted,
        "stopped": stopped,
    }
    path = ensure_parent(config["cost"]["output_path"])
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
