from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trustbench.agents.api_backends import ApiModelConfig
from trustbench.data.io import ensure_parent, read_jsonl, read_scenarios
from trustbench.data.schemas import ReliabilityLabel, Scenario, TrustAction
from trustbench.evaluation.real_runner import _rough_tokens, build_prompt as build_real_prompt, estimate_call_cost
from trustbench.policies.methods import (
    m0_naive,
    m0_naive_m2,
    m0_naive_m6,
    m0_naive_skeptical,
    m2_structured_skeptic,
    m6_checklist,
    m6_schema_aligned,
    m6_schema_aligned_action_defined,
)
from trustbench.utils.env import load_dotenv


PHASE3_POLICIES = ["baseline", "m0_naive", "m6_checklist", "m0_naive_m6"]
PHASE3_TOOL_CLASSES = ["calculator", "structured_api"]
F1_TARGET_CORRUPTIONS = {"off_by_one_or_arithmetic_bug", "hidden_edge_case_failure"}
M2_POLICIES = {"m2_structured_skeptic", "m0_naive_m2"}

PROMPT_BUILDERS: dict[str, Callable[[Scenario], dict[str, str]]] = {
    "baseline": lambda scenario: build_real_prompt("baseline", scenario),
    "skeptical": lambda scenario: build_real_prompt("skeptical", scenario),
    "m0_naive": m0_naive.build_prompt,
    "m0_naive_skeptical": m0_naive_skeptical.build_prompt,
    "m6_checklist": m6_checklist.build_prompt,
    "m6_schema_aligned": m6_schema_aligned.build_prompt,
    "m6_schema_aligned_action_defined": (
        m6_schema_aligned_action_defined.build_prompt
    ),
    "m0_naive_m6": m0_naive_m6.build_prompt,
    "m2_structured_skeptic": m2_structured_skeptic.build_prompt,
    "m0_naive_m2": m0_naive_m2.build_prompt,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 minimal robustness setup.")
    parser.add_argument("--config", default="configs/phase_3_minimal.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.dry_run:
        if not bool(config.get("run", {}).get("paid_run_authorized", False)):
            raise SystemExit(
                "Refusing Phase 3 paid run: run.paid_run_authorized is false. "
                "Run with --dry-run and get explicit user approval first."
            )
        summary = run_paid_phase3(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    summary = build_dry_run_summary(config)
    ensure_parent(config["cost"]["output_path"]).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def run_paid_phase3(config: dict) -> dict[str, Any]:
    load_dotenv()
    scenarios = filter_phase3_scenarios(config)
    policies = validate_policies(config["policies"])
    models = [ApiModelConfig(**model) for model in config["models"]]
    unsupported = [model.provider for model in models if model.provider not in {"gemini", "openai"}]
    if unsupported:
        raise ValueError(f"Unsupported providers for this runner: {unsupported}")

    max_output_tokens = int(config["run"]["max_output_tokens"])
    max_usd = float(config["run"]["max_usd"])
    projected = project_cost(scenarios, policies, models, max_output_tokens)
    if projected > max_usd:
        raise ValueError(f"Projected cost {projected:.6f} exceeds configured cap {max_usd:.6f}")

    output_path = ensure_parent(config["run"]["output_path"])
    cache_path = ensure_parent(config["run"]["cache_path"])
    run_summary_path = ensure_parent(Path(config["run"]["output_dir"]) / "run_summary.json")

    cache = _read_cache(cache_path)
    existing = {row["event_id"] for row in read_jsonl(output_path)} if output_path.exists() else set()
    spent = _spent_from_cache_lines(cache_path)
    written = 0
    attempted = 0
    invalid = 0

    with output_path.open("a", encoding="utf-8") as handle:
        for model in models:
            for policy in policies:
                for scenario in scenarios:
                    event_id = _event_id(model, policy, scenario)
                    if event_id in existing:
                        continue
                    prompt = PROMPT_BUILDERS[policy](scenario)
                    thinking_level, thinking_budget = _gemini_thinking_for_model(config, model)
                    projected_call = estimate_call_cost(
                        model,
                        _rough_tokens(prompt["system"] + prompt["user"]),
                        max_output_tokens,
                    )
                    if spent + projected_call > max_usd:
                        return _write_run_summary(
                            path=run_summary_path,
                            config=config,
                            scenarios=scenarios,
                            policies=policies,
                            models=models,
                            projected_usd=projected,
                            spent_usd=spent,
                            written=written,
                            attempted=attempted,
                            invalid=invalid,
                            stopped="budget_guard",
                        )
                    attempted += 1
                    try:
                        response = _cached_or_call(
                            model=model,
                            policy=policy,
                            cache=cache,
                            cache_path=cache_path,
                            event_id=event_id,
                            prompt=prompt,
                            timeout_seconds=int(config["run"].get("timeout_seconds", 180)),
                            temperature=float(config["run"].get("temperature", 0)),
                            max_output_tokens=max_output_tokens,
                            thinking_level=thinking_level,
                            thinking_budget=thinking_budget,
                        )
                        parsed = _parse_decision_json(response["text"])
                    except Exception as exc:
                        if event_id in cache and bool(config["run"].get("refresh_invalid_cache", True)):
                            retry_tokens = int(config["run"].get("retry_max_output_tokens", max_output_tokens))
                            projected_retry = estimate_call_cost(
                                model,
                                _rough_tokens(prompt["system"] + prompt["user"]),
                                retry_tokens,
                            )
                            if _spent_from_cache_lines(cache_path) + projected_retry > max_usd:
                                return _write_run_summary(
                                    path=run_summary_path,
                                    config=config,
                                    scenarios=scenarios,
                                    policies=policies,
                                    models=models,
                                    projected_usd=projected,
                                    spent_usd=_spent_from_cache_lines(cache_path),
                                    written=written,
                                    attempted=attempted,
                                    invalid=invalid,
                                    stopped="budget_guard_before_invalid_refresh",
                                )
                            try:
                                attempted += 1
                                response = _cached_or_call(
                                    model=model,
                                    policy=policy,
                                    cache=cache,
                                    cache_path=cache_path,
                                    event_id=event_id,
                                    prompt=prompt,
                                    timeout_seconds=int(config["run"].get("timeout_seconds", 180)),
                                    temperature=float(config["run"].get("temperature", 0)),
                                    max_output_tokens=retry_tokens,
                                    thinking_level=thinking_level,
                                    thinking_budget=thinking_budget,
                                    refresh=True,
                                )
                                parsed = _parse_decision_json(response["text"])
                            except Exception as retry_exc:
                                invalid += 1
                                if _invalid_rate_too_high(config, attempted, invalid):
                                    return _write_run_summary(
                                        path=run_summary_path,
                                        config=config,
                                        scenarios=scenarios,
                                        policies=policies,
                                        models=models,
                                        projected_usd=projected,
                                        spent_usd=_spent_from_cache_lines(cache_path),
                                        written=written,
                                        attempted=attempted,
                                        invalid=invalid,
                                        stopped=f"invalid_rate: {retry_exc}",
                                    )
                                continue
                        else:
                            invalid += 1
                            if _invalid_rate_too_high(config, attempted, invalid):
                                return _write_run_summary(
                                    path=run_summary_path,
                                    config=config,
                                    scenarios=scenarios,
                                    policies=policies,
                                    models=models,
                                    projected_usd=projected,
                                    spent_usd=_spent_from_cache_lines(cache_path),
                                    written=written,
                                    attempted=attempted,
                                    invalid=invalid,
                                    stopped=f"invalid_rate: {exc}",
                                )
                            continue

                    spent = _spent_from_cache_lines(cache_path)
                    row = _event_row(event_id, scenario, model, policy, parsed, response)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    existing.add(event_id)
                    written += 1
                    if written % 25 == 0:
                        print(
                            "[phase_3] events={events} spent_usd={usd:.6f} spent_inr={inr:.2f}".format(
                                events=len(existing),
                                usd=spent,
                                inr=spent * 83,
                            )
                        )

    return _write_run_summary(
        path=run_summary_path,
        config=config,
        scenarios=scenarios,
        policies=policies,
        models=models,
        projected_usd=projected,
        spent_usd=spent,
        written=written,
        attempted=attempted,
        invalid=invalid,
        stopped=None,
    )


def build_dry_run_summary(config: dict) -> dict:
    scenarios = filter_phase3_scenarios(config)
    policies = validate_policies(config["policies"])
    models = [ApiModelConfig(**model) for model in config["models"]]
    max_output_tokens = int(config["run"]["max_output_tokens"])
    projected = project_cost(scenarios, policies, models, max_output_tokens)

    by_tool = _counts_by_tool_and_reliability(scenarios)
    summary = {
        "phase": config.get("phase", "phase_3_minimal_robustness"),
        "scenario_path": config["source"]["scenarios_path"],
        "selected_tool_classes": list(config["source"].get("include_tool_classes", PHASE3_TOOL_CLASSES)),
        "excluded_tool_classes": _excluded_tool_classes(config),
        "scenarios": len(scenarios),
        "scenarios_by_tool_class": dict(Counter(_tool_value(scenario) for scenario in scenarios)),
        "reliability_by_tool_class": by_tool,
        "policies": policies,
        "policy_count": len(policies),
        "models": [model.name for model in models],
        "providers": [model.provider for model in models],
        "model_pricing": [
            {
                "model": model.name,
                "provider": model.provider,
                "input_per_1m": model.input_per_1m,
                "output_per_1m": model.output_per_1m,
                "pricing_basis": _pricing_basis(model),
                "gemini_thinking": _gemini_thinking_summary(config, model),
            }
            for model in models
        ],
        "events": len(scenarios) * len(policies) * len(models),
        "max_output_tokens": max_output_tokens,
        "projected_usd": round(projected, 6),
        "projected_inr": round(projected * 83, 2),
        "budget_cap_usd": float(config["run"]["max_usd"]),
        "budget_cap_inr": round(float(config["run"]["max_usd"]) * 83, 2),
        "fits_budget": projected <= float(config["run"]["max_usd"]),
        "output_dir": config["run"]["output_dir"],
        "output_path": config["run"]["output_path"],
        "cache_path": config["run"]["cache_path"],
        "paid_run_authorized": bool(config["run"].get("paid_run_authorized", False)),
        "will_create_paid_outputs": False,
        "f1_target_scenarios": _f1_target_count(scenarios),
        "f4_policy_pairs": _available_policy_pairs(policies),
        "pricing_sources": [
            "https://developers.openai.com/api/docs/pricing",
            "https://ai.google.dev/gemini-api/docs/models",
            "https://ai.google.dev/gemini-api/docs/pricing",
        ],
    }
    validate_dry_run_summary(summary, config)
    return summary


def filter_phase3_scenarios(config: dict) -> list[Scenario]:
    scenarios = read_scenarios(config["source"]["scenarios_path"])
    include = set(config["source"].get("include_tool_classes", PHASE3_TOOL_CLASSES))
    selected = [scenario for scenario in scenarios if _tool_value(scenario) in include]
    validate_scenario_subset(selected, config)
    return selected


def validate_scenario_subset(scenarios: list[Scenario], config: dict) -> None:
    expected_total = int(config["source"].get("expected_scenarios", 80))
    expected_per_tool = int(config["source"].get("expected_per_tool_class", 40))
    expected_reliable = int(config["source"].get("expected_reliable_per_tool_class", 20))
    expected_unreliable = int(config["source"].get("expected_unreliable_per_tool_class", 20))

    if len(scenarios) != expected_total:
        raise ValueError(f"Expected {expected_total} Phase 3 scenarios, found {len(scenarios)}")
    expected_tools = list(config["source"].get("include_tool_classes", PHASE3_TOOL_CLASSES))
    tools = Counter(_tool_value(scenario) for scenario in scenarios)
    if set(tools) != set(expected_tools):
        raise ValueError(f"Unexpected Phase 3 tool classes: {dict(tools)}")
    for tool_class in expected_tools:
        if tools[tool_class] != expected_per_tool:
            raise ValueError(f"Expected {expected_per_tool} {tool_class} scenarios, found {tools[tool_class]}")
        labels = Counter(
            _reliability_value(scenario) for scenario in scenarios if _tool_value(scenario) == tool_class
        )
        if labels["reliable"] != expected_reliable or labels["unreliable"] != expected_unreliable:
            raise ValueError(f"Unexpected reliability balance for {tool_class}: {dict(labels)}")


def validate_policies(policies: list[str]) -> list[str]:
    if policies == PHASE3_POLICIES:
        return list(policies)
    unknown = [policy for policy in policies if policy not in PROMPT_BUILDERS]
    if unknown:
        raise ValueError(f"Unsupported policies: {unknown}")
    if len(policies) != len(set(policies)):
        raise ValueError(f"Duplicate policies are not allowed: {policies}")
    if len(policies) == 0:
        raise ValueError("At least one policy is required")
    if all(policy in PHASE3_POLICIES for policy in policies):
        return list(policies)
    if set(policies).issubset(set(PROMPT_BUILDERS)):
        return list(policies)
    if policies != PHASE3_POLICIES:
        raise ValueError(f"Phase 3 policies must be exactly {PHASE3_POLICIES}; got {policies}")
    return list(policies)


def project_cost(
    scenarios: list[Scenario],
    policies: list[str],
    models: list[ApiModelConfig],
    max_output_tokens: int,
) -> float:
    total = 0.0
    for scenario in scenarios:
        for policy in policies:
            prompt = PROMPT_BUILDERS[policy](scenario)
            prompt_tokens = _rough_tokens(prompt["system"] + prompt["user"])
            for model in models:
                total += estimate_call_cost(model, prompt_tokens, max_output_tokens)
    return total


def validate_dry_run_summary(summary: dict, config: dict) -> None:
    expected_events = int(config["source"]["expected_scenarios"]) * len(config["policies"]) * len(config["models"])
    if summary["events"] != expected_events:
        raise ValueError(f"Expected {expected_events} dry-run events, found {summary['events']}")
    if summary["paid_run_authorized"]:
        raise ValueError("Dry-run summary cannot be paid-run authorized")
    if summary["will_create_paid_outputs"]:
        raise ValueError("Dry-run must not create paid run outputs")
    if not summary["fits_budget"]:
        raise ValueError("Projected Phase 3 dry-run cost exceeds configured cap")


def _gemini_thinking_for_model(config: dict, model: ApiModelConfig) -> tuple[str | None, int | None]:
    if model.provider != "gemini":
        return None, None

    by_model = config.get("run", {}).get("gemini_thinking", {})
    if model.name in by_model:
        settings = by_model[model.name] or {}
        has_level = "thinking_level" in settings and settings["thinking_level"] is not None
        has_budget = "thinking_budget" in settings and settings["thinking_budget"] is not None
        if has_level and has_budget:
            raise ValueError(f"Gemini model {model.name} cannot set both thinking_level and thinking_budget")
        if has_budget:
            return None, int(settings["thinking_budget"])
        if has_level:
            return str(settings["thinking_level"]), None
        return None, None

    has_level = config.get("run", {}).get("thinking_level") is not None
    has_budget = config.get("run", {}).get("thinking_budget") is not None
    if has_level and has_budget:
        raise ValueError("Run config cannot set both thinking_level and thinking_budget as defaults")
    if has_budget:
        return None, int(config["run"]["thinking_budget"])
    if has_level:
        return str(config["run"]["thinking_level"]), None
    return None, None


def _cached_or_call(
    model: ApiModelConfig,
    policy: str,
    cache: dict[str, dict],
    cache_path: Path,
    event_id: str,
    prompt: dict[str, str],
    timeout_seconds: int,
    temperature: float,
    max_output_tokens: int,
    thinking_level: str | None,
    thinking_budget: int | None,
    refresh: bool = False,
) -> dict:
    if event_id in cache and not refresh:
        return cache[event_id]
    started = time.perf_counter()
    if model.provider == "openai":
        text, input_tokens, output_tokens = _call_openai(
            model, policy, prompt, timeout_seconds, temperature, max_output_tokens
        )
    elif model.provider == "gemini":
        text, input_tokens, output_tokens = _call_gemini(
            model, policy, prompt, timeout_seconds, temperature, max_output_tokens, thinking_level, thinking_budget
        )
    else:
        raise ValueError(f"Unsupported Phase 3 provider: {model.provider}")
    latency_ms = (time.perf_counter() - started) * 1000
    row = {
        "event_id": event_id,
        "model_name": model.name,
        "provider": model.provider,
        "policy_name": policy,
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": estimate_call_cost(model, input_tokens, output_tokens),
        "latency_ms": latency_ms,
    }
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    cache[event_id] = row
    return row


def _call_openai(
    model: ApiModelConfig,
    policy: str,
    prompt: dict[str, str],
    timeout_seconds: int,
    temperature: float,
    max_output_tokens: int,
) -> tuple[str, int, int]:
    payload = {
        "model": model.name,
        "input": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": f"{policy}_decision",
                "strict": True,
                "schema": _openai_schema(policy),
            }
        },
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    data = _urlopen_json(
        urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {_api_key(model)}",
                "Content-Type": "application/json",
            },
            method="POST",
        ),
        timeout_seconds,
    )
    text = _extract_openai_text(data)
    usage = data.get("usage", {})
    return (
        text,
        int(usage.get("input_tokens", _rough_tokens(prompt["system"] + prompt["user"]))),
        int(usage.get("output_tokens", _rough_tokens(text))),
    )


def _call_gemini(
    model: ApiModelConfig,
    policy: str,
    prompt: dict[str, str],
    timeout_seconds: int,
    temperature: float,
    max_output_tokens: int,
    thinking_level: str | None,
    thinking_budget: int | None,
) -> tuple[str, int, int]:
    payload = _gemini_payload(policy, prompt, temperature, max_output_tokens, thinking_level, thinking_budget)
    data = _urlopen_json(
        urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model.name}:generateContent?key={_api_key(model)}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout_seconds,
    )
    text = _extract_gemini_text(data)
    usage = data.get("usageMetadata", {})
    candidate_tokens = int(usage.get("candidatesTokenCount", _rough_tokens(text)))
    thought_tokens = int(usage.get("thoughtsTokenCount", 0))
    return (
        text,
        int(usage.get("promptTokenCount", _rough_tokens(prompt["system"] + prompt["user"]))),
        candidate_tokens + thought_tokens,
    )


def _gemini_payload(
    policy: str,
    prompt: dict[str, str],
    temperature: float,
    max_output_tokens: int,
    thinking_level: str | None,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
        "responseSchema": _gemini_schema(policy),
    }
    if thinking_budget is not None:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    elif thinking_level:
        generation_config["thinkingConfig"] = {"thinkingLevel": str(thinking_level)}
    return {
        "systemInstruction": {"parts": [{"text": prompt["system"]}]},
        "contents": [{"role": "user", "parts": [{"text": prompt["user"]}]}],
        "generationConfig": generation_config,
    }


def _gemini_schema(policy: str) -> dict[str, Any]:
    schema = {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "enum": ["accept", "verify", "abstain", "escalate"]},
            "trust_score": {
                "type": "NUMBER",
                "minimum": 0,
                "maximum": 1,
            },
            "used_tool_output": {"type": "BOOLEAN"},
            "would_verify": {"type": "BOOLEAN"},
            "final_answer": {"type": "STRING", "maxLength": 180},
            "rationale": {"type": "STRING", "maxLength": 260},
        },
        "required": ["action", "trust_score", "used_tool_output", "would_verify", "final_answer", "rationale"],
    }
    if policy in M2_POLICIES:
        schema["properties"]["observed_concerns"] = {
            "type": "ARRAY",
            "maxItems": 6,
            "items": {"type": "STRING", "maxLength": 160},
        }
        schema["required"] = ["observed_concerns", *schema["required"]]
    return schema


def _openai_schema(policy: str) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["accept", "verify", "abstain", "escalate"]},
            "trust_score": {"type": "number", "minimum": 0, "maximum": 1},
            "used_tool_output": {"type": "boolean"},
            "would_verify": {"type": "boolean"},
            "final_answer": {"type": "string", "maxLength": 180},
            "rationale": {"type": "string", "maxLength": 260},
        },
        "required": ["action", "trust_score", "used_tool_output", "would_verify", "final_answer", "rationale"],
    }
    if policy in M2_POLICIES:
        schema["properties"]["observed_concerns"] = {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 160},
        }
        schema["required"] = ["observed_concerns", *schema["required"]]
    return schema


def _parse_decision_json(text: str) -> dict[str, Any]:
    data = json.loads(_extract_json_object(text))
    action = TrustAction(data["action"])
    trust_score = float(data["trust_score"])
    if not 0.0 <= trust_score <= 1.0:
        raise ValueError(f"trust_score outside [0,1]: {trust_score}")
    used_tool_output = data["used_tool_output"]
    would_verify = data["would_verify"]
    if not isinstance(used_tool_output, bool):
        raise ValueError("used_tool_output must be a JSON boolean")
    if not isinstance(would_verify, bool):
        raise ValueError("would_verify must be a JSON boolean")
    concerns = data.get("observed_concerns", [])
    if not isinstance(concerns, list):
        concerns = [str(concerns)]
    return {
        "action": action,
        "trust_score": trust_score,
        "used_tool_output": used_tool_output,
        "would_verify": would_verify,
        "final_answer": str(data["final_answer"]),
        "rationale": str(data["rationale"]),
        "observed_concerns": [str(item)[:180] for item in concerns],
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


def _event_row(
    event_id: str,
    scenario: Scenario,
    model: ApiModelConfig,
    policy: str,
    parsed: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    action = parsed["action"]
    row = {
        "event_id": event_id,
        "scenario_id": scenario.scenario_id,
        "model_name": model.name,
        "model_tier": model.tier,
        "policy_name": policy,
        "tool_class": _tool_value(scenario),
        "reliability_label": _reliability_value(scenario),
        "corruption_mode": _corruption_value(scenario),
        "gold_action": scenario.gold_action.value,
        "agent_action": action.value,
        "trust_score": parsed["trust_score"],
        "task_success": _task_success(scenario, action),
        "second_verification_call": action == TrustAction.VERIFY or parsed["would_verify"],
        "accepted_initial_output": action == TrustAction.ACCEPT and parsed["used_tool_output"],
        "used_tool_output": parsed["used_tool_output"],
        "would_verify": parsed["would_verify"],
        "cost_tokens": int(response["input_tokens"]) + int(response["output_tokens"]),
        "latency_ms": float(response["latency_ms"]),
        "reason": parsed["rationale"][:260],
        "final_answer": parsed["final_answer"][:220],
    }
    if policy in M2_POLICIES:
        row["observed_concerns"] = parsed.get("observed_concerns", [])
    return row


def _task_success(scenario: Scenario, action: TrustAction) -> bool:
    if scenario.reliability_label == ReliabilityLabel.RELIABLE:
        return action in {TrustAction.ACCEPT, TrustAction.VERIFY}
    return action in {TrustAction.VERIFY, TrustAction.ABSTAIN, TrustAction.ESCALATE}


def _invalid_rate_too_high(config: dict, attempted: int, invalid: int) -> bool:
    if attempted < 20:
        return False
    return invalid / attempted > float(config["run"].get("stop_if_invalid_rate_above", 0.05))


def _read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["event_id"]: row for row in read_jsonl(path)}


def _spent_from_cache(cache: dict[str, dict]) -> float:
    return float(sum(float(row.get("cost_usd", 0.0)) for row in cache.values()))


def _spent_from_cache_lines(path: Path) -> float:
    if not path.exists():
        return 0.0
    return float(sum(float(row.get("cost_usd", 0.0)) for row in read_jsonl(path)))


def _event_id(model: ApiModelConfig, policy: str, scenario: Scenario) -> str:
    raw = f"phase_3_minimal:{model.provider}:{model.name}:{policy}:{scenario.scenario_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _write_run_summary(
    path: Path,
    config: dict,
    scenarios: list[Scenario],
    policies: list[str],
    models: list[ApiModelConfig],
    projected_usd: float,
    spent_usd: float,
    written: int,
    attempted: int,
    invalid: int,
    stopped: str | None,
) -> dict[str, Any]:
    cache_path = Path(config["run"]["cache_path"])
    cache_rows = read_jsonl(cache_path) if cache_path.exists() else []
    latest_cache = {row["event_id"]: row for row in cache_rows}
    cache_count = len(latest_cache)
    output_path = Path(config["run"]["output_path"])
    parsed_count = len(read_jsonl(output_path)) if output_path.exists() else 0
    spent_latest = _spent_from_cache(latest_cache)
    spent_all_cache_lines = float(sum(float(row.get("cost_usd", 0.0)) for row in cache_rows))
    summary = {
        "phase": config.get("phase", "phase_3_minimal_robustness"),
        "scenario_path": config["source"]["scenarios_path"],
        "scenarios": len(scenarios),
        "policies": policies,
        "models": [model.name for model in models],
        "events_expected": len(scenarios) * len(policies) * len(models),
        "events_parsed_total": parsed_count,
        "events_cached_total": cache_count,
        "written_events_this_run": written,
        "attempted_responses_this_run": attempted,
        "invalid_responses_this_run": invalid,
        "projected_usd": round(projected_usd, 6),
        "projected_inr_approx": round(projected_usd * 83, 2),
        "spent_usd": round(spent_usd, 6),
        "spent_inr_approx": round(spent_usd * 83, 2),
        "spent_usd_all_cache_lines": round(spent_all_cache_lines, 6),
        "spent_inr_all_cache_lines_approx": round(spent_all_cache_lines * 83, 2),
        "spent_usd_unique_latest": round(spent_latest, 6),
        "spent_inr_unique_latest_approx": round(spent_latest * 83, 2),
        "duplicate_cache_lines": len(cache_rows) - len(latest_cache),
        "budget_cap_usd": float(config["run"]["max_usd"]),
        "budget_cap_inr_approx": round(float(config["run"]["max_usd"]) * 83, 2),
        "paid_run_authorized": bool(config["run"].get("paid_run_authorized", False)),
        "stopped": stopped,
        "output_path": config["run"]["output_path"],
        "cache_path": config["run"]["cache_path"],
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _api_key(model: ApiModelConfig) -> str:
    import os

    value = os.environ.get(model.api_env)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {model.api_env}")
    return value


def _urlopen_json(
    request: urllib.request.Request,
    timeout_seconds: int,
    max_attempts: int = 4,
) -> dict[str, Any]:
    transient_http_codes = {408, 409, 425, 429, 500, 502, 503, 504}
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code not in transient_http_codes or attempt + 1 >= max_attempts:
                raise RuntimeError(f"HTTP {error.code}: {body[:800]}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as error:
            if attempt + 1 >= max_attempts:
                raise RuntimeError(
                    f"Transient network error after {max_attempts} attempts: {error}"
                ) from error
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("Network request retry loop exited unexpectedly")


def _extract_gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return json.dumps(data)
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(str(part.get("text", "")) for part in parts if "text" in part)


def _extract_openai_text(data: dict[str, Any]) -> str:
    if "output_text" in data:
        return str(data["output_text"])
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    if chunks:
        return "\n".join(chunks)
    return json.dumps(data)


def _counts_by_tool_and_reliability(scenarios: list[Scenario]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for scenario in scenarios:
        counts[_tool_value(scenario)][_reliability_value(scenario)] += 1
    return {tool: dict(labels) for tool, labels in counts.items()}


def _excluded_tool_classes(config: dict) -> list[str]:
    all_tools = ["web_search", "calculator", "structured_api", "file_memory"]
    selected = set(config["source"].get("include_tool_classes", PHASE3_TOOL_CLASSES))
    return [tool for tool in all_tools if tool not in selected]


def _pricing_basis(model: ApiModelConfig) -> str:
    if model.provider == "gemini":
        return "Google Gemini Developer API pricing from config"
    if model.provider == "openai":
        return "OpenAI API pricing from config"
    return "Provider pricing from config"


def _gemini_thinking_summary(config: dict, model: ApiModelConfig) -> dict[str, Any] | None:
    if model.provider != "gemini":
        return None
    level, budget = _gemini_thinking_for_model(config, model)
    if budget is not None:
        return {"thinkingBudget": budget}
    if level is not None:
        return {"thinkingLevel": level}
    return {}


def _available_policy_pairs(policies: list[str]) -> list[list[str]]:
    pairs = []
    for primed, naive in [
        ("baseline", "m0_naive"),
        ("m6_checklist", "m0_naive_m6"),
        ("m2_structured_skeptic", "m0_naive_m2"),
    ]:
        if primed in policies and naive in policies:
            pairs.append([primed, naive])
    return pairs


def _f1_target_count(scenarios: list[Scenario]) -> int:
    return sum(
        1
        for scenario in scenarios
        if _tool_value(scenario) == "calculator"
        and _reliability_value(scenario) == "unreliable"
        and _corruption_value(scenario) in F1_TARGET_CORRUPTIONS
    )


def _tool_value(scenario: Scenario) -> str:
    return scenario.tool_class.value if hasattr(scenario.tool_class, "value") else str(scenario.tool_class)


def _reliability_value(scenario: Scenario) -> str:
    label = scenario.reliability_label
    if isinstance(label, ReliabilityLabel):
        return label.value
    return str(label)


def _corruption_value(scenario: Scenario) -> str:
    mode = scenario.corruption_mode
    return mode.value if hasattr(mode, "value") else str(mode)


if __name__ == "__main__":
    main()
