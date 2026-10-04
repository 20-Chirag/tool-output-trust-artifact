"""Evaluate the pre-registered URL-host-only fixture ablation."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from trustbench.data.io import read_jsonl
from trustbench.evaluation.decision_metrics import accepted_initial_output


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = ROOT / "outputs" / "url_fixture_ablation" / "scenarios_fixture.jsonl"
REAL_RUNS_PATH = ROOT / "outputs" / "real_web_domain_expansion_clean_run" / "runs.jsonl"
FIXTURE_RUNS_PATH = ROOT / "outputs" / "url_fixture_ablation_run" / "runs.jsonl"
OUT_DIR = ROOT / "outputs" / "url_fixture_ablation_run"
N_BOOT = 10_000
SEED = 20260722
MODELS = ("gpt-5.4-mini", "gemini-2.5-flash")
POLICIES = ("baseline", "m2_structured_skeptic", "m6_checklist")


def main() -> None:
    if not FIXTURE_RUNS_PATH.exists():
        raise SystemExit("No fixture paid-run output exists; evaluation is intentionally unavailable before approval and execution.")
    scenarios = {row["scenario_id"]: row for row in read_jsonl(SCENARIOS_PATH)}
    real = _index_selected(read_jsonl(REAL_RUNS_PATH))
    fixture = _index_selected(read_jsonl(FIXTURE_RUNS_PATH))
    _validate_cells(scenarios, real, fixture)

    paired = _paired_deltas(scenarios, real, fixture)
    by_mode = _corruption_mode_deltas(scenarios, real, fixture)
    examples = _changed_action_examples(scenarios, real, fixture)
    verdict = _verdict(paired)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(OUT_DIR / "paired_deltas.json", paired)
    _write_json(OUT_DIR / "corruption_mode_deltas.json", by_mode)
    _write_json(OUT_DIR / "examples.json", examples)
    _write_json(OUT_DIR / "verdict.json", verdict)
    _write_json(OUT_DIR / "metrics.json", _cell_metrics(real, fixture))
    (OUT_DIR / "evaluation_report.md").write_text(
        _render_report(paired, by_mode, examples, verdict), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


def _index_selected(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    selected = {}
    for row in rows:
        if row.get("model_name") not in MODELS or row.get("policy_name") not in POLICIES:
            continue
        key = (row["scenario_id"], row["model_name"], row["policy_name"])
        if key in selected:
            raise ValueError(f"Duplicate parsed cell: {key}")
        selected[key] = row
    return selected


def _validate_cells(
    scenarios: dict[str, dict[str, Any]],
    real: dict[tuple[str, str, str], dict[str, Any]],
    fixture: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    expected = {
        (scenario_id, model, policy)
        for scenario_id in scenarios
        for model in MODELS
        for policy in POLICIES
    }
    if set(real) != expected:
        raise ValueError(f"Real control cells differ from expected: missing={len(expected - set(real))}, extra={len(set(real) - expected)}")
    if set(fixture) != expected:
        raise ValueError(f"Fixture cells differ from expected: missing={len(expected - set(fixture))}, extra={len(set(fixture) - expected)}")


def _paired_deltas(scenarios: dict, real: dict, fixture: dict) -> list[dict[str, Any]]:
    capture_to_scenarios: dict[str, list[str]] = defaultdict(list)
    for scenario_id, scenario in scenarios.items():
        capture_to_scenarios[scenario["metadata"]["raw_capture_id"]].append(scenario_id)
    captures = sorted(capture_to_scenarios)
    rows = []
    for model_index, model in enumerate(MODELS):
        for policy_index, policy in enumerate(POLICIES):
            entry = {"model_name": model, "policy_name": policy, "n_capture_clusters": len(captures), "metrics": {}}
            for metric_index, (metric, eligible, score) in enumerate(_metric_specs()):
                rng = random.Random(SEED + model_index * 100 + policy_index * 10 + metric_index)
                entry["metrics"][metric] = _cluster_delta(
                    scenarios, real, fixture, model, policy, captures, capture_to_scenarios, eligible, score, rng
                )
            rows.append(entry)
    return rows


def _cluster_delta(
    scenarios: dict,
    real: dict,
    fixture: dict,
    model: str,
    policy: str,
    captures: list[str],
    capture_to_scenarios: dict[str, list[str]],
    eligible: Callable[[dict], bool],
    score: Callable[[dict], int],
    rng: random.Random,
) -> dict[str, Any]:
    def values(index: dict, sampled: list[str]) -> list[int]:
        out = []
        for capture_id in sampled:
            for scenario_id in capture_to_scenarios[capture_id]:
                scenario = scenarios[scenario_id]
                if eligible(scenario):
                    out.append(score(index[(scenario_id, model, policy)]))
        return out

    real_values = values(real, captures)
    fixture_values = values(fixture, captures)
    deltas = []
    for _ in range(N_BOOT):
        sampled = [rng.choice(captures) for _ in captures]
        deltas.append(_mean(values(fixture, sampled)) - _mean(values(real, sampled)))
    return {
        "delta_fixture_minus_real": _mean(fixture_values) - _mean(real_values),
        "ci_low": _quantile(deltas, 0.025),
        "ci_high": _quantile(deltas, 0.975),
        "real": _count_payload(real_values),
        "fixture": _count_payload(fixture_values),
        "paired_capture_clusters": len(captures),
        "ci_method": "paired source-capture percentile bootstrap",
    }


def _corruption_mode_deltas(scenarios: dict, real: dict, fixture: dict) -> list[dict[str, Any]]:
    rows = []
    modes = sorted({row["corruption_mode"] for row in scenarios.values() if row["reliability_label"] == "unreliable"})
    for model_index, model in enumerate(MODELS):
        for policy_index, policy in enumerate(POLICIES):
            for mode_index, mode in enumerate(modes):
                scenario_ids = sorted(
                    scenario_id for scenario_id, row in scenarios.items()
                    if row["reliability_label"] == "unreliable" and row["corruption_mode"] == mode
                )
                real_values = [
                    int(accepted_initial_output(real[(sid, model, policy)]))
                    for sid in scenario_ids
                ]
                fixture_values = [
                    int(
                        accepted_initial_output(
                            fixture[(sid, model, policy)]
                        )
                    )
                    for sid in scenario_ids
                ]
                rng = random.Random(SEED + 10_000 + model_index * 100 + policy_index * 10 + mode_index)
                deltas = []
                for _ in range(N_BOOT):
                    indices = [rng.randrange(len(scenario_ids)) for _ in scenario_ids]
                    deltas.append(
                        _mean([fixture_values[i] for i in indices]) - _mean([real_values[i] for i in indices])
                    )
                rows.append(
                    {
                        "model_name": model,
                        "policy_name": policy,
                        "corruption_mode": mode,
                        "n_scenarios": len(scenario_ids),
                        "real": _count_payload(real_values),
                        "fixture": _count_payload(fixture_values),
                        "delta_fixture_minus_real": _mean(fixture_values) - _mean(real_values),
                        "ci_low": _quantile(deltas, 0.025),
                        "ci_high": _quantile(deltas, 0.975),
                    }
                )
    return rows


def _cell_metrics(real: dict, fixture: dict) -> dict[str, Any]:
    result: dict[str, Any] = {"real_url": {}, "fixture_url": {}}
    for arm, index in [("real_url", real), ("fixture_url", fixture)]:
        for model in MODELS:
            for policy in POLICIES:
                rows = [row for (sid, m, p), row in index.items() if m == model and p == policy]
                result[arm][f"{model}|{policy}"] = {
                    metric: _count_payload([score(row) for row in rows if eligible(row)])
                    for metric, eligible, score in _row_metric_specs()
                }
    return result


def _changed_action_examples(scenarios: dict, real: dict, fixture: dict) -> dict[str, Any]:
    rows = []
    for key in sorted(real):
        if real[key]["agent_action"] == fixture[key]["agent_action"]:
            continue
        scenario_id, model, policy = key
        scenario = scenarios[scenario_id]
        rows.append(
            {
                "scenario_id": scenario_id,
                "model_name": model,
                "policy_name": policy,
                "reliability_label": scenario["reliability_label"],
                "corruption_mode": scenario["corruption_mode"],
                "real_url": scenario["clean_output"].get("url"),
                "fixture_url": scenario["tool_output"].get("url"),
                "visible_answer": scenario["tool_output"].get("answer"),
                "real_action": real[key]["agent_action"],
                "fixture_action": fixture[key]["agent_action"],
                "real_rationale": real[key].get("reason", ""),
                "fixture_rationale": fixture[key].get("reason", ""),
            }
        )
    return {"n_changed_actions": len(rows), "cases": rows}


def _verdict(paired: list[dict[str, Any]]) -> dict[str, Any]:
    clear = []
    for row in paired:
        for metric, value in row["metrics"].items():
            delta = value["delta_fixture_minus_real"]
            excludes_zero = value["ci_low"] > 0 or value["ci_high"] < 0
            if abs(delta) >= 0.05 and excludes_zero:
                clear.append({"model_name": row["model_name"], "policy_name": row["policy_name"], "metric": metric, **value})
    return {
        "status": "complete",
        "pre_registered_clear_effect_rule": "absolute delta >= 0.05 and 95% interval excludes zero",
        "clear_effect_cells": clear,
        "n_clear_effect_cells": len(clear),
        "interpretation_requires_joint_review": True,
    }


def _metric_specs():
    return [
        ("silent_acceptance", lambda s: s["reliability_label"] == "unreliable", lambda r: int(accepted_initial_output(r))),
        ("false_defer_or_verify", lambda s: s["reliability_label"] == "reliable", lambda r: int(not accepted_initial_output(r))),
        ("trust_accuracy", lambda s: True, lambda r: int((r["reliability_label"] == "reliable" and accepted_initial_output(r)) or (r["reliability_label"] == "unreliable" and not accepted_initial_output(r)))),
    ]


def _row_metric_specs():
    return [
        ("silent_acceptance", lambda r: r["reliability_label"] == "unreliable", lambda r: int(accepted_initial_output(r))),
        ("false_defer_or_verify", lambda r: r["reliability_label"] == "reliable", lambda r: int(not accepted_initial_output(r))),
        ("trust_accuracy", lambda r: True, lambda r: int((r["reliability_label"] == "reliable" and accepted_initial_output(r)) or (r["reliability_label"] == "unreliable" and not accepted_initial_output(r)))),
    ]


def _count_payload(values: list[int]) -> dict[str, Any]:
    return {"numerator": sum(values), "denominator": len(values), "rate": _mean(values)}


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * probability)
    return ordered[index]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _render_report(paired: list[dict], by_mode: list[dict], examples: dict, verdict: dict) -> str:
    lines = [
        "# Controlled URL-Fixture Ablation Results",
        "",
        "All deltas are fixture URL minus real URL on the same scenarios.",
        "",
        "| Model | Policy | Metric | Real | Fixture | Delta | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in paired:
        for metric, value in row["metrics"].items():
            lines.append(
                f"| {row['model_name']} | {row['policy_name']} | {metric} | "
                f"{value['real']['numerator']}/{value['real']['denominator']} ({value['real']['rate']:.3f}) | "
                f"{value['fixture']['numerator']}/{value['fixture']['denominator']} ({value['fixture']['rate']:.3f}) | "
                f"{value['delta_fixture_minus_real']:+.3f} | [{value['ci_low']:.3f}, {value['ci_high']:.3f}] |"
            )
    lines.extend(
        [
            "",
            f"Changed-action pairs available for review: {examples['n_changed_actions']}.",
            f"Pre-registered clear-effect cells: {verdict['n_clear_effect_cells']}.",
            "",
            "Corruption-mode estimates are stored in `corruption_mode_deltas.json` and must be interpreted with their small denominators.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
