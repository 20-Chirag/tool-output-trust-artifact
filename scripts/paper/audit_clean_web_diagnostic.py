from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trustbench.data.io import read_jsonl
from trustbench.evaluation.decision_metrics import accepted_initial_output


ROOT = Path(__file__).resolve().parents[2]
RUNS_PATH = ROOT / "outputs/real_web_domain_expansion_clean_run/runs.jsonl"
CACHE_PATH = ROOT / "outputs/real_web_domain_expansion_clean_run/cache.jsonl"
SCENARIOS_PATH = ROOT / "outputs/real_web_domain_expansion/scenarios_candidate.jsonl"
EXAMPLES_PATH = ROOT / "outputs/real_web_domain_expansion_clean_run/examples.json"
OUT_DIR = ROOT / "outputs/clean_web_diagnostic"
PAPER_OUT = ROOT / "paper/clean_web_diagnostic_results.md"
N_BOOT = 10_000
SEED = 20260722
METRICS = ("silent_acceptance", "false_defer_or_verify", "trust_accuracy")
COMPARISON_POLICIES = ("m2_structured_skeptic", "m6_checklist")


def main() -> None:
    runs = read_jsonl(RUNS_PATH)
    scenarios = {row["scenario_id"]: row for row in read_jsonl(SCENARIOS_PATH)}
    cache = _latest_by_event(read_jsonl(CACHE_PATH))
    integrity = _validate(runs, scenarios)

    paired = _paired_policy_deltas(runs, scenarios)
    corruption = _corruption_mode_metrics(runs, scenarios)
    gpt_cases = _gpt_silent_acceptance_cases(runs, scenarios, cache)
    omissions = _omission_construct_audit(runs, scenarios, cache)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        OUT_DIR / "paired_policy_deltas.json",
        {
            "analysis": "paired policy deltas within the clean broader-web run",
            "bootstrap_resamples": N_BOOT,
            "seed": SEED,
            "resampling_unit": "raw source capture",
            "integrity": integrity,
            "comparisons": paired,
        },
    )
    _write_json(
        OUT_DIR / "corruption_mode_metrics.json",
        {
            "analysis": "clean broader-web silent acceptance by corruption mode",
            "bootstrap_resamples": N_BOOT,
            "seed": SEED,
            "interval_policy": (
                "Wilson 95% intervals are paper-facing for single proportions. "
                "Empirical source-capture bootstrap intervals are retained as "
                "diagnostic fields and may degenerate for all-zero cells."
            ),
            "rows": corruption,
        },
    )
    _write_json(OUT_DIR / "gpt_silent_acceptance_cases.json", gpt_cases)
    _write_json(OUT_DIR / "omission_construct_audit.json", omissions)
    PAPER_OUT.write_text(
        _render_report(integrity, paired, corruption, gpt_cases, omissions),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "paired_comparisons": len(paired),
                "corruption_cells": len(corruption),
                "gpt_m2_m6_silent_accepts": len(gpt_cases["cases"]),
                "partial_omission_silent_accepts": len(omissions["cases"]),
                "output_dir": str(OUT_DIR.relative_to(ROOT)),
                "paper_report": str(PAPER_OUT.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate(runs: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_models = {"gpt-5.4-mini", "gemini-2.5-flash", "gemini-3.1-pro-preview"}
    expected_policies = {
        "baseline",
        "m2_structured_skeptic",
        "m6_checklist",
        "m0_naive_m2",
        "m0_naive_m6",
    }
    keys = [(row["scenario_id"], row["model_name"], row["policy_name"]) for row in runs]
    if len(runs) != 1_500 or len(set(keys)) != 1_500:
        raise ValueError(f"Expected 1500 unique run cells, found {len(runs)} rows / {len(set(keys))} unique")
    if set(row["model_name"] for row in runs) != expected_models:
        raise ValueError("Unexpected model set in clean run")
    if set(row["policy_name"] for row in runs) != expected_policies:
        raise ValueError("Unexpected policy set in clean run")
    if set(row["scenario_id"] for row in runs) != set(scenarios):
        raise ValueError("Run/scenario IDs do not match")
    actions = Counter(row["agent_action"] for row in runs)
    invalid_actions = sorted(set(actions) - {"accept", "verify", "abstain", "escalate"})
    if invalid_actions:
        raise ValueError(f"Invalid actions: {invalid_actions}")
    captures = {row["metadata"]["raw_capture_id"] for row in scenarios.values()}
    if len(captures) != 50:
        raise ValueError(f"Expected 50 source captures, found {len(captures)}")
    if not EXAMPLES_PATH.exists():
        raise ValueError(f"Missing clean examples artifact: {EXAMPLES_PATH}")
    return {
        "parsed_rows": len(runs),
        "unique_model_policy_scenario_cells": len(set(keys)),
        "duplicate_parsed_cells": len(runs) - len(set(keys)),
        "invalid_actions": 0,
        "source_captures": len(captures),
        "clean_examples_path": str(EXAMPLES_PATH.relative_to(ROOT)),
        "clean_examples_last_modified": EXAMPLES_PATH.stat().st_mtime,
    }


def _paired_policy_deltas(
    runs: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    indexed = {
        (row["scenario_id"], row["model_name"], row["policy_name"]): row
        for row in runs
    }
    models = sorted({row["model_name"] for row in runs})
    capture_to_scenarios: dict[str, list[str]] = defaultdict(list)
    for scenario_id, scenario in scenarios.items():
        capture_to_scenarios[scenario["metadata"]["raw_capture_id"]].append(scenario_id)
    capture_ids = sorted(capture_to_scenarios)
    results = []
    for model_index, model in enumerate(models):
        for policy_index, policy in enumerate(COMPARISON_POLICIES):
            baseline_rows = [indexed[(scenario_id, model, "baseline")] for scenario_id in scenarios]
            policy_rows = [indexed[(scenario_id, model, policy)] for scenario_id in scenarios]
            entry = {
                "model_name": model,
                "reference_policy": "baseline",
                "comparison_policy": policy,
                "n_scenarios": len(scenarios),
                "n_capture_clusters": len(capture_ids),
                "metrics": {},
            }
            for metric_index, metric in enumerate(METRICS):
                rng = random.Random(SEED + model_index * 100 + policy_index * 10 + metric_index)
                entry["metrics"][metric] = _paired_cluster_delta(
                    baseline_rows,
                    policy_rows,
                    scenarios,
                    capture_to_scenarios,
                    capture_ids,
                    metric,
                    rng,
                )
            results.append(entry)
    return results


def _paired_cluster_delta(
    baseline_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    capture_to_scenarios: dict[str, list[str]],
    capture_ids: list[str],
    metric: str,
    rng: random.Random,
) -> dict[str, Any]:
    baseline_by_scenario = {row["scenario_id"]: row for row in baseline_rows}
    policy_by_scenario = {row["scenario_id"]: row for row in policy_rows}
    baseline_values = _eligible_values(baseline_rows, metric)
    policy_values = _eligible_values(policy_rows, metric)
    baseline_point = _mean(baseline_values)
    policy_point = _mean(policy_values)
    samples = []
    for _ in range(N_BOOT):
        sampled_baseline: list[dict[str, Any]] = []
        sampled_policy: list[dict[str, Any]] = []
        for _ in capture_ids:
            capture_id = rng.choice(capture_ids)
            for scenario_id in capture_to_scenarios[capture_id]:
                sampled_baseline.append(baseline_by_scenario[scenario_id])
                sampled_policy.append(policy_by_scenario[scenario_id])
        samples.append(_metric(sampled_policy, metric) - _metric(sampled_baseline, metric))
    return {
        "delta_comparison_minus_baseline": policy_point - baseline_point,
        "ci_low": _quantile(samples, 0.025),
        "ci_high": _quantile(samples, 0.975),
        "baseline": _count_payload(baseline_values),
        "comparison": _count_payload(policy_values),
        "paired_capture_clusters": len(capture_ids),
        "ci_method": "paired source-capture percentile bootstrap",
    }


def _corruption_mode_metrics(
    runs: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    cells = sorted({(row["model_name"], row["policy_name"]) for row in runs})
    for cell_index, (model, policy) in enumerate(cells):
        selected = [
            row
            for row in runs
            if row["model_name"] == model
            and row["policy_name"] == policy
            and row["reliability_label"] == "unreliable"
        ]
        for mode_index, mode in enumerate(sorted({row["corruption_mode"] for row in selected})):
            mode_rows = [row for row in selected if row["corruption_mode"] == mode]
            values = [
                int(accepted_initial_output(row)) for row in mode_rows
            ]
            capture_ids = [scenarios[row["scenario_id"]]["metadata"]["raw_capture_id"] for row in mode_rows]
            if len(capture_ids) != len(set(capture_ids)):
                raise ValueError(f"Duplicate capture within corruption cell: {model}/{policy}/{mode}")
            rng = random.Random(SEED + 10_000 + cell_index * 10 + mode_index)
            samples = [_mean([rng.choice(values) for _ in values]) for _ in range(N_BOOT)]
            ci_low, ci_high = _wilson(int(sum(values)), len(values))
            rows.append(
                {
                    "model_name": model,
                    "policy_name": policy,
                    "corruption_mode": mode,
                    "silent_acceptance": {
                        **_count_payload(values),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "ci_method": "Wilson 95% interval",
                        "capture_bootstrap_ci_low": _quantile(samples, 0.025),
                        "capture_bootstrap_ci_high": _quantile(samples, 0.975),
                        "capture_bootstrap_note": (
                            "Empirical percentile bootstrap over the observed "
                            "capture rows; retained for audit, not paper-facing."
                        ),
                        "capture_clusters": len(capture_ids),
                    },
                }
            )
    return rows


def _gpt_silent_acceptance_cases(
    runs: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in runs
        if row["model_name"] == "gpt-5.4-mini"
        and row["policy_name"] in COMPARISON_POLICIES
        and row["reliability_label"] == "unreliable"
        and accepted_initial_output(row)
    ]
    cases = [_case_payload(row, scenarios[row["scenario_id"]], cache.get(row["event_id"])) for row in selected]
    overlap: dict[str, list[str]] = defaultdict(list)
    for row in cases:
        overlap[row["scenario_id"]].append(row["policy_name"])
    return {
        "analysis": "GPT mini silent accepts under M2 and M6 on the clean broader-web set",
        "status": "candidate mechanisms require human adjudication",
        "n_cases": len(cases),
        "counts_by_policy": dict(sorted(Counter(row["policy_name"] for row in cases).items())),
        "counts_by_corruption_mode": dict(
            sorted(Counter(row["corruption_mode"] for row in cases).items())
        ),
        "scenario_overlap": [
            {"scenario_id": scenario_id, "policies": sorted(policies)}
            for scenario_id, policies in sorted(overlap.items())
            if len(policies) > 1
        ],
        "cases": sorted(cases, key=lambda row: (row["scenario_id"], row["policy_name"])),
    }


def _omission_construct_audit(
    runs: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in runs
        if row["reliability_label"] == "unreliable"
        and row["corruption_mode"] == "partial_field_omission"
        and accepted_initial_output(row)
    ]
    cases = []
    for row in selected:
        scenario = scenarios[row["scenario_id"]]
        payload = _case_payload(row, scenario, cache.get(row["event_id"]))
        visible_text = json.dumps(scenario["tool_output"], sort_keys=True)
        clean = str(scenario["clean_output"].get("answer", ""))
        final = str(row.get("final_answer", ""))
        clean_in_visible = _contains_normalized(visible_text, clean)
        clean_in_final = _contains_normalized(final, clean)
        if clean_in_final and not clean_in_visible:
            heuristic = "final_contains_clean_answer_not_present_in_visible_output"
        elif not clean_in_visible:
            heuristic = "accepted_omission_without_visible_clean_answer"
        else:
            heuristic = "clean_answer_visible_check_required"
        payload.update(
            {
                "clean_answer_in_visible_output": clean_in_visible,
                "clean_answer_in_final_answer": clean_in_final,
                "heuristic_construct_flag": heuristic,
                "human_label": "",
                "human_notes": "",
            }
        )
        cases.append(payload)
    return {
        "analysis": "partial-field-omission silent accepts in the clean broader-web run",
        "status": "heuristic flags only; human adjudication required",
        "n_cases": len(cases),
        "counts_by_heuristic_flag": dict(
            sorted(Counter(row["heuristic_construct_flag"] for row in cases).items())
        ),
        "cases": sorted(
            cases,
            key=lambda row: (row["scenario_id"], row["model_name"], row["policy_name"]),
        ),
    }


def _case_payload(
    row: dict[str, Any], scenario: dict[str, Any], cache_row: dict[str, Any] | None
) -> dict[str, Any]:
    raw = _parse_cached_text(cache_row)
    rationale = str(row.get("reason") or raw.get("rationale") or "")
    rationale_norm = rationale.lower()
    authority_terms = ("official", "source", "cited", "page", "matches the query", "direct result")
    check_terms = ("calculated", "cross-check", "contradict", "compared", "independent")
    if any(term in rationale_norm for term in authority_terms) and not any(
        term in rationale_norm for term in check_terms
    ):
        candidate_mechanism = "source_authority_without_explicit_content_check"
    else:
        candidate_mechanism = "unclear_requires_human_review"
    return {
        "scenario_id": row["scenario_id"],
        "model_name": row["model_name"],
        "policy_name": row["policy_name"],
        "corruption_mode": row["corruption_mode"],
        "user_task": scenario["user_task"],
        "visible_answer": scenario["tool_output"].get("answer"),
        "visible_title": scenario["tool_output"].get("title"),
        "visible_url": scenario["tool_output"].get("url"),
        "visible_snippet": scenario["tool_output"].get("snippet"),
        "clean_answer": scenario["clean_output"].get("answer"),
        "agent_action": row["agent_action"],
        "trust_score": row.get("trust_score"),
        "accepted_initial_output": row.get("accepted_initial_output"),
        "observed_concerns": row.get("observed_concerns", raw.get("observed_concerns", [])),
        "rationale": rationale,
        "final_answer": row.get("final_answer"),
        "heuristic_candidate_mechanism": candidate_mechanism,
        "human_mechanism_label": "",
        "human_notes": "",
    }


def _latest_by_event(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["event_id"]: row for row in rows}


def _parse_cached_text(cache_row: dict[str, Any] | None) -> dict[str, Any]:
    if not cache_row:
        return {}
    try:
        return json.loads(str(cache_row.get("text", "")))
    except json.JSONDecodeError:
        return {}


def _metric(rows: list[dict[str, Any]], metric: str) -> float:
    return _mean(_eligible_values(rows, metric))


def _eligible_values(rows: list[dict[str, Any]], metric: str) -> list[int]:
    if metric == "silent_acceptance":
        return [
            int(accepted_initial_output(row))
            for row in rows
            if row["reliability_label"] == "unreliable"
        ]
    if metric == "false_defer_or_verify":
        return [
            int(not accepted_initial_output(row))
            for row in rows
            if row["reliability_label"] == "reliable"
        ]
    if metric == "trust_accuracy":
        return [
            int(
                (
                    row["reliability_label"] == "reliable"
                    and accepted_initial_output(row)
                )
                or (
                    row["reliability_label"] == "unreliable"
                    and not accepted_initial_output(row)
                )
            )
            for row in rows
        ]
    raise ValueError(f"Unsupported metric: {metric}")


def _count_payload(values: list[int]) -> dict[str, Any]:
    return {
        "numerator": int(sum(values)),
        "denominator": len(values),
        "estimate": _mean(values),
    }


def _contains_normalized(text: str, value: str) -> bool:
    text_norm = _normalize(text)
    value_norm = _normalize(value)
    if not text_norm or not value_norm:
        return False
    return re.search(rf"(^| )({re.escape(value_norm)})( |$)", text_norm) is not None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _mean(values: list[int] | list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean over an empty selection")
    return sum(values) / len(values)


def _wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(
        (p * (1 - p) + z * z / (4 * total)) / total
    ) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _render_report(
    integrity: dict[str, Any],
    paired: list[dict[str, Any]],
    corruption: list[dict[str, Any]],
    gpt_cases: dict[str, Any],
    omissions: dict[str, Any],
) -> str:
    lines = [
        "# Clean-Web Diagnostic Results",
        "",
        "**Generated:** 2026-08-22  ",
        "**Status:** deterministic no-API audit; qualitative mechanisms remain provisional",
        "",
        "## Integrity",
        "",
        f"- Parsed rows: {integrity['parsed_rows']}",
        f"- Unique model/policy/scenario cells: {integrity['unique_model_policy_scenario_cells']}",
        f"- Duplicate parsed cells: {integrity['duplicate_parsed_cells']}",
        f"- Invalid actions: {integrity['invalid_actions']}",
        f"- Source-capture clusters: {integrity['source_captures']}",
        f"- Current examples source: `{integrity['clean_examples_path']}`",
        "",
        "## Paired Policy Deltas Within The Clean Dataset",
        "",
        "Deltas are comparison policy minus clean baseline. Negative silent acceptance and false defer/verify are improvements; positive trust accuracy is an improvement.",
        "",
        "| Model | Policy | Metric | Baseline | Policy | Delta | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for comparison in paired:
        for metric in METRICS:
            data = comparison["metrics"][metric]
            lines.append(
                "| {model} | {policy} | {metric} | {base} ({bk}/{bn}) | {comp} ({ck}/{cn}) | {delta} | [{lo}, {hi}] |".format(
                    model=comparison["model_name"],
                    policy=comparison["comparison_policy"],
                    metric=metric,
                    base=_fmt(data["baseline"]["estimate"]),
                    bk=data["baseline"]["numerator"],
                    bn=data["baseline"]["denominator"],
                    comp=_fmt(data["comparison"]["estimate"]),
                    ck=data["comparison"]["numerator"],
                    cn=data["comparison"]["denominator"],
                    delta=_fmt(data["delta_comparison_minus_baseline"]),
                    lo=_fmt(data["ci_low"]),
                    hi=_fmt(data["ci_high"]),
                )
            )
    lines.extend(
        [
            "",
            "## Baseline Corruption-Type Sensitivity",
            "",
            "Single-proportion uncertainty uses Wilson 95% intervals. The original empirical capture-bootstrap bounds remain in the JSON for audit but are not used as paper-facing intervals.",
            "",
            "| Model | Corruption mode | Silent acceptance | Wilson 95% CI | n |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in corruption:
        if row["policy_name"] != "baseline":
            continue
        metric = row["silent_acceptance"]
        lines.append(
            f"| {row['model_name']} | {row['corruption_mode']} | {_fmt(metric['estimate'])} | [{_fmt(metric['ci_low'])}, {_fmt(metric['ci_high'])}] | {metric['denominator']} |"
        )
    lines.extend(
        [
            "",
            "## GPT M2/M6 Silent Accepts",
            "",
            f"- Cases: {gpt_cases['n_cases']}",
            f"- By policy: `{json.dumps(gpt_cases['counts_by_policy'], sort_keys=True)}`",
            f"- By corruption mode: `{json.dumps(gpt_cases['counts_by_corruption_mode'], sort_keys=True)}`",
            f"- Shared scenarios: {len(gpt_cases['scenario_overlap'])}",
            "",
            "The rationales repeatedly cite source legitimacy or direct task/source match without an explicit value-level check. This is a candidate source-authority-substitution mechanism, not a confirmed qualitative finding until human adjudication.",
            "",
            "| Scenario | Policy | Visible | Clean | Trust | Rationale |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for case in gpt_cases["cases"]:
        rationale = str(case["rationale"]).replace("|", "/")
        lines.append(
            f"| {case['scenario_id']} | {case['policy_name']} | {case['visible_answer']} | {case['clean_answer']} | {_fmt(float(case['trust_score']))} | {rationale} |"
        )
    lines.extend(
        [
            "",
            "## Partial-Omission Construct Audit",
            "",
            f"- Silent-accept cases: {omissions['n_cases']}",
            f"- Heuristic flags: `{json.dumps(omissions['counts_by_heuristic_flag'], sort_keys=True)}`",
            "",
            "These flags only identify cases for human review. A correct final answer that was absent from the visible output may indicate parametric-memory substitution rather than trust in the tool output.",
            "",
            "## Interpretation Lock",
            "",
            "- The old fixture-to-clean comparison must report both false-deferral decreases and silent-acceptance increases.",
            "- The unpaired comparison is not a pure causal URL effect.",
            "- Corruption-type sensitivity is the primary clean-web content result.",
            "- Source-authority and parametric-memory mechanisms remain provisional until human coding.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
