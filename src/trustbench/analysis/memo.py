from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trustbench.data.io import ensure_parent


def write_decision_memo(config: dict) -> str:
    metrics = pd.read_csv(config["metrics"]["output_csv"])
    overall = metrics[metrics["tool_class"] == "all"]
    baseline = overall[overall["policy_name"] == "baseline"]
    skeptical = overall[overall["policy_name"] == "skeptical"]
    calibration_policy = "heuristic_trust" if "heuristic_trust" in set(overall["policy_name"]) else "trust_calibrated"
    calibrated = overall[overall["policy_name"] == calibration_policy]

    baseline_silent = float(baseline["silent_acceptance_rate"].mean())
    skeptical_silent = float(skeptical["silent_acceptance_rate"].mean())
    calibrated_silent = float(calibrated["silent_acceptance_rate"].mean())
    calibrated_success = float(calibrated["task_success_rate"].mean())
    baseline_success = float(baseline["task_success_rate"].mean())
    mean_auroc = float(overall[overall["policy_name"].isin(["baseline", "skeptical", "heuristic_trust", "learned_trust", "trust_calibrated"])]["trust_AUROC"].mean())
    variation = float(metrics[(metrics["tool_class"] != "all") & (metrics["policy_name"] == "baseline")]["silent_acceptance_rate"].std())

    skeptical_gain = baseline_silent - skeptical_silent
    calibrated_gain = baseline_silent - calibrated_silent
    success_collapse = baseline_success - calibrated_success > 0.20
    signal = (
        baseline_silent > 0.15 and variation > 0.03
        or skeptical_gain > 0.04
        or (calibrated_gain > 0.08 and not success_collapse)
        or mean_auroc > 0.60
    )
    if signal and calibrated_gain > 0.03:
        recommendation = "scale"
    elif signal:
        recommendation = "revise"
    else:
        recommendation = "stop"

    rows = overall[
        [
            "model_name",
            "policy_name",
            "silent_acceptance_rate",
            "verification_rate",
            "abstention_rate",
            "false_verification_rate",
            "task_success_rate",
            "trust_AUROC",
        ]
    ].round(3)
    table = _markdown_table(rows)
    cost_path = Path(config["cost"]["output_path"])
    cost_text = ""
    if cost_path.exists():
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        if "projected_full_events" in cost:
            cost_text = (
                f"\nProjected full run: {cost['projected_full_events']} events, "
                f"{cost['projected_full_tokens']} tokens, "
                f"${cost['projected_full_api_cost']:.4f} simulated API cost, "
                f"{cost['projected_annotation_hours']:.1f} annotation hours.\n"
            )
        else:
            cost_text = (
                f"\nCompleted real run cache total: {cost.get('input_tokens', 0)} input tokens, "
                f"{cost.get('output_tokens', 0)} output tokens, "
                f"${float(cost.get('spent_usd', 0)):.4f} API cost recorded by the run summary.\n"
            )

    memo = f"""# Pilot Decision Memo

Recommendation: **{recommendation}**

## Decision Signals

- Mean baseline silent acceptance on unreliable outputs: {baseline_silent:.3f}
- Baseline variation across tool classes: {variation:.3f}
- Skeptical prompting reduction in silent acceptance: {skeptical_gain:.3f}
- {calibration_policy} reduction in silent acceptance: {calibrated_gain:.3f}
- Mean non-oracle trust AUROC: {mean_auroc:.3f}
- {calibration_policy} task success drop vs baseline: {baseline_success - calibrated_success:.3f}
{cost_text}
## Metrics by Model and Policy

{table}

## Interpretation

The pilot is designed to falsify the project before scale-up. A `scale` recommendation means the benchmark produces non-trivial silent acceptance, separable reliability signals, and a trust-calibration gain that does not collapse task success. A `revise` recommendation means the task has some signal but the corruptions, policy thresholds, or model set should be adjusted before full collection. A `stop` recommendation means the current benchmark does not justify scaling.
"""
    ensure_parent(config["analysis"]["memo_path"]).write_text(memo, encoding="utf-8")
    return recommendation


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in frame.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, divider, *body])
