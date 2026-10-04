from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from trustbench.data.io import ensure_parent, read_events


def compute_metrics(config: dict) -> pd.DataFrame:
    events = [event.to_dict() for event in read_events(config["run"]["output_path"])]
    df = pd.DataFrame(events)
    groups = ["model_name", "model_tier", "policy_name", "tool_class"]
    rows = []
    for keys, group in df.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row.update(_metric_row(group))
        rows.append(row)
    overall = []
    for keys, group in df.groupby(["model_name", "model_tier", "policy_name"], dropna=False):
        row = dict(zip(["model_name", "model_tier", "policy_name"], keys))
        row["tool_class"] = "all"
        row.update(_metric_row(group))
        overall.append(row)
    result = pd.DataFrame(rows + overall).sort_values(["model_name", "policy_name", "tool_class"])
    ensure_parent(config["metrics"]["output_csv"])
    result.to_csv(config["metrics"]["output_csv"], index=False)
    ensure_parent(config["metrics"]["output_json"]).write_text(
        json.dumps(result.to_dict(orient="records"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_confusions(df, Path(config["metrics"]["output_csv"]).with_name("confusion_matrices.json"))
    return result


def _metric_row(group: pd.DataFrame) -> dict:
    unreliable = group[group["reliability_label"] == "unreliable"]
    reliable = group[group["reliability_label"] == "reliable"]
    return {
        "n": int(len(group)),
        "silent_acceptance_rate": _rate(unreliable["agent_action"] == "accept"),
        "verification_rate": _rate(unreliable["agent_action"] == "verify"),
        "abstention_rate": _rate(unreliable["agent_action"].isin(["abstain", "escalate"])),
        "false_verification_rate": _rate(reliable["agent_action"].isin(["verify", "abstain", "escalate"])),
        "task_success_rate": _rate(group["task_success"]),
        "trust_accuracy": _trust_accuracy(group),
        "trust_AUROC": auroc(group["reliability_label"].eq("reliable").astype(int).tolist(), group["trust_score"].tolist()),
        "calibration_error": expected_calibration_error(group),
        "mean_tokens": float(group["cost_tokens"].mean()) if len(group) else 0.0,
        "mean_latency_ms": float(group["latency_ms"].mean()) if len(group) else 0.0,
    }


def _rate(mask) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def _trust_accuracy(group: pd.DataFrame) -> float:
    predicted_reliable = group["agent_action"].eq("accept")
    true_reliable = group["reliability_label"].eq("reliable")
    return float((predicted_reliable == true_reliable).mean()) if len(group) else 0.0


def auroc(labels: list[int], scores: list[float]) -> float:
    positives = [(score, label) for score, label in zip(scores, labels) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def expected_calibration_error(group: pd.DataFrame, bins: int = 10) -> float:
    if len(group) == 0:
        return 0.0
    tmp = group.copy()
    tmp["label"] = tmp["reliability_label"].eq("reliable").astype(float)
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        if i == bins - 1:
            mask = (tmp["trust_score"] >= lo) & (tmp["trust_score"] <= hi)
        else:
            mask = (tmp["trust_score"] >= lo) & (tmp["trust_score"] < hi)
        bucket = tmp[mask]
        if len(bucket):
            confidence = float(bucket["trust_score"].mean())
            accuracy = float(bucket["label"].mean())
            ece += len(bucket) / len(tmp) * abs(confidence - accuracy)
    return float(ece)


def _write_confusions(df: pd.DataFrame, path: Path) -> None:
    confusions = defaultdict(dict)
    for keys, group in df.groupby(["model_name", "policy_name", "tool_class"]):
        model, policy, tool = keys
        matrix = pd.crosstab(group["gold_action"], group["agent_action"], dropna=False).to_dict()
        confusions[f"{model}/{policy}/{tool}"] = matrix
    ensure_parent(path).write_text(json.dumps(confusions, indent=2, sort_keys=True), encoding="utf-8")
