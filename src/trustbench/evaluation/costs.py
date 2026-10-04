from __future__ import annotations

import json
from pathlib import Path

from trustbench.data.io import ensure_parent, read_events


def estimate_costs(config: dict) -> dict:
    events = read_events(config["run"]["output_path"])
    prices = config.get("cost", {}).get("model_prices_per_1k_tokens", {})
    total_tokens = sum(event.cost_tokens for event in events)
    by_model: dict[str, dict] = {}
    for event in events:
        bucket = by_model.setdefault(event.model_name, {"tokens": 0, "events": 0, "pilot_cost": 0.0})
        bucket["tokens"] += event.cost_tokens
        bucket["events"] += 1
    for model, bucket in by_model.items():
        bucket["pilot_cost"] = bucket["tokens"] / 1000 * float(prices.get(model, 0.0))

    pilot_events = max(1, len(events))
    full_scenarios = int(config.get("cost", {}).get("full_scenarios", 600))
    full_events = full_scenarios * len(config["models"]) * len(config["policies"])
    scale = full_events / pilot_events
    annotation_minutes = full_scenarios * float(config.get("cost", {}).get("annotation_minutes_per_scenario", 3.5))
    estimate = {
        "pilot_events": pilot_events,
        "pilot_total_tokens": total_tokens,
        "pilot_estimated_api_cost": sum(bucket["pilot_cost"] for bucket in by_model.values()),
        "by_model": by_model,
        "projected_full_events": full_events,
        "projected_full_tokens": int(total_tokens * scale),
        "projected_full_api_cost": sum(bucket["pilot_cost"] for bucket in by_model.values()) * scale,
        "projected_annotation_hours": annotation_minutes / 60,
        "projected_runtime_minutes": sum(event.latency_ms for event in events) * scale / 60000,
    }
    path = ensure_parent(config["cost"]["output_path"])
    path.write_text(json.dumps(estimate, indent=2, sort_keys=True), encoding="utf-8")
    return estimate
