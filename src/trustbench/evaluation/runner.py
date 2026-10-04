from __future__ import annotations

import random

from trustbench.agents.backends import make_backends
from trustbench.data.io import read_scenarios, write_events
from trustbench.data.schemas import AgentEvent, Scenario
from trustbench.policies.base import PolicyContext
from trustbench.policies.registry import build_policies


def split_for_learning(scenarios: list[Scenario], fraction: float, seed: int) -> tuple[list[Scenario], list[Scenario]]:
    rng = random.Random(seed)
    shuffled = list(scenarios)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * fraction))
    return shuffled[:n_train], shuffled[n_train:]


def run_experiment(config: dict) -> list[AgentEvent]:
    scenarios = read_scenarios(config["dataset"]["output_path"])
    train_scenarios, _ = split_for_learning(
        scenarios,
        float(config.get("run", {}).get("learned_train_fraction", 0.25)),
        int(config.get("seed", 7)),
    )
    backends = make_backends(config["models"])
    policies = build_policies(config["policies"], train_scenarios)
    context = PolicyContext(
        seed=int(config.get("seed", 7)),
        enable_second_verification=bool(config.get("run", {}).get("enable_second_verification", True)),
    )
    events: list[AgentEvent] = []
    for backend in backends:
        for policy in policies:
            for scenario in scenarios:
                events.append(policy.run(scenario, backend, context))
    write_events(config["run"]["output_path"], events)
    return events
