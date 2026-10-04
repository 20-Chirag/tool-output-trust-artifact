"""Temperature scaling — post-hoc recalibration of trust scores.

Fits a single scalar temperature T on a held-out validation split so that
sigmoid(trust_score / T) is better aligned with empirical reliability.
Then transforms trust scores for the test split and recomputes actions
using the same M4 threshold rule.

This is the standard post-hoc calibration baseline from Guo et al. 2017
(ICML), applied to verbalized trust scores instead of logits.

Costs ₹0 — operates entirely on existing events.

Usage:
    cal = TempScaling.fit(val_events)
    test_events_rescaled = cal.apply_many(test_events)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from trustbench.data.schemas import TrustAction


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    n = len(scores)
    for k in range(n_bins):
        in_bin = (scores >= bins[k]) & (scores < bins[k + 1] if k < n_bins - 1 else scores <= bins[k + 1])
        if in_bin.sum() == 0:
            continue
        conf = scores[in_bin].mean()
        acc = labels[in_bin].mean()
        err += (in_bin.sum() / n) * abs(conf - acc)
    return float(err)


@dataclass
class TempScaling:
    temperature: float

    @classmethod
    def fit(cls, events: list[dict], n_grid: int = 200) -> "TempScaling":
        """Grid search for T in [0.1, 10] that minimises ECE on `events`."""
        scores = np.array([float(e["trust_score"]) for e in events])
        labels = np.array([1.0 if e["reliability_label"] == "reliable" else 0.0 for e in events])
        best_t, best_ece = 1.0, float("inf")
        for t in np.linspace(0.1, 10.0, n_grid):
            scaled = np.array([_sigmoid((s - 0.5) / t + 0.5) for s in scores])
            e = _ece(scaled, labels)
            if e < best_ece:
                best_ece = e
                best_t = float(t)
        return cls(temperature=best_t)

    def scale(self, trust_score: float) -> float:
        return float(_sigmoid((trust_score - 0.5) / self.temperature + 0.5))

    def apply(self, event: dict, verify_below: float = 0.55, abstain_below: float = 0.25) -> dict:
        out = dict(event)
        new_trust = self.scale(float(event["trust_score"]))
        if new_trust < abstain_below:
            new_action = TrustAction.ABSTAIN.value
        elif new_trust < verify_below:
            new_action = TrustAction.VERIFY.value
        else:
            new_action = event["agent_action"]
        out["trust_score"] = new_trust
        out["agent_action"] = new_action
        out["policy_name"] = "temperature_scaling"
        out["temp_scaling_T"] = self.temperature
        return out

    def apply_many(self, events: Iterable[dict], verify_below: float = 0.55, abstain_below: float = 0.25) -> list[dict]:
        return [self.apply(e, verify_below, abstain_below) for e in events]
