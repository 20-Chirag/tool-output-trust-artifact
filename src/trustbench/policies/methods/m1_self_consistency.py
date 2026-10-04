"""M1 — Self-consistency over the baseline prompt.

Sample N actions from the baseline prompt at non-zero temperature, then
take the majority action. The agreement fraction acts as an auxiliary
trust score: when all N samples agree the score is 1.0; when they
disagree the score is the fraction agreeing with the chosen action.

Costs API calls (N x baseline cost). Reuses build_prompt(policy="baseline")
from the existing real_runner.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from trustbench.data.schemas import TrustAction


@dataclass(frozen=True)
class SelfConsistencyConfig:
    n_samples: int = 3
    temperature: float = 0.7


def aggregate(parsed_samples: list[dict]) -> dict:
    """Vote across parsed responses. Each parsed dict must contain
    `action` (TrustAction) and `trust_score`.

    Returns a single parsed-event-like dict with the majority action,
    auxiliary trust score = agreement fraction, and average of the
    samples' own trust scores under the auxiliary key.
    """
    if not parsed_samples:
        raise ValueError("aggregate requires >=1 sample")
    actions = [p["action"] for p in parsed_samples]
    counts = Counter(a.value if isinstance(a, TrustAction) else str(a) for a in actions)
    majority_value, majority_count = counts.most_common(1)[0]
    agreement = majority_count / len(parsed_samples)
    avg_inner_trust = sum(float(p.get("trust_score", 0.0)) for p in parsed_samples) / len(parsed_samples)
    representative = next(p for p in parsed_samples if (p["action"].value if isinstance(p["action"], TrustAction) else str(p["action"])) == majority_value)
    out = dict(representative)
    out["action"] = TrustAction(majority_value)
    out["trust_score"] = agreement  # agreement = trust signal
    out["m1_inner_avg_trust"] = avg_inner_trust
    out["m1_action_counts"] = dict(counts)
    return out
