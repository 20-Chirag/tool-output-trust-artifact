from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["scenario_id"]),
        str(row["visible_answer"]),
        str(row["clean_answer"]),
        str(row["rationale"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the evidence behind the paper's secondary calculator diagnostic."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    labels_path = (
        root
        / "outputs/adjudication/claude_results_2026_07_22/"
        "f1_calculator_claude_review.csv"
    )
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))

    phase2 = read_json(
        root / "outputs/phase_2_prompt_audit/m6_mental_math_audit.json"
    )
    phase_b = read_json(root / "outputs/phase_b/finding_robustness.json")
    phase3 = read_json(root / "outputs/phase_3_minimal/finding_robustness.json")
    adjudication = read_json(
        root / "outputs/adjudication/claude_results_2026_07_22/analysis.json"
    )

    assert len(labels) == 50
    source_counts = Counter(row["source_run"] for row in labels)
    assert source_counts == {
        "phase_2_prompt_audit": 24,
        "phase_b_m0_naive_m6": 26,
    }
    assert Counter(row["CLAUDE_verbal_compliance"] for row in labels) == {"yes": 50}
    assert Counter(row["CLAUDE_actual_arithmetic_evidence"] for row in labels) == {
        "no": 50
    }
    assert len({row["scenario_id"] for row in labels}) == 13

    phase2_cases = {case_key(row) for row in phase2["cases"]}
    phase_b_cases = {case_key(row) for row in phase_b["F1"]["cases"]}
    for row in labels:
        key = (
            row["scenario_id"],
            row["visible_answer"],
            row["clean_answer"],
            row["model_rationale"],
        )
        expected = (
            phase2_cases
            if row["source_run"] == "phase_2_prompt_audit"
            else phase_b_cases
        )
        assert key in expected, f"Label row is absent from its source cases: {row['case_id']}"

    f1_counts = adjudication["headline_counts"]
    assert f1_counts["f1_verbal_compliance"]["numerator"] == 50
    assert f1_counts["f1_verbal_compliance"]["denominator"] == 50
    assert f1_counts["f1_actual_arithmetic_evidence"]["numerator"] == 0
    assert f1_counts["f1_actual_arithmetic_evidence"]["denominator"] == 50
    assert phase3["F1"]["target_events"] == 26
    assert phase3["F1"]["accepted_target_events"] == 0

    print(
        json.dumps(
            {
                "status": "pass",
                "labeled_decisions": len(labels),
                "source_counts": dict(source_counts),
                "unique_scenarios": 13,
                "verbal_compliance_yes": 50,
                "auditable_arithmetic_yes": 0,
                "stronger_gemini_target_accepts": 0,
                "stronger_gemini_target_events": 26,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
