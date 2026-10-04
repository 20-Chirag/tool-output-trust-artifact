from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from scripts.phase_3 import run as phase3_run
from trustbench.data.io import read_jsonl, read_scenarios


SOURCE = Path("outputs/real_web_domain_expansion/scenarios_candidate.jsonl")
FIXTURE = Path("outputs/url_fixture_ablation/scenarios_fixture.jsonl")
MANIFEST = Path("outputs/url_fixture_ablation/build_manifest.json")
CONFIG = "configs/url_fixture_ablation.yaml"


def test_url_fixture_arm_is_balanced_and_host_replaced() -> None:
    rows = read_jsonl(FIXTURE)
    assert len(rows) == 100
    assert Counter(row["reliability_label"] for row in rows) == {"reliable": 50, "unreliable": 50}
    assert all(urlsplit(row["tool_output"]["url"]).netloc.lower() == "example.org" for row in rows)
    assert all("example.org" not in str(row["clean_output"].get("url", "")).lower() for row in rows)


def test_url_fixture_arm_changes_only_visible_url_host() -> None:
    source = read_jsonl(SOURCE)
    fixture = read_jsonl(FIXTURE)
    assert [row["scenario_id"] for row in source] == [row["scenario_id"] for row in fixture]
    for real_row, fixture_row in zip(source, fixture, strict=True):
        normalized = json.loads(json.dumps(fixture_row))
        normalized["tool_output"]["url"] = real_row["tool_output"]["url"]
        assert normalized == real_row, real_row["scenario_id"]
        real_url = urlsplit(real_row["tool_output"]["url"])
        fixture_url = urlsplit(fixture_row["tool_output"]["url"])
        assert (real_url.scheme, real_url.path, real_url.query, real_url.fragment) == (
            fixture_url.scheme,
            fixture_url.path,
            fixture_url.query,
            fixture_url.fragment,
        )


def test_url_fixture_prompts_differ_only_in_visible_url() -> None:
    real_scenarios = {scenario.scenario_id: scenario for scenario in read_scenarios(SOURCE)}
    fixture_scenarios = {scenario.scenario_id: scenario for scenario in read_scenarios(FIXTURE)}
    policies = ["baseline", "m2_structured_skeptic", "m6_checklist"]
    for scenario_id, real_scenario in real_scenarios.items():
        fixture_scenario = fixture_scenarios[scenario_id]
        for policy in policies:
            real_prompt = phase3_run.PROMPT_BUILDERS[policy](real_scenario)
            fixture_prompt = phase3_run.PROMPT_BUILDERS[policy](fixture_scenario)
            assert real_prompt["system"] == fixture_prompt["system"]
            real_user = json.loads(real_prompt["user"])
            fixture_user = json.loads(fixture_prompt["user"])
            fixture_user["tool_output"]["url"] = real_user["tool_output"]["url"]
            assert fixture_user == real_user, (scenario_id, policy)


def test_url_fixture_prompts_do_not_expose_hidden_fields() -> None:
    forbidden = {
        "clean_output",
        "gold_action",
        "reliability_label",
        "corruption_mode",
        "expected_answer",
        "synthetic_fault",
        "hidden_from_agent",
    }
    config = phase3_run.load_config(CONFIG)
    for scenario in phase3_run.filter_phase3_scenarios(config):
        for policy in config["policies"]:
            prompt = phase3_run.PROMPT_BUILDERS[policy](scenario)
            text = json.dumps(prompt, sort_keys=True).lower()
            assert not any(marker in text for marker in forbidden), (scenario.scenario_id, policy)


def test_url_fixture_config_and_dry_run_are_isolated_and_unapproved() -> None:
    config = phase3_run.load_config(CONFIG)
    summary = phase3_run.build_dry_run_summary(config)
    assert config["run"]["paid_run_authorized"] is False
    assert config["run"]["output_dir"] == "outputs/url_fixture_ablation_run"
    assert config["run"]["max_usd"] == 2.0
    assert summary["events"] == 600
    assert summary["scenarios"] == 100
    assert summary["models"] == ["gpt-5.4-mini", "gemini-2.5-flash"]
    assert summary["policies"] == ["baseline", "m2_structured_skeptic", "m6_checklist"]
    assert summary["paid_run_authorized"] is False
    assert summary["will_create_paid_outputs"] is False
    assert summary["fits_budget"] is True


def test_url_fixture_manifest_records_no_paid_calls() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["scenario_count"] == 100
    assert manifest["fixture_url_count"] == 100
    assert manifest["only_changed_model_visible_field"] == "tool_output.url"
    assert manifest["url_component_changed"] == "host only"
    assert manifest["paid_model_calls_made"] is False
