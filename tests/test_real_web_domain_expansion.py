from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from scripts.phase_3 import run as phase3_run
from trustbench.data.io import read_scenarios


SCENARIOS_PATH = Path("outputs/real_web_domain_expansion/scenarios_candidate.jsonl")
CAPTURES_PATH = Path("outputs/real_web_domain_expansion/raw/captures.jsonl")
DOMAIN_AUDIT_PATH = Path("outputs/real_web_domain_expansion/domain_audit.json")


def test_real_web_domain_expansion_artifacts_exist() -> None:
    assert SCENARIOS_PATH.exists()
    assert CAPTURES_PATH.exists()
    assert DOMAIN_AUDIT_PATH.exists()
    assert Path("outputs/real_web_domain_expansion/inspection_note.md").exists()


def test_real_web_domain_expansion_balance_and_tool_class() -> None:
    scenarios = read_scenarios(SCENARIOS_PATH)
    assert len(scenarios) == 100
    assert {scenario.tool_class.value for scenario in scenarios} == {"web_search"}
    labels = Counter(scenario.reliability_label.value for scenario in scenarios)
    assert labels == {"reliable": 50, "unreliable": 50}


def test_real_web_domain_expansion_domain_constraints() -> None:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    urls = [row["tool_output"]["url"] for row in rows]
    text = json.dumps(rows, sort_keys=True).lower()
    assert "example.org" not in text
    assert "example.com" not in text
    assert "example.net" not in text
    domains = Counter(urlparse(url).netloc.lower() for url in urls)
    assert domains["en.wikipedia.org"] <= 10
    assert max(domains.values()) <= 15
    assert len(domains) >= 25


def test_real_web_domain_expansion_capture_manifest_constraints() -> None:
    captures = [json.loads(line) for line in CAPTURES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(captures) == 50
    groups = Counter(row["domain_group"] for row in captures)
    assert groups == {
        "official_documentation": 10,
        "government_public_service": 10,
        "university": 10,
        "product_vendor_docs": 10,
        "transit_local_service": 10,
    }
    domains = Counter(row["source_domain"] for row in captures)
    assert domains["en.wikipedia.org"] == 0
    assert max(domains.values()) == 1


def test_real_web_domain_expansion_has_no_model_visible_timestamps() -> None:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    forbidden = {"captured_at", "retrieved_at", "timestamp", "capture_date", "retrieval_date"}
    for row in rows:
        visible_keys = {key.lower() for key in row["tool_output"].keys()}
        assert visible_keys.isdisjoint(forbidden)


def test_real_web_domain_expansion_has_no_model_visible_lineage_ids() -> None:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    forbidden_visible_keys = {"id", "capture_id", "scenario_id", "raw_capture_id"}
    forbidden_value_markers = {"domain-web-corrupt", "domain-web-omission", "domain-web-wrong-entity"}
    for row in rows:
        visible_keys = {key.lower() for key in row["tool_output"].keys()}
        assert visible_keys.isdisjoint(forbidden_visible_keys)
        visible_text = json.dumps(row["tool_output"], sort_keys=True).lower()
        assert not any(marker in visible_text for marker in forbidden_value_markers)


def test_real_web_domain_expansion_queries_do_not_leak_clean_answers() -> None:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    leaks = []
    for row in rows:
        answer = row["clean_output"]["answer"]
        visible_query = row["tool_invocation"]["arguments"]["query"]
        if _contains_answer(visible_query, answer):
            leaks.append((row["scenario_id"], "tool_invocation.query", visible_query, answer))
        if _contains_answer(row["user_task"], answer):
            leaks.append((row["scenario_id"], "user_task", row["user_task"], answer))
        assert "query" not in row["tool_output"], row["scenario_id"]
    assert leaks == []


def test_real_web_domain_expansion_unreliable_outputs_do_not_leak_clean_answers() -> None:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    leaks = []
    visible_fields = ["answer", "answer_field", "entity", "snippet", "title", "url"]
    for row in rows:
        if row["reliability_label"] != "unreliable":
            continue
        answer = row["clean_output"]["answer"]
        for field in visible_fields:
            value = row["tool_output"].get(field, "")
            if _contains_answer(value, answer):
                leaks.append((row["scenario_id"], field, value, answer))
    assert leaks == []


def test_real_web_domain_expansion_wrong_entity_outputs_are_internally_consistent() -> None:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    wrong_entity_rows = [row for row in rows if row["corruption_mode"] == "wrong_user_or_wrong_entity"]
    assert wrong_entity_rows
    for row in wrong_entity_rows:
        visible = row["tool_output"]
        task = row["user_task"]
        assert visible["entity"] not in row["clean_output"]["entity"]
        assert visible["url"] != row["clean_output"]["url"]
        assert "query" not in visible
        assert not _contains_answer(task, row["clean_output"]["answer"])
        assert not any(_contains_answer(str(value), row["clean_output"]["answer"]) for value in visible.values())


def test_real_web_domain_expansion_is_paired_by_capture_id() -> None:
    rows = [json.loads(line) for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    reliable = [row for row in rows if row["reliability_label"] == "reliable"]
    unreliable = [row for row in rows if row["reliability_label"] == "unreliable"]
    assert len(reliable) == len(unreliable) == 50
    reliable_capture_ids = {row["clean_output"]["capture_id"] for row in reliable}
    unreliable_clean_capture_ids = {row["clean_output"]["capture_id"] for row in unreliable}
    assert reliable_capture_ids == unreliable_clean_capture_ids
    assert all(row["metadata"].get("paired_design") is True for row in rows)


def test_real_web_domain_expansion_config_dry_run_summary() -> None:
    config = phase3_run.load_config("configs/real_web_domain_expansion.yaml")
    assert config["source"]["scenarios_path"] == str(SCENARIOS_PATH).replace("\\", "/")
    assert config["run"]["paid_run_authorized"] is False
    assert config["run"]["output_dir"] == "outputs/real_web_domain_expansion_clean_run"
    scenarios = phase3_run.filter_phase3_scenarios(config)
    summary = phase3_run.build_dry_run_summary(config)
    assert len(scenarios) == 100
    assert summary["events"] == 1500
    assert summary["paid_run_authorized"] is False
    assert summary["will_create_paid_outputs"] is False
    assert summary["fits_budget"] is True


def test_real_web_domain_expansion_prompts_do_not_expose_hidden_fields() -> None:
    config = phase3_run.load_config("configs/real_web_domain_expansion.yaml")
    scenarios = phase3_run.filter_phase3_scenarios(config)
    forbidden = {
        "clean_output",
        "corrupt",
        "domain-web-corrupt",
        "domain-web-omission",
        "domain-web-wrong-entity",
        "gold_action",
        "omission",
        "reliability_label",
        "corruption_mode",
        "expected_answer",
        "synthetic_fault",
        "hidden_from_agent",
        "wrong-entity",
    }
    for scenario in scenarios:
        for policy in config["policies"]:
            prompt = phase3_run.PROMPT_BUILDERS[policy](scenario)
            text = json.dumps(prompt, sort_keys=True).lower()
            assert not any(marker in text for marker in forbidden), (scenario.scenario_id, policy)


def _contains_answer(text: str, answer: str) -> bool:
    text_norm = _normalize_for_leak_check(text)
    answer_norm = _normalize_for_leak_check(answer)
    if not text_norm or not answer_norm:
        return False
    return re.search(rf"(^| )({re.escape(answer_norm)})( |$)", text_norm) is not None


def _normalize_for_leak_check(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()
