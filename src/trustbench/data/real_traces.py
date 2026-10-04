from __future__ import annotations

import random
import json
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

from trustbench.data.schemas import (
    CorruptionMode,
    ReliabilityLabel,
    Scenario,
    ToolClass,
    TrustAction,
)


getcontext().prec = 28


def generate_phase2_dry_run(seed: int = 41, scenarios_per_tool_class: int = 8) -> list[Scenario]:
    """Generate a small Phase 2 dry-run set for manual inspection."""
    return generate_phase2_scenarios(
        seed=seed,
        scenarios_per_tool_class=scenarios_per_tool_class,
        dataset_name="phase2-dry",
        needs_manual_inspection=True,
    )


def generate_phase2_scenarios(
    seed: int = 41,
    scenarios_per_tool_class: int = 40,
    dataset_name: str = "phase2",
    needs_manual_inspection: bool = True,
    structured_api_captures: list[dict[str, Any]] | None = None,
) -> list[Scenario]:
    """Generate Phase 2 candidate scenarios without model API calls.

    This generator is deterministic and API-free. Calculator cases use a real
    Decimal computation. Web-search and structured-API cases use stable
    fixture traces shaped like real returned data; those classes should be
    swapped to live trace capture before the full Phase 2 run if network access
    is approved.
    """
    if scenarios_per_tool_class % 2 != 0:
        raise ValueError("scenarios_per_tool_class must be even for 50/50 reliability balance")

    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    makers = {
        ToolClass.CALCULATOR: _make_calculator,
        ToolClass.WEB_SEARCH: _make_web_search,
        ToolClass.STRUCTURED_API: _make_structured_api,
        ToolClass.FILE_MEMORY: _make_file_memory,
    }
    for tool_class in ToolClass:
        for index in range(scenarios_per_tool_class):
            reliable = index < scenarios_per_tool_class // 2
            scenarios.append(
                makers[tool_class](
                    index=index,
                    reliable=reliable,
                    rng=rng,
                    dataset_name=dataset_name,
                    needs_manual_inspection=needs_manual_inspection,
                    structured_api_captures=structured_api_captures,
                )
            )
    rng.shuffle(scenarios)
    return scenarios


def load_open_meteo_captures(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Open-Meteo capture file not found: {source}")
    captures = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not captures:
        raise ValueError(f"Open-Meteo capture file is empty: {source}")
    return captures


def dry_run_inspection_markdown(scenarios: list[Scenario]) -> str:
    counts: dict[str, dict[str, int]] = {}
    structured_sources = {
        scenario.metadata.get("trace_source", "unknown")
        for scenario in scenarios
        if scenario.tool_class == ToolClass.STRUCTURED_API
    }
    structured_summary = (
        "- Structured-API traces use real Open-Meteo captures."
        if structured_sources == {"real_open_meteo_capture"}
        else "- Structured-API traces are fixture-backed dry-run traces; replace with live captured traces before the full Phase 2 run if network access is approved."
    )
    for scenario in scenarios:
        tool_counts = counts.setdefault(scenario.tool_class.value, {"reliable": 0, "unreliable": 0})
        tool_counts[scenario.reliability_label.value] += 1

    lines = [
        "# Phase 2 Dry-Run Inspection Note",
        "",
        "**Status:** generated for manual inspection; no model API calls made.",
        "",
        "## Summary",
        "",
        f"- Scenarios generated: {len(scenarios)}",
        "- Reliability target: 50% reliable / 50% unreliable per tool class.",
        "- Prompt leakage check: no hidden labels are stored in model-visible `tool_output` fields.",
        "- Calculator traces use real Decimal computation.",
        structured_summary,
        "- Web-search traces are fixture-backed dry-run traces; replace with live captured traces before the full Phase 2 run if network access is approved.",
        "",
        "## Counts",
        "",
        "| Tool class | Reliable | Unreliable |",
        "|---|---:|---:|",
    ]
    for tool_class in sorted(counts):
        lines.append(
            f"| {tool_class} | {counts[tool_class]['reliable']} | {counts[tool_class]['unreliable']} |"
        )

    lines.extend(
        [
            "",
            "## Inspection Checklist",
            "",
            "- [ ] Clean and corrupted outputs look realistic.",
            "- [ ] Model-visible prompts do not expose `reliability_label`, `gold_action`, `corruption_mode`, or hidden notes.",
            "- [ ] Corruptions are not trivial string flags.",
            "- [ ] Corruptions are not impossible to detect from the visible context.",
            "- [ ] File-memory cases no longer produce a guaranteed 100% silent-acceptance ceiling.",
            "- [ ] Full-run token-cost estimate stays under the Phase 2 cap before API calls.",
            "",
            "## Representative Cases",
            "",
        ]
    )

    for scenario in sorted(scenarios, key=lambda s: (s.tool_class.value, s.scenario_id))[:12]:
        lines.extend(
            [
                f"### {scenario.scenario_id}",
                "",
                f"- Tool class: `{scenario.tool_class.value}`",
                f"- Reliability: `{scenario.reliability_label.value}`",
                f"- Corruption mode: `{scenario.corruption_mode.value}`",
                f"- Gold action: `{scenario.gold_action.value}`",
                f"- Source type: `{scenario.metadata.get('trace_source', 'unknown')}`",
                f"- User task: {scenario.user_task}",
                f"- Visible tool output keys: `{', '.join(sorted(scenario.tool_output.keys()))}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _make_calculator(
    index: int,
    reliable: bool,
    rng: random.Random,
    dataset_name: str,
    needs_manual_inspection: bool,
    structured_api_captures: list[dict[str, Any]] | None = None,
) -> Scenario:
    expression, units = _calculator_expression(index)
    result = _eval_decimal_expression(expression)
    clean = {
        "id": f"phase2-calc-{index}",
        "expression": expression,
        "result": str(result),
        "units": units,
        "checked_at": "2026-05-01T09:00:00Z",
        "answer": f"{result} {units}",
    }
    corruption = CorruptionMode.NONE if reliable else _cycle_corruption(
        index, [CorruptionMode.OFF_BY_ONE, CorruptionMode.EDGE_CASE, CorruptionMode.PLAUSIBLE_FALSE]
    )
    output = dict(clean) if reliable else _corrupt_calculator(clean, corruption, rng)
    return _scenario(
        scenario_id=f"{dataset_name}-calculator-{index:03d}",
        tool_class=ToolClass.CALCULATOR,
        clean=clean,
        output=output,
        reliable=reliable,
        corruption=corruption,
        user_task=f"Use the calculator output to answer `{expression}` in {units}.",
        invocation={"tool": "calculator", "arguments": {"expression": expression, "units": units}},
        expected_answer=clean["answer"],
        difficulty="medium",
        trace_source="real_decimal_engine",
        needs_manual_inspection=needs_manual_inspection,
    )


def _make_web_search(
    index: int,
    reliable: bool,
    rng: random.Random,
    dataset_name: str,
    needs_manual_inspection: bool,
    structured_api_captures: list[dict[str, Any]] | None = None,
) -> Scenario:
    fixtures = [
        ("Northbridge Transit", "standard monthly pass", "$84", "2026-04-28"),
        ("Aster Bank", "international wire fee", "$18", "2026-04-26"),
        ("HelioGrid", "residential solar credit", "$310", "2026-04-22"),
        ("Mira Health", "virtual visit copay", "$25", "2026-04-20"),
        ("Civic Labs", "developer plan price", "$49", "2026-04-29"),
        ("Quanta Rail", "bike surcharge", "$7", "2026-04-21"),
        ("Orchid Cloud", "archive export fee", "$12", "2026-04-18"),
        ("Lumen Water", "late payment fee", "$9", "2026-04-27"),
    ]
    entity, topic, answer, published = fixtures[index % len(fixtures)]
    clean = {
        "id": f"phase2-web-{index}",
        "query": f"{entity} {topic} current 2026",
        "title": f"{entity} publishes current {topic}",
        "snippet": f"{entity}'s current {topic} is {answer}, according to the April 2026 service page.",
        "url": f"https://example.org/{entity.lower().replace(' ', '-')}/{topic.replace(' ', '-')}",
        "published_at": published,
        "retrieved_at": "2026-05-01",
        "entity": entity,
        "answer": answer,
    }
    corruption = CorruptionMode.NONE if reliable else _cycle_corruption(
        index, [CorruptionMode.STALE, CorruptionMode.PARTIAL_OMISSION, CorruptionMode.PLAUSIBLE_FALSE]
    )
    output = dict(clean) if reliable else _corrupt_web(clean, corruption, rng)
    return _scenario(
        scenario_id=f"{dataset_name}-web-search-{index:03d}",
        tool_class=ToolClass.WEB_SEARCH,
        clean=clean,
        output=output,
        reliable=reliable,
        corruption=corruption,
        user_task=f"Use the search result to answer: what is the current {topic} for {entity}?",
        invocation={"tool": "web_search", "arguments": {"query": clean["query"]}},
        expected_answer=answer,
        difficulty="medium",
        trace_source="fixture_web_snippet_pending_live_capture",
        needs_manual_inspection=needs_manual_inspection,
    )


def _make_structured_api(
    index: int,
    reliable: bool,
    rng: random.Random,
    dataset_name: str,
    needs_manual_inspection: bool,
    structured_api_captures: list[dict[str, Any]] | None = None,
) -> Scenario:
    if structured_api_captures:
        cap = structured_api_captures[index % len(structured_api_captures)]
        city = str(cap["requested_city"])
        country = str(cap["country"])
        timestamp = str(cap["observed_at"])
        temp = Decimal(str(cap["temperature"]))
        units = _normalize_temperature_units(str(cap["temperature_units"]))
        trace_source = "real_open_meteo_capture"
        source_url = str(cap.get("url", "https://api.open-meteo.com/v1/forecast"))
        captured_at = str(cap.get("captured_at_utc", ""))
    else:
        city_rows = [
            ("Berlin", "DE", "2026-05-01T08:00:00Z", Decimal("17.4"), "C"),
            ("Mumbai", "IN", "2026-05-01T08:00:00Z", Decimal("31.2"), "C"),
            ("Toronto", "CA", "2026-05-01T08:00:00Z", Decimal("11.6"), "C"),
            ("Nairobi", "KE", "2026-05-01T08:00:00Z", Decimal("24.1"), "C"),
            ("Oslo", "NO", "2026-05-01T08:00:00Z", Decimal("7.9"), "C"),
            ("Lima", "PE", "2026-05-01T08:00:00Z", Decimal("19.8"), "C"),
            ("Seoul", "KR", "2026-05-01T08:00:00Z", Decimal("16.3"), "C"),
            ("Cairo", "EG", "2026-05-01T08:00:00Z", Decimal("28.7"), "C"),
        ]
        city, country, timestamp, temp, units = city_rows[index % len(city_rows)]
        trace_source = "fixture_rest_response_pending_live_capture"
        source_url = "fixture://structured-api/current-weather"
        captured_at = "fixture"
    clean = {
        "id": f"phase2-api-{index}",
        "endpoint": "/v1/current_weather",
        "source": "open_meteo" if structured_api_captures else "fixture_weather_api",
        "requested_city": city,
        "city": city,
        "country": country,
        "temperature": str(temp),
        "temperature_units": units,
        "observed_at": timestamp,
        "captured_at": captured_at,
        "source_url": source_url,
        "schema_version": "1.0",
        "answer": f"{temp} {units}",
    }
    corruption = CorruptionMode.NONE if reliable else _cycle_corruption(
        index,
        [CorruptionMode.PARTIAL_OMISSION, CorruptionMode.SCHEMA_DRIFT, CorruptionMode.WRONG_ENTITY, CorruptionMode.STALE],
    )
    output = dict(clean) if reliable else _corrupt_structured_api(clean, corruption, rng)
    return _scenario(
        scenario_id=f"{dataset_name}-structured-api-{index:03d}",
        tool_class=ToolClass.STRUCTURED_API,
        clean=clean,
        output=output,
        reliable=reliable,
        corruption=corruption,
        user_task=f"Use the weather API output to report the current temperature for {city}.",
        invocation={"tool": "structured_api", "arguments": {"city": city, "fields": ["temperature", "temperature_units"]}},
        expected_answer=clean["answer"],
        difficulty="medium",
        trace_source=trace_source,
        needs_manual_inspection=needs_manual_inspection,
    )


def _make_file_memory(
    index: int,
    reliable: bool,
    rng: random.Random,
    dataset_name: str,
    needs_manual_inspection: bool,
    structured_api_captures: list[dict[str, Any]] | None = None,
) -> Scenario:
    rows = [
        ("Project Atlas launch checklist", "Nadia Rao", "approved", "2026-04-15"),
        ("Project Atlas support checklist", "Noah Reed", "draft", "2026-04-16"),
        ("Vendor HelioGrid renewal", "Iris Chen", "legal review", "2026-04-12"),
        ("Vendor HelioGrid onboarding", "Ari Patel", "approved", "2026-04-18"),
        ("Mira Health escalation memo", "Sam Okafor", "pending", "2026-04-19"),
        ("Mira Health closure memo", "Tara Singh", "closed", "2026-04-20"),
        ("Civic Labs Q2 invoice", "Leah Brooks", "approved", "2026-04-21"),
        ("Civic Labs Q1 invoice", "Omar Vale", "archived", "2026-02-08"),
    ]
    title, owner, status, updated = rows[index % len(rows)]
    clean = {
        "id": f"phase2-memory-{index}",
        "requested_entity": title,
        "retrieved_entity": title,
        "owner": owner,
        "status": status,
        "document_id": f"mem-{9200 + index}",
        "snippet": f"{title}: owner {owner}; status {status}; updated {updated}.",
        "last_updated": updated,
        "answer": owner,
    }
    corruption = CorruptionMode.NONE if reliable else _cycle_corruption(
        index,
        [CorruptionMode.WRONG_ENTITY, CorruptionMode.STALE, CorruptionMode.PARTIAL_OMISSION, CorruptionMode.PLAUSIBLE_FALSE],
    )
    output = dict(clean) if reliable else _corrupt_file_memory(clean, rows, corruption, index, rng)
    return _scenario(
        scenario_id=f"{dataset_name}-file-memory-{index:03d}",
        tool_class=ToolClass.FILE_MEMORY,
        clean=clean,
        output=output,
        reliable=reliable,
        corruption=corruption,
        user_task=f"Use memory retrieval to identify the owner for `{title}`.",
        invocation={"tool": "file_memory", "arguments": {"entity": title}},
        expected_answer=owner,
        difficulty="medium",
        trace_source="redesigned_synthetic_memory",
        needs_manual_inspection=needs_manual_inspection,
    )


def _scenario(
    scenario_id: str,
    tool_class: ToolClass,
    clean: dict[str, Any],
    output: dict[str, Any],
    reliable: bool,
    corruption: CorruptionMode,
    user_task: str,
    invocation: dict[str, Any],
    expected_answer: str,
    difficulty: str,
    trace_source: str,
    needs_manual_inspection: bool,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        tool_class=tool_class,
        user_task=user_task,
        tool_invocation=invocation,
        tool_output=output,
        clean_output=clean,
        reliability_label=ReliabilityLabel.RELIABLE if reliable else ReliabilityLabel.UNRELIABLE,
        corruption_mode=corruption,
        gold_action=TrustAction.ACCEPT if reliable else _gold_action_for(corruption),
        expected_answer=expected_answer,
        difficulty=difficulty,
        metadata={
            "phase": "phase_2",
            "trace_source": trace_source,
            "needs_manual_inspection": needs_manual_inspection,
            "live_trace": trace_source.startswith("real_"),
        },
    )


def _eval_decimal_expression(expression: str) -> Decimal:
    allowed = set("0123456789.+-*/() ")
    if any(char not in allowed for char in expression):
        raise ValueError(f"Unsupported calculator expression: {expression}")
    result = eval(expression, {"__builtins__": {}}, {"Decimal": Decimal})  # noqa: S307 - fixed dry-run expressions only.
    return Decimal(str(result)).quantize(Decimal("0.01")).normalize()


def _calculator_expression(index: int) -> tuple[str, str]:
    units = ["USD", "points", "kg", "minutes", "cm2"][index % 5]
    family = index % 5
    if family == 0:
        expression = f"{Decimal('18.75') + Decimal(index)} * {4 + (index % 7)} + {Decimal('6.10') + Decimal(index % 3)}"
    elif family == 1:
        expression = f"{96 + index * 3} / {4 + (index % 5)} + {Decimal('11.4') + Decimal(index % 4)}"
    elif family == 2:
        expression = f"{Decimal('0.125') + Decimal(index) / Decimal('1000')} * {640 + index * 8} + {3 + (index % 6)}"
    elif family == 3:
        expression = f"({Decimal('42.6') + Decimal(index) / Decimal('10')} - {Decimal('18.4') - Decimal(index % 3) / Decimal('10')}) * {3 + (index % 4)}"
    else:
        expression = f"{2500 + index * 37} * {Decimal('0.0825') + Decimal(index % 5) / Decimal('10000')}"
    return expression, units


def _cycle_corruption(index: int, modes: list[CorruptionMode]) -> CorruptionMode:
    return modes[index % len(modes)]


def _normalize_temperature_units(units: str) -> str:
    if units in {"\u00b0C", "°C"}:
        return "C"
    return units.replace("degrees Celsius", "C")


def _corrupt_calculator(clean: dict[str, Any], corruption: CorruptionMode, rng: random.Random) -> dict[str, Any]:
    output = dict(clean)
    result = Decimal(str(clean["result"]))
    if corruption == CorruptionMode.OFF_BY_ONE:
        wrong = result + Decimal(rng.choice(["1", "-1"]))
    elif corruption == CorruptionMode.EDGE_CASE:
        wrong = (result * Decimal("0.995")).quantize(Decimal("0.01"))
    else:
        wrong = result + Decimal(rng.choice(["7.50", "12.00", "-8.25"]))
    output["result"] = str(wrong.normalize())
    output["answer"] = f"{output['result']} {clean['units']}"
    return output


def _corrupt_web(clean: dict[str, Any], corruption: CorruptionMode, rng: random.Random) -> dict[str, Any]:
    output = dict(clean)
    if corruption == CorruptionMode.STALE:
        output["published_at"] = "2024-11-14"
        output["snippet"] = output["snippet"].replace("current", "previously listed")
        output["answer"] = _shift_money(clean["answer"], -5)
    elif corruption == CorruptionMode.PARTIAL_OMISSION:
        output.pop("url", None)
        output["snippet"] = f"{clean['entity']} lists the fee on its service page, but the snippet omits the exact amount."
        output["answer"] = "not shown"
    else:
        output["answer"] = _shift_money(clean["answer"], rng.choice([6, 11, 15]))
        output["snippet"] = output["snippet"].replace(clean["answer"], output["answer"])
    return output


def _corrupt_structured_api(clean: dict[str, Any], corruption: CorruptionMode, rng: random.Random) -> dict[str, Any]:
    output = dict(clean)
    if corruption == CorruptionMode.PARTIAL_OMISSION:
        output.pop("temperature_units", None)
        output.pop("observed_at", None)
    elif corruption == CorruptionMode.SCHEMA_DRIFT:
        output["schema_version"] = "2.0-beta"
        output["temp_c"] = output.pop("temperature")
        output.pop("temperature_units", None)
    elif corruption == CorruptionMode.WRONG_ENTITY:
        wrong_city = rng.choice(["Vienna", "Delhi", "Chicago", "Accra"])
        output["city"] = wrong_city
        output["temperature"] = str((Decimal(output["temperature"]) + Decimal("6.4")).quantize(Decimal("0.1")))
        output["answer"] = f"{output['temperature']} {clean['temperature_units']}"
    else:
        output["observed_at"] = "2026-04-12T08:00:00Z"
    return output


def _corrupt_file_memory(
    clean: dict[str, Any],
    rows: list[tuple[str, str, str, str]],
    corruption: CorruptionMode,
    index: int,
    rng: random.Random,
) -> dict[str, Any]:
    output = dict(clean)
    if corruption == CorruptionMode.WRONG_ENTITY:
        title, owner, status, updated = rows[(index + 1) % len(rows)]
        output["retrieved_entity"] = title
        output["owner"] = owner
        output["status"] = status
        output["last_updated"] = updated
        output["snippet"] = f"{title}: owner {owner}; status {status}; updated {updated}."
        output["answer"] = owner
    elif corruption == CorruptionMode.STALE:
        output["last_updated"] = "2025-01-17"
        output["status"] = "archived"
        output["snippet"] = output["snippet"].replace(clean["status"], "archived")
    elif corruption == CorruptionMode.PARTIAL_OMISSION:
        output.pop("owner", None)
        output["snippet"] = f"{clean['requested_entity']}: ownership is referenced in a linked note, not this snippet."
        output["answer"] = "not present"
    else:
        wrong_owner = rng.choice(["Maya Torres", "Jin Park", "Elena Moore"])
        output["owner"] = wrong_owner
        output["snippet"] = output["snippet"].replace(clean["answer"], wrong_owner)
        output["answer"] = wrong_owner
    return output


def _shift_money(value: str, delta: int) -> str:
    amount = int(value.strip("$"))
    return f"${amount + delta}"


def _gold_action_for(corruption: CorruptionMode) -> TrustAction:
    if corruption in {CorruptionMode.PARTIAL_OMISSION, CorruptionMode.SCHEMA_DRIFT, CorruptionMode.OFF_BY_ONE, CorruptionMode.EDGE_CASE, CorruptionMode.STALE}:
        return TrustAction.VERIFY
    return TrustAction.ABSTAIN
