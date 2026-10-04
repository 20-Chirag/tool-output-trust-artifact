from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from trustbench.data.schemas import (
    CorruptionMode,
    ReliabilityLabel,
    Scenario,
    ToolClass,
    TrustAction,
)


CORRUPTIONS_BY_TOOL = {
    ToolClass.WEB_SEARCH: [CorruptionMode.STALE, CorruptionMode.PARTIAL_OMISSION, CorruptionMode.PLAUSIBLE_FALSE],
    ToolClass.CALCULATOR: [CorruptionMode.OFF_BY_ONE, CorruptionMode.EDGE_CASE, CorruptionMode.PLAUSIBLE_FALSE],
    ToolClass.STRUCTURED_API: [CorruptionMode.PARTIAL_OMISSION, CorruptionMode.SCHEMA_DRIFT, CorruptionMode.WRONG_ENTITY],
    ToolClass.FILE_MEMORY: [CorruptionMode.WRONG_ENTITY, CorruptionMode.STALE, CorruptionMode.PLAUSIBLE_FALSE],
}


def generate_scenarios(
    scenarios_per_tool_class: int = 25,
    seed: int = 7,
    reliability_balance: float = 0.5,
    dataset_name: str = "pilot",
) -> list[Scenario]:
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    for tool_class in ToolClass:
        for i in range(scenarios_per_tool_class):
            reliable = rng.random() < reliability_balance
            scenario = _make_scenario(
                dataset_name=dataset_name,
                tool_class=tool_class,
                index=i,
                reliable=reliable,
                rng=rng,
            )
            scenarios.append(scenario)
    rng.shuffle(scenarios)
    return scenarios


def _make_scenario(
    dataset_name: str,
    tool_class: ToolClass,
    index: int,
    reliable: bool,
    rng: random.Random,
) -> Scenario:
    clean = _clean_output(tool_class, index, rng)
    corruption = CorruptionMode.NONE if reliable else rng.choice(CORRUPTIONS_BY_TOOL[tool_class])
    output = clean if reliable else _corrupt_output(tool_class, clean, corruption, rng)
    reliability = ReliabilityLabel.RELIABLE if reliable else ReliabilityLabel.UNRELIABLE
    gold_action = TrustAction.ACCEPT if reliable else _gold_action_for(corruption)
    task = _user_task(tool_class, clean)
    expected_answer = _expected_answer(tool_class, clean)
    difficulty = rng.choice(["easy", "medium", "hard"])
    return Scenario(
        scenario_id=f"{dataset_name}-{tool_class.value}-{index:03d}",
        tool_class=tool_class,
        user_task=task,
        tool_invocation={"tool": tool_class.value, "arguments": _invocation_args(tool_class, clean)},
        tool_output=output,
        clean_output=clean,
        reliability_label=reliability,
        corruption_mode=corruption,
        gold_action=gold_action,
        expected_answer=expected_answer,
        difficulty=difficulty,
        metadata={
            "synthetic": True,
            "source_clean_id": clean["id"],
            "needs_human_adjudication": False,
        },
    )


def _clean_output(tool_class: ToolClass, index: int, rng: random.Random) -> dict[str, Any]:
    today = date(2026, 4, 25)
    if tool_class == ToolClass.WEB_SEARCH:
        entities = ["Northbridge Transit", "Aster Bank", "Civic Labs", "HelioGrid", "Mira Health"]
        entity = entities[index % len(entities)]
        amount = 120 + index * 3
        return {
            "id": f"web-{index}",
            "query": f"{entity} 2026 service fee",
            "title": f"{entity} updates 2026 service fee",
            "snippet": f"{entity}'s current service fee is ${amount} for the standard plan.",
            "url": f"https://example.org/{entity.lower().replace(' ', '-')}/fees",
            "published_at": str(today - timedelta(days=rng.randint(1, 12))),
            "retrieved_at": str(today),
            "entity": entity,
            "answer": f"${amount}",
        }
    if tool_class == ToolClass.CALCULATOR:
        a = 18 + index
        b = 7 + (index % 9)
        result = a * b + 3
        return {
            "id": f"calc-{index}",
            "expression": f"{a} * {b} + 3",
            "result": result,
            "units": "points",
            "engine": "decimal-v1",
            "checked_at": "2026-04-25T09:00:00Z",
            "answer": str(result),
        }
    if tool_class == ToolClass.STRUCTURED_API:
        user_id = f"user_{1000 + index}"
        balance = 2000 + index * 17
        return {
            "id": f"api-{index}",
            "endpoint": "/v2/accounts/summary",
            "requested_user_id": user_id,
            "user_id": user_id,
            "account_status": "active",
            "balance_cents": balance,
            "currency": "USD",
            "updated_at": "2026-04-24T18:30:00Z",
            "schema_version": "2.1",
            "answer": f"{balance} cents",
        }
    project = ["Project Delta", "Atlas notes", "Mira case file", "Quarterly memo", "Vendor brief"][index % 5]
    owner = f"employee_{200 + index}"
    return {
        "id": f"mem-{index}",
        "requested_entity": project,
        "retrieved_entity": project,
        "owner_id": owner,
        "document_id": f"doc-{8000 + index}",
        "snippet": f"{project} lists owner {owner} and status approved.",
        "last_updated": "2026-04-22",
        "answer": owner,
    }


def _corrupt_output(
    tool_class: ToolClass,
    clean: dict[str, Any],
    corruption: CorruptionMode,
    rng: random.Random,
) -> dict[str, Any]:
    output = dict(clean)
    if corruption == CorruptionMode.STALE:
        if tool_class == ToolClass.WEB_SEARCH:
            stale_date = date.fromisoformat(clean["published_at"]) - timedelta(days=420)
            output["published_at"] = str(stale_date)
            output["snippet"] = output["snippet"].replace("current", "reported")
            output["answer"] = f"${int(clean['answer'].strip('$')) - 20}"
        else:
            output["last_updated"] = "2024-11-03"
            output["snippet"] = output["snippet"].replace("approved", "pending review")
    elif corruption == CorruptionMode.PARTIAL_OMISSION:
        if tool_class == ToolClass.STRUCTURED_API:
            output.pop("currency", None)
            output.pop("account_status", None)
        else:
            output.pop("url", None)
            output["snippet"] = "The standard plan fee is listed in the account table."
            output["answer"] = "not shown"
    elif corruption == CorruptionMode.OFF_BY_ONE:
        output["result"] = int(clean["result"]) + rng.choice([-1, 1])
        output["answer"] = str(output["result"])
    elif corruption == CorruptionMode.SCHEMA_DRIFT:
        output["schema_version"] = "3.0-beta"
        output["balance"] = str(output.pop("balance_cents"))
        output["user"] = output.pop("user_id")
    elif corruption == CorruptionMode.WRONG_ENTITY:
        if tool_class == ToolClass.STRUCTURED_API:
            output["user_id"] = f"user_{rng.randint(3000, 3999)}"
            output["balance_cents"] = int(clean["balance_cents"]) + 911
            output["answer"] = f"{output['balance_cents']} cents"
        else:
            output["retrieved_entity"] = clean["retrieved_entity"] + " archive"
            output["owner_id"] = f"employee_{rng.randint(700, 799)}"
            output["answer"] = output["owner_id"]
            output["snippet"] = output["snippet"].replace(clean["answer"], output["owner_id"])
    elif corruption == CorruptionMode.PLAUSIBLE_FALSE:
        if tool_class == ToolClass.CALCULATOR:
            output["result"] = int(clean["result"]) + rng.choice([9, 11, 13])
            output["answer"] = str(output["result"])
            output["engine"] = "fast-int-v2"
        elif tool_class == ToolClass.WEB_SEARCH:
            output["snippet"] = output["snippet"].replace(clean["answer"], f"${int(clean['answer'].strip('$')) + 35}")
            output["answer"] = f"${int(clean['answer'].strip('$')) + 35}"
        else:
            output["snippet"] = output["snippet"].replace("approved", "approved with finance exception")
            output["answer"] = "finance exception"
    elif corruption == CorruptionMode.EDGE_CASE:
        output["result"] = int(clean["result"]) - 3
        output["answer"] = str(output["result"])
        output["engine"] = "float-fastpath"
    return output


def _gold_action_for(corruption: CorruptionMode) -> TrustAction:
    if corruption in {CorruptionMode.PARTIAL_OMISSION, CorruptionMode.SCHEMA_DRIFT}:
        return TrustAction.VERIFY
    if corruption in {CorruptionMode.WRONG_ENTITY, CorruptionMode.PLAUSIBLE_FALSE}:
        return TrustAction.ABSTAIN
    return TrustAction.VERIFY


def _user_task(tool_class: ToolClass, clean: dict[str, Any]) -> str:
    if tool_class == ToolClass.WEB_SEARCH:
        return f"Use the search result to answer: what is the current service fee for {clean['entity']}?"
    if tool_class == ToolClass.CALCULATOR:
        return f"Use the calculator output to answer the expression {clean['expression']}."
    if tool_class == ToolClass.STRUCTURED_API:
        return f"Use the account API output to report the balance for {clean['requested_user_id']}."
    return f"Use memory retrieval to identify the owner for {clean['requested_entity']}."


def _invocation_args(tool_class: ToolClass, clean: dict[str, Any]) -> dict[str, Any]:
    if tool_class == ToolClass.WEB_SEARCH:
        return {"query": clean["query"]}
    if tool_class == ToolClass.CALCULATOR:
        return {"expression": clean["expression"]}
    if tool_class == ToolClass.STRUCTURED_API:
        return {"user_id": clean["requested_user_id"]}
    return {"entity": clean["requested_entity"]}


def _expected_answer(tool_class: ToolClass, clean: dict[str, Any]) -> str:
    if tool_class == ToolClass.STRUCTURED_API:
        return f"{clean['requested_user_id']} balance is {clean['balance_cents']} {clean['currency']} cents."
    return str(clean["answer"])
