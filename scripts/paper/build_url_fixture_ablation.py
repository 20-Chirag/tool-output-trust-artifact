"""Build the URL-only fixture arm for the controlled web ablation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "real_web_domain_expansion" / "scenarios_candidate.jsonl"
OUT_DIR = ROOT / "outputs" / "url_fixture_ablation"
OUTPUT = OUT_DIR / "scenarios_fixture.jsonl"
MANIFEST = OUT_DIR / "build_manifest.json"
INSPECTION = OUT_DIR / "inspection_note.md"


def main() -> None:
    source_rows = _read_jsonl(SOURCE)
    fixture_rows = [_fixture_row(row) for row in source_rows]
    _validate(source_rows, fixture_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUTPUT, fixture_rows)
    manifest = _manifest(source_rows, fixture_rows)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    INSPECTION.write_text(_inspection_note(source_rows, fixture_rows, manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _fixture_row(row: dict) -> dict:
    copied = json.loads(json.dumps(row))
    copied["tool_output"]["url"] = _replace_host(str(row["tool_output"]["url"]))
    return copied


def _replace_host(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.netloc:
        raise ValueError(f"Visible URL has no host: {url}")
    return urlunsplit((parsed.scheme or "https", "example.org", parsed.path, parsed.query, parsed.fragment))


def _validate(source_rows: list[dict], fixture_rows: list[dict]) -> None:
    if len(source_rows) != 100 or len(fixture_rows) != 100:
        raise ValueError(f"Expected 100 matched scenarios, got {len(source_rows)} and {len(fixture_rows)}")
    if len({row["scenario_id"] for row in source_rows}) != 100:
        raise ValueError("Source scenario IDs are not unique")
    for source, fixture in zip(source_rows, fixture_rows, strict=True):
        if source["scenario_id"] != fixture["scenario_id"]:
            raise ValueError("Scenario order or identity changed")
        expected = json.loads(json.dumps(source))
        expected["tool_output"]["url"] = fixture["tool_output"]["url"]
        if fixture != expected:
            raise ValueError(f"More than visible URL changed for {source['scenario_id']}")
        if urlsplit(fixture["tool_output"]["url"]).netloc.lower() != "example.org":
            raise ValueError(f"Fixture URL is not example.org for {source['scenario_id']}")
        if "example.org" in str(fixture["clean_output"].get("url", "")).lower():
            raise ValueError(f"Hidden clean URL was modified for {source['scenario_id']}")


def _manifest(source_rows: list[dict], fixture_rows: list[dict]) -> dict:
    labels = Counter(row["reliability_label"] for row in fixture_rows)
    modes = Counter(row["corruption_mode"] for row in fixture_rows)
    return {
        "design": "matched URL-host-only fixture arm",
        "source_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "output_path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": _sha256(SOURCE),
        "scenario_count": len(fixture_rows),
        "reliability_counts": dict(sorted(labels.items())),
        "corruption_mode_counts": dict(sorted(modes.items())),
        "fixture_host": "example.org",
        "fixture_url_count": sum(
            urlsplit(row["tool_output"]["url"]).netloc.lower() == "example.org"
            for row in fixture_rows
        ),
        "scenario_ids_preserved": True,
        "only_changed_model_visible_field": "tool_output.url",
        "url_component_changed": "host only",
        "clean_output_preserved": True,
        "paid_model_calls_made": False,
    }


def _inspection_note(source_rows: list[dict], fixture_rows: list[dict], manifest: dict) -> str:
    examples = []
    chosen_groups = set()
    for source, fixture in zip(source_rows, fixture_rows, strict=True):
        group = source.get("metadata", {}).get("domain_group", "unknown")
        if group in chosen_groups:
            continue
        chosen_groups.add(group)
        examples.append((source, fixture, group))
    lines = [
        "# Controlled URL-Fixture Candidate Inspection",
        "",
        "This candidate is a matched ablation of the clean broader-web dataset.",
        "The only changed model-visible field is `tool_output.url`; specifically,",
        "the original host is replaced by `example.org` while the URL path and all",
        "task, query, title, snippet, answer, and corruption content remain fixed.",
        "",
        "## Integrity",
        "",
        f"- Scenarios: {manifest['scenario_count']}",
        f"- Reliable / unreliable: {manifest['reliability_counts'].get('reliable', 0)} / {manifest['reliability_counts'].get('unreliable', 0)}",
        f"- Visible fixture URLs: {manifest['fixture_url_count']} / {manifest['scenario_count']}",
        "- Scenario IDs preserved: yes",
        "- Hidden clean outputs preserved: yes",
        "- Paid model calls made: no",
        "",
        "## Examples",
        "",
    ]
    for source, fixture, group in examples:
        lines.extend(
            [
                f"### `{source['scenario_id']}` ({group})",
                "",
                f"- Original URL: `{source['tool_output']['url']}`",
                f"- Fixture URL: `{fixture['tool_output']['url']}`",
                f"- Visible answer: `{fixture['tool_output'].get('answer', '')}`",
                f"- Reliability: `{fixture['reliability_label']}`",
                f"- Corruption mode: `{fixture['corruption_mode']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "This dataset does not simulate a different search engine or a different",
            "retrieval result. It tests only whether a placeholder-looking URL host",
            "changes the model's decision when the rest of the visible evidence is held",
            "constant.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
