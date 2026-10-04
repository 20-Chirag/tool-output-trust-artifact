from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _minimal_yaml_load(text)


def _minimal_yaml_load(text: str) -> dict:
    """Parse the small YAML subset used by this repository's configs.

    This keeps the pilot runnable in lightweight environments where PyYAML is
    not installed. It supports nested dictionaries, scalar lists, and lists of
    flat dictionaries.
    """

    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    parsed, next_index = _parse_dict(lines, 0, 0)
    if next_index != len(lines):
        raise ValueError("Could not parse entire config file")
    return parsed


def _parse_dict(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict, int]:
    result: dict = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation near: {content}")
        if content.startswith("- "):
            break
        key, value = _split_key_value(content)
        if value is not None:
            result[key] = _parse_scalar(value)
            index += 1
            continue
        if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
            result[key] = {}
            index += 1
            continue
        child_indent = lines[index + 1][0]
        if lines[index + 1][1].startswith("- "):
            result[key], index = _parse_list(lines, index + 1, child_indent)
        else:
            result[key], index = _parse_dict(lines, index + 1, child_indent)
    return result, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list, int]:
    result = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or not content.startswith("- "):
            break
        item = content[2:].strip()
        if ":" in item:
            key, value = _split_key_value(item)
            entry = {key: _parse_scalar(value) if value is not None else {}}
            index += 1
            while index < len(lines) and lines[index][0] > indent:
                child_indent, child_content = lines[index]
                if child_content.startswith("- "):
                    break
                child_key, child_value = _split_key_value(child_content)
                if child_value is not None:
                    entry[child_key] = _parse_scalar(child_value)
                    index += 1
                else:
                    if index + 1 < len(lines) and lines[index + 1][0] > child_indent:
                        if lines[index + 1][1].startswith("- "):
                            entry[child_key], index = _parse_list(lines, index + 1, lines[index + 1][0])
                        else:
                            entry[child_key], index = _parse_dict(lines, index + 1, lines[index + 1][0])
                    else:
                        entry[child_key] = {}
                        index += 1
            result.append(entry)
        else:
            result.append(_parse_scalar(item))
            index += 1
    return result, index


def _split_key_value(content: str) -> tuple[str, str | None]:
    if ":" not in content:
        raise ValueError(f"Expected key/value pair: {content}")
    key, value = content.split(":", 1)
    value = value.strip()
    return key.strip(), value if value else None


def _parse_scalar(value: str):
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
