from __future__ import annotations

from pathlib import Path

import pandas as pd

from trustbench.data.io import ensure_parent


def render_plots(config: dict) -> list[Path]:
    metrics = pd.read_csv(config["metrics"]["output_csv"])
    out_dir = Path(config["analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_silent_by_tool(metrics, out_dir / "silent_acceptance_by_tool.svg"),
        _plot_auroc(metrics, out_dir / "trust_auroc_by_policy.svg"),
        _plot_success_vs_silent(metrics, out_dir / "success_vs_silent.svg"),
    ]
    return paths


def _plot_silent_by_tool(metrics: pd.DataFrame, path: Path) -> Path:
    data = metrics[
        (metrics["tool_class"] != "all")
        & (metrics["policy_name"].isin(["baseline", "skeptical", "heuristic_trust", "trust_calibrated"]))
    ]
    grouped = data.groupby(["tool_class", "policy_name"])["silent_acceptance_rate"].mean().unstack().fillna(0)
    return _bar_svg(grouped, path, "Silent Acceptance by Tool Class", "silent_acceptance_rate")


def _plot_auroc(metrics: pd.DataFrame, path: Path) -> Path:
    data = metrics[metrics["tool_class"] == "all"].groupby("policy_name")["trust_AUROC"].mean().sort_values()
    df = pd.DataFrame({"trust_AUROC": data})
    return _bar_svg(df, path, "Reliability Separation by Policy", "trust_AUROC")


def _plot_success_vs_silent(metrics: pd.DataFrame, path: Path) -> Path:
    data = metrics[metrics["tool_class"] == "all"]
    width, height = 760, 440
    margin = 60
    points = []
    for _, row in data.iterrows():
        x = margin + row["silent_acceptance_rate"] * (width - 2 * margin)
        y = height - margin - row["task_success_rate"] * (height - 2 * margin)
        label = f"{row['model_name']} {row['policy_name']}"
        points.append((x, y, label))
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="60" y="32" font-family="Arial" font-size="20" font-weight="700">Task Success vs Silent Acceptance</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
        f'<text x="{width/2-80}" y="{height-18}" font-family="Arial" font-size="13">silent_acceptance_rate</text>',
        f'<text x="12" y="{height/2}" font-family="Arial" font-size="13" transform="rotate(-90 12 {height/2})">task_success_rate</text>',
    ]
    palette = ["#3b82f6", "#f97316", "#16a34a", "#9333ea", "#64748b"]
    for i, (x, y, label) in enumerate(points):
        color = palette[i % len(palette)]
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        body.append(f'<text x="{x+7:.1f}" y="{y-7:.1f}" font-family="Arial" font-size="10">{label}</text>')
    body.append("</svg>")
    ensure_parent(path).write_text("\n".join(body), encoding="utf-8")
    return path


def _bar_svg(df: pd.DataFrame, path: Path, title: str, y_label: str) -> Path:
    width, height = 880, 480
    margin_left, margin_bottom, margin_top = 72, 76, 56
    plot_w = width - margin_left - 40
    plot_h = height - margin_top - margin_bottom
    max_value = max(0.05, float(df.max().max()))
    columns = list(df.columns)
    rows = list(df.index)
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#475569"]
    group_w = plot_w / max(1, len(rows))
    bar_w = group_w / max(1, len(columns) + 1)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="32" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-40}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<text x="18" y="{height/2}" font-family="Arial" font-size="13" transform="rotate(-90 18 {height/2})">{y_label}</text>',
    ]
    for r, row_name in enumerate(rows):
        for c, col in enumerate(columns):
            value = float(df.loc[row_name, col])
            x = margin_left + r * group_w + (c + 0.5) * bar_w
            h = value / max_value * plot_h
            y = height - margin_bottom - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{colors[c % len(colors)]}"/>')
        body.append(f'<text x="{margin_left + r*group_w + group_w/2 - 42:.1f}" y="{height-46}" font-family="Arial" font-size="11">{row_name}</text>')
    for c, col in enumerate(columns):
        lx = margin_left + c * 145
        body.append(f'<rect x="{lx}" y="{height-24}" width="10" height="10" fill="{colors[c % len(colors)]}"/>')
        body.append(f'<text x="{lx+15}" y="{height-15}" font-family="Arial" font-size="12">{col}</text>')
    body.append(f'<text x="{margin_left-30}" y="{margin_top+5}" font-family="Arial" font-size="11">{max_value:.2f}</text>')
    body.append("</svg>")
    ensure_parent(path).write_text("\n".join(body), encoding="utf-8")
    return path
