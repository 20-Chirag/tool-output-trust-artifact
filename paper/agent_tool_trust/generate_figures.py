"""Generate the short-paper figures with the vendored ORX style.

Regenerate from the repository root with:
    .venv\\Scripts\\python.exe paper\\agent_tool_trust\\generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orx_figstyle import (
    BASELINE,
    PALETTE,
    TEXT,
    figure,
    figure_grid,
    panel_labels,
    save,
    use_style,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def corruption_modes() -> None:
    data = read_json(
        ROOT / "outputs" / "clean_web_diagnostic" / "corruption_mode_metrics.json"
    )
    rows = [row for row in data["rows"] if row["policy_name"] == "baseline"]
    by_key = {(row["model_name"], row["corruption_mode"]): row for row in rows}

    models = [
        ("gemini-2.5-flash", "Gemini 2.5\nFlash"),
        ("gpt-5.4-mini", "GPT-5.4\nmini"),
        ("gemini-3.1-pro-preview", "Gemini 3.1\nPro"),
    ]
    modes = [
        ("plausible_hallucinated_content", "Same-entity false value", PALETTE["blue"], "o", -0.11),
        ("wrong_user_or_wrong_entity", "Wrong entity", PALETTE["orange"], "D", 0.11),
    ]

    use_style()
    fig, ax = figure(width=TEXT, ratio=0.40)
    x = np.arange(len(models))

    for mode, label, color, marker, offset in modes:
        estimates, lower, upper, labels, ci_highs = [], [], [], [], []
        for model, _ in models:
            metric = by_key[(model, mode)]["silent_acceptance"]
            estimate = metric["estimate"]
            estimates.append(estimate)
            lower.append(estimate - metric["ci_low"])
            upper.append(metric["ci_high"] - estimate)
            labels.append(f'{metric["numerator"]}/{metric["denominator"]}')
            ci_highs.append(metric["ci_high"])

        positions = x + offset
        ax.errorbar(
            positions,
            estimates,
            yerr=np.asarray([lower, upper]),
            fmt=marker,
            linestyle="none",
            markersize=5.3,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.45,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            label=label,
            zorder=3,
            clip_on=False,
        )
        for position, count, ci_high in zip(positions, labels, ci_highs):
            ax.annotate(
                count,
                xy=(position, ci_high),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#222222",
            )

    ax.axhline(0, color=BASELINE, linewidth=0.7, zorder=1)
    ax.set_ylabel("Silent acceptance rate")
    ax.set_xticks(x, [label for _, label in models])
    ax.set_ylim(-0.04, 0.88)
    ax.set_yticks(np.arange(0, 0.9, 0.2))
    ax.legend(loc="upper right", frameon=True, framealpha=1.0, edgecolor="#777777")
    save(fig, str(OUT / "corruption_modes"), formats=("pdf", "svg", "png"))


def _add_count_label(
    ax,
    x: float,
    y: float,
    count: str,
    *,
    dx: float = 0,
    dy: float = 5,
    va: str = "bottom",
) -> None:
    ax.annotate(
        count,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=6.5,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.3},
        zorder=5,
    )


def url_cue() -> None:
    rows = read_json(ROOT / "outputs" / "url_fixture_ablation_run" / "paired_deltas.json")
    selected = {
        row["policy_name"]: row
        for row in rows
        if row["model_name"] == "gpt-5.4-mini"
        and row["policy_name"] in {"m2_structured_skeptic", "m6_checklist"}
    }
    policies = [("m2_structured_skeptic", "M2"), ("m6_checklist", "M6")]
    metrics = [
        ("silent_acceptance", "Silent acceptance rate"),
        ("false_defer_or_verify", "False defer / verify rate"),
    ]

    use_style()
    fig, axes = figure_grid(nrows=1, ncols=2, width=TEXT, ratio=0.42, sharey=True)
    x = np.arange(len(policies))
    offset = 0.13
    series = [
        ("real", "Real domain", PALETTE["blue"], "o", -offset),
        ("fixture", "Placeholder host (example.org)", PALETTE["orange"], "D", offset),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        values: dict[str, list[float]] = {"real": [], "fixture": []}
        counts: dict[str, list[str]] = {"real": [], "fixture": []}
        for key, _ in policies:
            record = selected[key]["metrics"][metric]
            for arm in ("real", "fixture"):
                values[arm].append(record[arm]["rate"])
                counts[arm].append(
                    f'{record[arm]["numerator"]}/{record[arm]["denominator"]}'
                )

        for i in range(len(policies)):
            ax.plot(
                [x[i] - offset, x[i] + offset],
                [values["real"][i], values["fixture"][i]],
                color="#8A8A8A",
                linewidth=0.85,
                zorder=1,
            )

        for arm, label, color, marker, arm_offset in series:
            ax.plot(
                x + arm_offset,
                values[arm],
                linestyle="none",
                marker=marker,
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.45,
                color=color,
                label=label,
                zorder=3,
            )
            for xpos, value, count in zip(x + arm_offset, values[arm], counts[arm]):
                if metric == "false_defer_or_verify" and arm == "fixture":
                    _add_count_label(ax, xpos, value, count, dy=-7, va="top")
                else:
                    _add_count_label(ax, xpos, value, count)

        for i, (key, _) in enumerate(policies):
            record = selected[key]["metrics"][metric]
            delta = record["delta_fixture_minus_real"]
            low = record["ci_low"]
            high = record["ci_high"]
            ax.annotate(
                f'$\\Delta$={delta:+.2f}\n[{low:+.2f}, {high:+.2f}]',
                xy=(x[i], 0.80),
                ha="center",
                va="top",
                fontsize=6.2,
                color="#222222",
            )

        ax.axhline(0, color=BASELINE, linewidth=0.7, zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [label for _, label in policies])
        ax.set_ylim(-0.04, 0.86)
        ax.set_yticks(np.arange(0, 0.9, 0.2))
        ax.tick_params(axis="x", length=0)

    panel_labels(axes, labels=["(a) Unreliable outputs", "(b) Reliable outputs"])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2)
    save(fig, str(OUT / "url_cue_ablation"), formats=("pdf", "svg", "png"))


def main() -> None:
    corruption_modes()
    url_cue()


if __name__ == "__main__":
    main()
