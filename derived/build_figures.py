#!/usr/bin/env python3
"""Rebuild the three manuscript figures from the portable release bundle."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
COLORS = {
    "Luna 5.6 API": "#0072B2", "Luna 6 API": "#D55E00",
    "Sol CLI": "#009E73", "Opus CLI": "#CC79A7",
    "pathway": "#7570B3", "ink": "#25313C", "muted": "#66717C",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    ax.set(xlim=(0, 12.0), ylim=(0, 5.0))
    ax.axis("off")
    boxes = [
        (0.35, 2.45, 2.00, 1.05, "Candidate or\nprocessing\nquestion", "#E7F2F8"),
        (2.70, 2.45, 2.00, 1.05, "Thermal\nevent-map\nproposal", "#DFF2EA"),
        (5.05, 2.45, 2.00, 1.05, "Experiment and\nphase assignment", "#F4ECE3"),
        (7.40, 2.45, 2.00, 1.05, "State-aware\nprocessing\ndecision", "#EEE9F5"),
        (9.75, 2.45, 1.55, 1.05, "Design\nupdate", "#E7F2F8"),
    ]
    for x, y, w, h, label, fill in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor=fill, edgecolor="#82909A", linewidth=1.1))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8.6,
                color=COLORS["ink"], weight="semibold")
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2]+0.08, 2.98), (right[0]-0.08, 2.98),
                                     arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#66717C"))
    ax.add_patch(FancyArrowPatch((10.5, 2.35), (1.35, 2.05), connectionstyle="arc3,rad=-0.18",
                                 arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#9AA4AC"))
    ax.add_patch(FancyBboxPatch((1.75, 0.38), 5.25, 1.15, boxstyle="round,pad=0.05,rounding_size=0.08",
                                facecolor="white", edgecolor=COLORS["pathway"], linewidth=2.0))
    ax.text(2.12, 1.24, "MEASURED HERE", fontsize=9.3, weight="bold", color=COLORS["pathway"])
    ax.text(2.12, 0.92, "Closed-book reconstruction of source-qualified", fontsize=8.5,
            weight="semibold", color=COLORS["ink"])
    ax.text(2.12, 0.67, "thermal identities and pathways", fontsize=8.5,
            weight="semibold", color=COLORS["ink"])
    ax.add_patch(FancyArrowPatch((4.35, 1.55), (3.65, 2.38), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=COLORS["pathway"]))
    ax.text(7.55, 1.10, "Proposed downstream workflow", fontsize=9.8, weight="semibold", color=COLORS["ink"])
    ax.text(7.55, 0.69, "Candidate selection, automated experiments,\nprocessing decisions and design updates are\nshown as proposed uses.",
            fontsize=8.6, color=COLORS["muted"], va="center")
    ax.text(0.35, 4.5, "Thermal-pathway reconstruction in an AI-guided workflow",
            fontsize=14, weight="bold", color=COLORS["ink"])
    save(fig, "figure_1_workflow")
    plt.close(fig)


def family_figure() -> None:
    data = read_csv("event_family_recall.csv")
    models = ["Luna 5.6 API", "Luna 6 API", "Sol CLI", "Opus CLI"]
    families = list(dict.fromkeys(row["event_family_label"] for row in data))
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 5.5), sharey=True, gridspec_kw={"wspace": 0.07})
    y = np.arange(len(families))
    for ax, model in zip(axes, models):
        subset = [row for row in data if row["model_label"] == model]
        values = np.array([float(row["recall"])*100 for row in subset])
        ax.barh(y, values, color=COLORS[model], alpha=0.88, height=0.66)
        for yi, row, value in zip(y, subset, values):
            ax.text(min(value+2.0, 86), yi, f"{row['numerator']}/{row['denominator']}",
                    va="center", fontsize=8.2, color=COLORS["ink"])
        ax.set(title=model, xlim=(0, 108), xticks=[0, 50, 100], xlabel="Identity recall (%)")
        ax.title.set(fontsize=11.2, weight="bold", color=COLORS[model])
        ax.grid(axis="x", color="#DCE2E6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, families, fontsize=9.2)
    axes[0].invert_yaxis()
    fig.suptitle("Broad-prompt recovery depends on the thermal event family", x=0.06, ha="left",
                 fontsize=15, weight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.015, "Post hoc descriptive aggregation of three generations per material. Opportunities repeat within materials; small family denominators do not support ranking inference.",
             fontsize=8.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.25, right=0.985, top=0.84, bottom=0.15)
    save(fig, "figure_2_event_families")
    plt.close(fig)


def protocol_figure() -> None:
    protocol = read_csv("paired_protocol_recall.csv")
    conditional = read_csv("conditional_direction_temperature.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [0.9, 1.45]})
    ax = axes[0]
    for offset, row in zip([-0.08, 0.08], protocol):
        model = row["model_label"]
        broad, specified = float(row["broad_recall"])*100, float(row["protocol_recall"])*100
        ax.plot([0, 1], [broad+offset*10, specified+offset*10], marker="o", linewidth=2.2,
                markersize=7, color=COLORS[model], label=model)
        ax.text(1.05, specified+offset*10,
                f"{float(row['difference'])*100:+.1f} pp\n[{float(row['ci95_low'])*100:+.1f}, {float(row['ci95_high'])*100:+.1f}]",
                va="center", fontsize=8.4, color=COLORS[model])
    ax.set_xticks([0, 1], ["Broad", "Protocol-specified"])
    ax.set(xlim=(-0.15, 1.55), ylim=(25, 72), ylabel="Recall on fixed pathway units (%)")
    ax.grid(axis="y", color="#DCE2E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("a  Protocol bundle", loc="left", fontsize=11.5, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.text(-0.12, 26.2, "41 units × 3 generations\n22 materials; paired material bootstrap", fontsize=8.2, color=COLORS["muted"])

    ax = axes[1]
    ext = [row for row in conditional if row["model_label"] in {"Sol CLI", "Opus CLI"}]
    labels, direction, temperature, d_cov, t_cov, d_acc, colors = [], [], [], [], [], [], []
    for row in ext:
        labels.append(f"{row['model_label']}\n{'Melt' if row['event_class']=='terminal_melting' else 'Other'}")
        direction.append(float(row["direction_coverage"])*100)
        temperature.append(float(row["temperature_coverage"])*100)
        d_cov.append(f"{int(row['direction_scorable_n'])}/{int(row['identity_correct_n'])}")
        t_cov.append(f"{int(row['temperature_compatible_n'])}/{int(row['identity_correct_n'])}")
        d_acc.append(f"{int(row['direction_correct_n'])}/{int(row['direction_scorable_n'])}")
        colors.append(COLORS[row["model_label"]])
    x, width = np.arange(len(labels)), 0.34
    ax.bar(x-width/2, direction, width, color=colors, alpha=0.9, label="Direction scorable")
    ax.bar(x+width/2, temperature, width, color=colors, alpha=0.35, hatch="///", edgecolor=colors,
           label="Temperature-field compatible")
    for xi, height, label in zip(x-width/2, direction, d_cov):
        ax.text(xi, height+2, label, ha="center", fontsize=8.2, color=COLORS["ink"])
    for xi, height, label in zip(x+width/2, temperature, t_cov):
        ax.text(xi, height+2, label, ha="center", fontsize=7.7, color=COLORS["muted"])
    ax.set_xticks(x, labels, fontsize=8.7)
    ax.set(ylim=(0, 112), ylabel="Eligible share of identity-correct matches (%)")
    ax.grid(axis="y", color="#DCE2E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("b  Conditional direction and temperature coverage", loc="left", fontsize=11.2, weight="bold")
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    fig.suptitle("Protocol-context and conditional-measurement results", x=0.06, ha="left",
                 fontsize=14.5, weight="bold", color=COLORS["ink"])
    fig.text(0.59, 0.048, "Panel b bar labels are eligible/identity-correct. Direction accuracy (correct/scorable), left to right: " + ", ".join(d_acc) + ".",
             ha="center", fontsize=7.8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.23, wspace=0.3)
    save(fig, "figure_3_protocol_and_coverage")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.labelcolor": COLORS["ink"], "xtick.color": COLORS["ink"],
                         "ytick.color": COLORS["ink"]})
    workflow_figure()
    family_figure()
    protocol_figure()


if __name__ == "__main__":
    main()
