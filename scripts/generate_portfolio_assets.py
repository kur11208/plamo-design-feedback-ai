from __future__ import annotations

from pathlib import Path
import sys
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app import (
    _issue_category_label,
    _part_area_label,
    analyze_feedback_dataframe,
    build_priority_ranking_dataframe,
    build_result_dataframe,
    build_runner_image_report,
    most_common_issue_category,
)
from image_based_analyzer import build_image_based_findings
from part_visualizer import plot_part_risk_map


DATA_PATH = ROOT_DIR / "data" / "sample_feedback.csv"
ASSETS_DIR = ROOT_DIR / "assets"
SCREENSHOTS_DIR = ROOT_DIR / "docs" / "screenshots"
OUTPUTS_DIR = ROOT_DIR / "outputs"


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    feedback_df = pd.read_csv(DATA_PATH, encoding="utf-8")
    records = analyze_feedback_dataframe(feedback_df)
    result_df = build_result_dataframe(records)
    priority_df = build_priority_ranking_dataframe(records)

    _save_risk_map(records, "runner_state", ASSETS_DIR / "runner_sample.png")
    shutil.copyfile(ASSETS_DIR / "runner_sample.png", SCREENSHOTS_DIR / "risk_map_runner_view.png")
    _save_dashboard_overview(records, result_df, priority_df, SCREENSHOTS_DIR / "dashboard.png")
    _save_image_based_report(records, OUTPUTS_DIR / "image_based_analysis.md")


def _save_risk_map(records: list[dict], inspection_phase: str, output_path: Path) -> None:
    fig = plot_part_risk_map(records, inspection_phase)
    fig.write_image(str(output_path), scale=2)


def _save_image_based_report(records: list[dict], output_path: Path) -> None:
    runner_findings = build_image_based_findings(records, "runner_state")
    report = build_runner_image_report(runner_findings, "assets/runner_sample.png")
    output_path.write_text(report, encoding="utf-8")


def _save_dashboard_overview(
    records: list[dict],
    result_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 9), facecolor="#F8FAFC")
    fig.suptitle("Plamo Design Feedback AI - Development Improvement Dashboard", fontsize=18, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.925,
        "Synthetic portfolio screenshot. No official product image or design data is used.",
        ha="center",
        fontsize=9,
        color="#64748B",
    )

    metric_values = [
        ("Total Feedback", str(len(records))),
        ("High Risk", str(sum(1 for record in records if int(record["risk_score"]) >= 70))),
        ("Top Priority Part", str(priority_df.iloc[0]["part_area_label"])),
        ("Main Issue", _issue_category_label(most_common_issue_category(records))),
        ("Avg Risk", f"{result_df['risk_score'].mean():.1f}"),
    ]
    for index, (label, value) in enumerate(metric_values):
        x = 0.04 + index * 0.19
        fig.patches.append(Rectangle((x, 0.79), 0.17, 0.1, transform=fig.transFigure, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0))
        fig.text(x + 0.015, 0.855, label, fontsize=9, color="#475569")
        fig.text(x + 0.015, 0.812, value, fontsize=16, weight="bold", color="#111827")

    priority_ax = fig.add_axes((0.04, 0.47, 0.43, 0.25))
    priority_ax.axis("off")
    priority_ax.set_title("Improvement Priority Ranking", loc="left", fontsize=12, weight="bold")
    table_df = priority_df.head(5)[
        ["part_area_label", "feedback_count", "average_risk_score", "max_risk_score", "priority_level"]
    ]
    table = priority_ax.table(
        cellText=table_df.values,
        colLabels=["部位", "件数", "平均", "最大", "優先度"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)

    part_ax = fig.add_axes((0.55, 0.47, 0.39, 0.25))
    part_scores = result_df.groupby("part_area")["risk_score"].mean().sort_values(ascending=False).head(8)
    part_scores.index = [_part_area_label(str(part_area)) for part_area in part_scores.index]
    part_scores.plot(kind="bar", ax=part_ax, color="#4C78A8")
    part_ax.set_title("部位別 平均リスク", loc="left", fontsize=12, weight="bold")
    part_ax.set_xlabel("")
    part_ax.set_ylabel("score")
    part_ax.set_ylim(0, 100)
    part_ax.grid(axis="y", linestyle="--", alpha=0.35)
    part_ax.tick_params(axis="x", labelrotation=35, labelsize=8)

    category_ax = fig.add_axes((0.04, 0.12, 0.43, 0.25))
    category_counts: dict[str, int] = {}
    for record in records:
        for category in record["issue_categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
    category_series = pd.Series(category_counts).sort_values(ascending=False).head(8)
    category_series.index = [_issue_category_label(str(category)) for category in category_series.index]
    category_series.plot(kind="bar", ax=category_ax, color="#F58518")
    category_ax.set_title("問題カテゴリ別 件数", loc="left", fontsize=12, weight="bold")
    category_ax.set_xlabel("")
    category_ax.set_ylabel("count")
    category_ax.grid(axis="y", linestyle="--", alpha=0.35)
    category_ax.tick_params(axis="x", labelrotation=35, labelsize=8)

    feature_ax = fig.add_axes((0.55, 0.12, 0.39, 0.25))
    feature_ax.axis("off")
    feature_ax.set_title("Score Explanation Example", loc="left", fontsize=12, weight="bold")
    top_record = max(records, key=lambda record: int(record["risk_score"]))
    breakdown = top_record["risk_explanation"]["breakdown"][:5]
    y = 0.85
    feature_ax.text(0.0, y, f"{top_record['feedback_id']} / {_part_area_label(str(top_record['part_area']))} / risk {top_record['risk_score']}", fontsize=10, weight="bold")
    for item in breakdown:
        y -= 0.14
        factor = str(item.get("factor", ""))
        label = _issue_category_label(factor) if item.get("factor_type") == "issue_category" else factor
        feature_ax.text(0.0, y, f"+{item['points']}  {label}", fontsize=9, color="#111827")

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
