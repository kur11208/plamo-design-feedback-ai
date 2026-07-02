from __future__ import annotations

from pathlib import Path
import sys
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFilter

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

    _generate_runner_background(ASSETS_DIR / "runner_bg_realistic.png")
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


def _generate_runner_background(output_path: Path) -> None:
    """Generate a fictional runner sheet whose A1 part is visibly a V antenna."""

    width, height, scale = 1672, 941, 2
    image = Image.new("RGB", (width * scale, height * scale), "#E9ECEF")
    draw = ImageDraw.Draw(image)

    def pt(x: float, y: float) -> tuple[int, int]:
        px = int((x - 0.30) / 9.40 * width * scale)
        py = int((8.10 - y) / 5.30 * height * scale)
        return px, py

    def line(points: list[tuple[float, float]], color: str = "#414A52", width_px: int = 15) -> None:
        draw.line([pt(x, y) for x, y in points], fill=color, width=width_px * scale, joint="curve")

    def poly(points: list[tuple[float, float]], fill: str = "#68717A", outline: str = "#2E353B") -> None:
        draw.polygon([pt(x, y) for x, y in points], fill=fill, outline=outline)

    def ellipse(cx: float, cy: float, rx: float, ry: float, fill: str = "#59636C", outline: str = "#2E353B") -> None:
        x0, y0 = pt(cx - rx, cy + ry)
        x1, y1 = pt(cx + rx, cy - ry)
        draw.ellipse([x0, y0, x1, y1], fill=fill, outline=outline, width=2 * scale)

    def rect(x0: float, y0: float, x1: float, y1: float, fill: str = "#68717A", outline: str = "#2E353B") -> None:
        a = pt(x0, y1)
        b = pt(x1, y0)
        draw.rounded_rectangle([a, b], radius=8 * scale, fill=fill, outline=outline, width=2 * scale)

    # Subtle surface and frame.
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [pt(0.72, 7.78), pt(9.32, 3.02)],
        radius=46 * scale,
        fill=(0, 0, 0, 58),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(8 * scale))
    image = Image.alpha_composite(image.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Runner frame and internal sprues.
    line([(0.92, 7.45), (9.12, 7.45), (9.32, 7.25), (9.32, 3.22), (9.10, 3.02), (0.92, 3.02), (0.72, 3.22), (0.72, 7.25), (0.92, 7.45)], width_px=22)
    for y in (6.15, 4.85, 3.85):
        line([(0.82, y), (9.22, y)], width_px=14)
    for x in (1.35, 2.45, 3.95, 5.00, 6.10, 7.25, 8.35):
        line([(x, 7.38), (x, 3.10)], width_px=12)
    line([(5.00, 6.15), (5.35, 5.55), (5.00, 4.85)], width_px=13)
    ellipse(5.35, 5.50, 0.28, 0.28, fill="#E9ECEF", outline="#414A52")

    # A1: fictional V antenna with tip gates.
    poly([(1.28, 5.45), (1.48, 5.45), (1.42, 6.12), (0.98, 7.14), (0.86, 7.04), (1.18, 5.82)], fill="#6B737C")
    poly([(1.76, 5.45), (1.96, 5.45), (1.86, 5.82), (2.18, 7.04), (2.04, 7.14), (1.62, 6.12)], fill="#6B737C")
    poly([(1.50, 5.52), (1.60, 6.85), (1.70, 5.52)], fill="#757E86")
    rect(1.15, 5.35, 2.00, 5.55, fill="#555E67")
    line([(0.98, 7.14), (0.98, 7.45)], width_px=7)
    line([(2.18, 7.04), (2.18, 7.45)], width_px=7)

    # A2: hand parts, two small clustered pieces.
    for cx in (3.05, 3.72):
        ellipse(cx, 6.70, 0.18, 0.22)
        rect(cx - 0.12, 6.22, cx + 0.12, 6.58, fill="#5D6670")
        for dx in (-0.18, -0.06, 0.06, 0.18):
            ellipse(cx + dx, 6.98, 0.035, 0.10, fill="#6E7780")
        line([(cx, 6.15), (cx, 6.32)], width_px=7)

    # A3: grip / tool-like long part.
    rect(4.25, 5.90, 4.55, 7.12, fill="#5D6670")
    ellipse(4.40, 7.18, 0.14, 0.14)
    rect(4.70, 6.70, 5.10, 7.06, fill="#68717A")
    ellipse(4.83, 6.88, 0.09, 0.09, fill="#E9ECEF")

    # B1 / filler armor panels.
    poly([(6.28, 6.82), (6.60, 7.25), (6.90, 7.12), (6.98, 6.10), (6.54, 5.80), (6.20, 6.10)])
    poly([(7.18, 6.10), (7.55, 6.90), (7.86, 6.82), (7.74, 5.78), (7.28, 5.80)])

    # B2: backpack block with nozzles.
    poly([(8.10, 5.92), (8.85, 5.92), (9.00, 6.28), (8.86, 7.04), (8.12, 7.04), (7.96, 6.28)], fill="#6A737C")
    rect(8.22, 6.28, 8.74, 6.82, fill="#5B646D")
    ellipse(8.22, 5.95, 0.13, 0.18, fill="#303841")
    ellipse(8.72, 5.95, 0.13, 0.18, fill="#303841")

    # B3: gate-mark inspection armor panel.
    poly([(0.98, 3.65), (1.28, 3.50), (1.98, 3.50), (2.22, 3.85), (2.12, 4.72), (1.68, 5.05), (1.12, 4.85), (0.90, 4.25)], fill="#6A737C")
    poly([(1.18, 3.92), (1.78, 3.85), (1.88, 4.56), (1.34, 4.76), (1.08, 4.34)], fill="#5B646D")
    line([(0.88, 4.50), (0.72, 4.50)], width_px=7)
    line([(2.05, 4.80), (2.45, 4.85)], width_px=7)
    line([(2.10, 3.78), (2.45, 3.85)], width_px=7)

    # Lower-row small filler parts.
    for cx in (2.75, 3.35, 4.02, 4.62, 5.78, 6.18):
        rect(cx - 0.12, 3.46, cx + 0.12, 3.88, fill="#59636C")
        ellipse(cx, 4.05, 0.10, 0.10)
        line([(cx, 3.85), (cx, 4.85)], width_px=7)
    for cx in (7.55, 7.95):
        rect(cx - 0.08, 3.22, cx + 0.08, 5.80, fill="#68717A")

    # Embossed runner labels.
    for label, x, y in [("A1", 0.95, 7.22), ("A2", 2.66, 7.24), ("A3", 4.50, 7.30), ("B1", 6.48, 7.22), ("B2", 8.15, 7.20), ("B3", 0.92, 4.66)]:
        px, py = pt(x, y)
        draw.rounded_rectangle([px - 20 * scale, py - 14 * scale, px + 34 * scale, py + 12 * scale], radius=5 * scale, fill="#6D7680", outline="#515B64", width=1 * scale)
        draw.text((px - 14 * scale, py - 10 * scale), label, fill="#2F3740")

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


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
