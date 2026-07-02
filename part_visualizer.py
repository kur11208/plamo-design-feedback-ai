"""Inspection-focused visualizations for fictional plamo feedback.

Fictional mecha design only — no real product, trademark, or official design data.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

_ASSETS_DIR = Path(__file__).parent / "assets"


def _load_bg_image(filename: str) -> str | None:
    path = _ASSETS_DIR / filename
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(path.suffix.lower(), "image/png")
    return f"data:{mime_type};base64,{data}"


RUNNER_PARTS = ("antenna", "hand_parts", "weapon_grip", "backpack", "gate_area", "instruction_step")
ASSEMBLED_PARTS = (
    "shoulder_joint",
    "elbow_joint",
    "waist_joint",
    "leg_joint",
    "hand_parts",
    "weapon_grip",
    "backpack",
)

PART_LABELS = {
    "shoulder_joint": "shoulder_joint",
    "elbow_joint": "elbow_joint",
    "waist_joint": "waist_joint",
    "antenna": "antenna",
    "backpack": "backpack",
    "hand_parts": "hand_parts",
    "weapon_grip": "weapon_grip",
    "leg_joint": "leg_joint",
    "gate_area": "gate_area",
    "instruction_step": "instruction_step",
}

RUNNER_PART_LAYOUT = {
    "antenna": {
        "part_no": "A1",
        "center": (1.58, 6.48),
        "gate_points": [(1.04, 7.16), (2.12, 7.16)],
        "ann": {"ax": -55, "ay": -40},
    },
    "hand_parts": {
        "part_no": "A2",
        "center": (3.38, 6.88),
        "gate_points": [(2.45, 6.88), (3.90, 6.88)],
        "ann": {"ax": -60, "ay": -42},
    },
    "weapon_grip": {
        "part_no": "A3",
        "center": (4.47, 6.91),
        "gate_points": [(4.34, 6.08), (5.12, 6.98)],
        "ann": {"ax": 58, "ay": -42},
    },
    "backpack": {
        "part_no": "B2",
        "center": (8.70, 6.65),
        "gate_points": [(8.17, 6.50), (9.25, 6.50)],
        "ann": {"ax": 65, "ay": -42},
    },
    "gate_area": {
        "part_no": "B3",
        "center": (1.34, 4.45),
        "gate_points": [(0.92, 4.48), (2.02, 4.80), (2.05, 4.08)],
        "ann": {"ax": -60, "ay": 42},
    },
}

ASSEMBLED_BG_BOUNDS = {"x": 0.70, "y": 8.80, "w": 8.60, "h": 5.73}

ASSEMBLED_PART_LAYOUT = {
    "shoulder_joint": {"fx": 0.407, "fy": 0.188, "shape": "circle", "ann": {"ax": -92, "ay": -28}},
    "elbow_joint":    {"fx": 0.337, "fy": 0.318, "shape": "circle", "ann": {"ax": -86, "ay": -18}},
    "waist_joint":    {"fx": 0.500, "fy": 0.405, "shape": "square", "ann": {"ax": 0,   "ay": 64}},
    "leg_joint":      {"fx": 0.397, "fy": 0.606, "shape": "circle", "ann": {"ax": -82, "ay": 44}},
    "hand_parts":     {"fx": 0.324, "fy": 0.500, "shape": "circle", "ann": {"ax": -88, "ay": 34}},
    "weapon_grip":    {"x": 8.35, "y": 4.75, "shape": "square", "external": True, "note": "別パーツ保持部"},
    "backpack":       {"x": 8.35, "y": 5.85, "shape": "square", "external": True, "note": "背面/別視点"},
}

ASSEMBLED_CONNECTIONS = [
    ("shoulder_joint", "waist_joint"),
    ("shoulder_joint", "elbow_joint"),
    ("elbow_joint", "hand_parts"),
    ("waist_joint", "leg_joint"),
]

RISK_COLORS = {
    "high":   "#E53E3E",
    "medium": "#F6AD55",
    "low":    "#48BB78",
    "none":   "#718096",
}

_BG      = "#0E1117"
_SURFACE = "#1A1F2E"
_BORDER  = "#2D3748"
_TEXT    = "#E2E8F0"
_MUTED   = "#718096"

_DISCLAIMER = "架空の検査模式図です。実在商品画像・公式デザインデータは使用していません。"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_runner_inspection_map(
    records: Sequence[Mapping[str, Any]],
    highlight_part_area: str | None = None,
) -> go.Figure:
    """Runner sheet inspection map with mecha-style part shapes."""

    summary = _summarize_by_part(records, "runner_state")
    fig = go.Figure()
    _apply_base_layout(fig, "切り出し前リスクマップ",
                       "ゲート位置・小型部品・切り出し時の破損リスクを架空ランナー模式図上で可視化します。")

    bg_src = _load_bg_image("runner_bg_realistic.png")
    if bg_src:
        fig.add_layout_image(
            source=bg_src,
            xref="x", yref="y",
            x=0.30, y=8.10,
            sizex=9.40, sizey=5.30,
            xanchor="left", yanchor="top",
            opacity=0.82,
            layer="below",
            sizing="stretch",
        )

    if not bg_src:
        _draw_runner_frame(fig)
        _draw_sprue_network(fig)

    for part_area, layout in RUNNER_PART_LAYOUT.items():
        info = summary.get(part_area, {})
        risk_score  = info.get("average_risk_score")
        feedback_count = int(info.get("feedback_count", 0))
        issue_cats  = info.get("issue_categories", [])
        gate_pos    = info.get("gate_position", "hidden")
        is_hl       = highlight_part_area == part_area
        has_feedback = risk_score is not None
        cx, cy      = _point(layout["center"])
        color       = _risk_color(risk_score)
        score_text  = f"{risk_score:.0f}" if has_feedback else "参考"

        if bg_src and part_area == "antenna":
            _draw_runner_antenna_overlay(fig, cx, cy, color)
        elif not bg_src:
            _draw_runner_part_shape(fig, part_area, cx, cy, color)
        _draw_gate_markers(fig, layout["gate_points"], issue_cats, gate_pos, active=has_feedback or is_hl)

        if is_hl:
            fig.add_shape(type="circle",
                          x0=cx - 0.95, y0=cy - 0.95, x1=cx + 0.95, y1=cy + 0.95,
                          fillcolor="rgba(0,0,0,0)",
                          line=dict(color="#FC8181", width=2.2, dash="dash"))

        if risk_score is not None and risk_score >= 70:
            fig.add_shape(type="circle",
                          x0=cx - 0.75, y0=cy - 0.75, x1=cx + 0.75, y1=cy + 0.75,
                          fillcolor="rgba(229,62,62,0.07)",
                          line=dict(color="#E53E3E", width=1.5, dash="dot"))

        # Invisible hover marker
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(size=44, color="rgba(0,0,0,0)", symbol="circle"),
            hovertemplate=(
                f"<b>{layout['part_no']} {part_area}</b><br>"
                f"risk: {score_text}<br>feedback: {feedback_count}<br>gate: {gate_pos}<br>"
                f"categories: {', '.join(issue_cats) or '-'}<extra></extra>"
            ),
            showlegend=False,
        ))

        if not has_feedback and not is_hl:
            # The background image contains this part, but the current phase has
            # no feedback for it. Keep it available in hover only so the map
            # does not pretend to score an unobserved part.
            continue

        # Arrow label
        ann = layout.get("ann", {"ax": 0, "ay": -62})
        flag = "<br><span style='color:#FC8181'>▶ selected</span>" if is_hl else ""
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{layout['part_no']}</b> {part_area}<br>risk <b>{score_text}</b>{flag}",
            showarrow=True, ax=ann["ax"], ay=ann["ay"],
            axref="pixel", ayref="pixel",
            arrowhead=2, arrowwidth=1.3, arrowcolor=_MUTED,
            bgcolor=_SURFACE, bordercolor=_BORDER,
            borderwidth=1, borderpad=5,
            font=dict(size=8.5, color=_TEXT),
            xref="x", yref="y",
        )

    instr = summary.get("instruction_step", {})
    _add_manual_note_card(fig, instr.get("average_risk_score"),
                          instr.get("main_issue_category", "assembly_difficulty"),
                          highlight=highlight_part_area == "instruction_step")
    _add_risk_legend(fig)
    return fig


def plot_assembled_inspection_map(
    records: Sequence[Mapping[str, Any]],
    highlight_part_area: str | None = None,
) -> go.Figure:
    """Post-assembly inspection map with fictional mecha silhouette."""

    summary = _summarize_by_part(records, "assembled_state")
    fig = go.Figure()
    _apply_base_layout(fig, "組み立て後リスクマップ",
                       "関節の固さ・保持力不足・ポージング安定性を可視化します。")
    _focus_assembled_layout(fig)

    bg_src = _load_bg_image("assembled_bg_original.png")
    if bg_src:
        bounds = ASSEMBLED_BG_BOUNDS
        fig.add_layout_image(
            source=bg_src,
            xref="x", yref="y",
            x=bounds["x"], y=bounds["y"],
            sizex=bounds["w"], sizey=bounds["h"],
            xanchor="left", yanchor="top",
            opacity=0.42,
            layer="below",
            sizing="stretch",
        )
    else:
        _draw_mecha_silhouette(fig)

    # Skeleton lines (on top of silhouette)
    for part_a, part_b in ASSEMBLED_CONNECTIONS:
        la = ASSEMBLED_PART_LAYOUT.get(part_a)
        lb = ASSEMBLED_PART_LAYOUT.get(part_b)
        if la and lb:
            ax, ay = _assembled_layout_point(la)
            bx, by = _assembled_layout_point(lb)
            fig.add_trace(go.Scatter(
                x=[ax, bx],
                y=[ay, by],
                mode="lines",
                line=dict(color="rgba(160,180,220,0.35)", width=1.8),
                hoverinfo="skip", showlegend=False,
            ))

    for part_area in ASSEMBLED_PARTS:
        layout     = ASSEMBLED_PART_LAYOUT[part_area]
        info       = summary.get(part_area, {})
        risk_score = info.get("average_risk_score")
        feedback_count = int(info.get("feedback_count", 0))
        main_cat   = info.get("main_issue_category", "n/a")
        issue_cats = info.get("issue_categories", [])
        color      = _risk_color(risk_score)
        x, y       = _assembled_layout_point(layout)
        shape      = "square" if layout.get("shape") in {"rect", "square"} else "circle"
        is_hl      = highlight_part_area == part_area
        score_text = f"{risk_score:.0f}" if risk_score is not None else "n/a"
        msize      = 32 if (risk_score or 0) >= 70 else 24 if (risk_score or 0) >= 40 else 18

        if layout.get("external"):
            _add_assembled_external_card(
                fig,
                part_area,
                risk_score,
                main_cat,
                str(layout.get("note", "画像外確認")),
                x=x,
                y=y,
                highlight=is_hl,
            )
            continue

        if risk_score is not None and risk_score >= 70:
            fig.add_shape(type="circle",
                          x0=x - 0.50, y0=y - 0.50, x1=x + 0.50, y1=y + 0.50,
                          fillcolor="rgba(229,62,62,0.12)",
                          line=dict(color="#E53E3E", width=1.8))

        if is_hl:
            fig.add_shape(type="circle",
                          x0=x - 0.62, y0=y - 0.62, x1=x + 0.62, y1=y + 0.62,
                          fillcolor="rgba(0,0,0,0)",
                          line=dict(color="#FC8181", width=2.2, dash="dash"))

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=msize, color=color, opacity=0.95,
                        line=dict(color="#E2E8F0", width=1.5), symbol=shape),
            hovertemplate=(
                f"<b>{part_area}</b><br>risk: {score_text}<br>"
                f"feedback: {feedback_count}<br>"
                f"main: {main_cat}<br>"
                f"categories: {', '.join(issue_cats) or '-'}<extra></extra>"
            ),
            showlegend=False,
        ))

        ann  = layout.get("ann", {"ax": 72, "ay": -28})
        flag = "<br><span style='color:#FC8181'>▶ selected</span>" if is_hl else ""
        fig.add_annotation(
            x=x, y=y,
            text=(f"<b>{part_area}</b><br>"
                  f"risk <b>{score_text}</b><br>"
                  f"<span style='color:{_MUTED}'>{main_cat}</span>{flag}"),
            showarrow=True, ax=ann["ax"], ay=ann["ay"],
            axref="pixel", ayref="pixel",
            arrowhead=2, arrowwidth=1.3, arrowcolor=_MUTED,
            bgcolor=_SURFACE, bordercolor=_BORDER,
            borderwidth=1, borderpad=5,
            font=dict(size=8.5, color=_TEXT),
            xref="x", yref="y",
        )

    _add_risk_legend(fig)
    return fig


def plot_feedback_target(record: Mapping[str, Any]) -> go.Figure:
    part_area = str(record.get("part_area", "unknown"))
    if str(record.get("inspection_phase", "assembled_state")) == "runner_state":
        return plot_runner_inspection_map([record], highlight_part_area=part_area)
    return plot_assembled_inspection_map([record], highlight_part_area=part_area)


def plot_part_risk_map(records: Sequence[Mapping[str, Any]],
                       inspection_phase: str = "assembled_state") -> go.Figure:
    if inspection_phase == "runner_state":
        return plot_runner_inspection_map(records)
    return plot_assembled_inspection_map(records)


def build_fix_point_rows(record: Mapping[str, Any]) -> list[dict[str, str]]:
    risk_score  = int(record.get("risk_score", 0))
    suggestions = record.get("improvement_suggestions", {})
    if not isinstance(suggestions, Mapping):
        return []
    rows: list[dict[str, str]] = []
    for viewpoint, values in suggestions.items():
        for value in _as_list(values):
            rows.append({"優先度": _priority_label(risk_score), "観点": str(viewpoint),
                         "修正ポイント": value, "確認方法": _verification_hint(str(viewpoint))})
    return rows[:8]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _apply_base_layout(fig: go.Figure, title: str, subtitle: str) -> None:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16, color=_TEXT,
                   family="Yu Gothic, Meiryo, sans-serif"),
                   x=0.5, xanchor="center"),
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_BG,
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[0, 9.5], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(orientation="h", x=1.0, y=0.0, xanchor="right", yanchor="top",
                    font=dict(size=10, color=_TEXT),
                    bgcolor="rgba(26,31,46,0.9)",
                    bordercolor=_BORDER, borderwidth=1),
        margin=dict(l=10, r=10, t=62, b=42),
        height=580,
        hovermode="closest",
        hoverlabel=dict(bgcolor=_SURFACE, font_size=12,
                        font_family="Yu Gothic, Meiryo, sans-serif",
                        bordercolor=_BORDER, font_color=_TEXT),
    )
    fig.add_annotation(x=5, y=9.15, text=subtitle, showarrow=False,
                       font=dict(size=10, color=_MUTED),
                       xref="x", yref="y", xanchor="center")
    fig.add_annotation(x=5, y=0.22, text=_DISCLAIMER, showarrow=False,
                       font=dict(size=8, color="#4A5568"),
                       xref="x", yref="y", xanchor="center")


def _focus_assembled_layout(fig: go.Figure) -> None:
    fig.update_layout(
        xaxis=dict(range=[0.20, 9.80], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[2.45, 9.30], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
    )
    if len(fig.layout.annotations) >= 2:
        fig.layout.annotations[-2].update(y=9.08)
        fig.layout.annotations[-1].update(y=2.60)


def _draw_runner_frame(fig: go.Figure) -> None:
    fig.add_shape(type="rect", x0=0.55, y0=1.05, x1=9.45, y1=8.55,
                  fillcolor="rgba(26,31,46,0)",
                  line=dict(color=_BORDER, width=2.5))
    fig.add_annotation(x=0.75, y=1.30, text="架空ランナー A",
                       showarrow=False, xanchor="left",
                       font=dict(size=9, color=_MUTED), xref="x", yref="y")


def _draw_sprue_network(fig: go.Figure) -> None:
    sprue = dict(color="rgba(90,105,130,0.70)")
    branch = dict(color="rgba(110,125,150,0.80)")
    # Main horizontal bars
    fig.add_shape(type="line", x0=1.0, y0=6.15, x1=9.0, y1=6.15,
                  line=dict(**sprue, width=7))
    fig.add_shape(type="line", x0=1.2, y0=3.20, x1=8.8, y1=3.20,
                  line=dict(**sprue, width=6))
    # Vertical spine
    fig.add_shape(type="line", x0=5.0, y0=1.40, x1=5.0, y1=8.35,
                  line=dict(**sprue, width=6))
    # Branches
    for (x1, y1), (x2, y2) in [
        ((2.0, 7.05), (2.0, 6.15)),
        ((3.7, 7.00), (3.7, 6.15)),
        ((4.3, 7.00), (4.55, 6.15)),
        ((5.8, 6.80), (5.2, 6.15)),
        ((6.7, 6.80), (6.3, 6.15)),
        ((7.2, 7.10), (7.0, 6.15)),
        ((8.4, 7.20), (8.2, 6.15)),
        ((2.4, 3.80), (2.4, 3.20)),
        ((3.5, 4.20), (3.5, 3.20)),
    ]:
        fig.add_shape(type="line", x0=x1, y0=y1, x1=x2, y1=y2,
                      line=dict(**branch, width=2.5))


def _draw_runner_part_shape(fig: go.Figure, part_area: str,
                            cx: float, cy: float, color: str) -> None:
    """Draw a mecha-part-like shape for each runner part."""

    fill = color
    edge = dict(color="rgba(200,210,230,0.55)", width=1.0)
    acc  = dict(color="rgba(160,180,210,0.40)", width=0.7)

    def p(path: str, *, accent: bool = False) -> None:
        fig.add_shape(type="path", path=path,
                      fillcolor=fill if not accent else "rgba(45,55,80,0.70)",
                      line=edge if not accent else acc)

    if part_area == "antenna":
        # V-fin pieces: two angular fins + base
        p(f"M {cx-0.62} {cy-0.18} L {cx-0.10} {cy+0.62} L {cx+0.08} {cy+0.10} L {cx-0.18} {cy-0.18} Z")
        p(f"M {cx+0.62} {cy-0.18} L {cx+0.10} {cy+0.62} L {cx-0.08} {cy+0.10} L {cx+0.18} {cy-0.18} Z")
        p(f"M {cx-0.08} {cy+0.10} L {cx} {cy+0.50} L {cx+0.08} {cy+0.10} Z")
        p(f"M {cx-0.25} {cy-0.35} L {cx+0.25} {cy-0.35} L {cx+0.22} {cy-0.18} L {cx-0.22} {cy-0.18} Z", accent=True)

    elif part_area == "hand_parts":
        # Forearm outer piece + wrist joint disc + knuckle detail
        p(f"M {cx-0.38} {cy-0.42} L {cx+0.38} {cy-0.42} L {cx+0.30} {cy+0.38} L {cx-0.30} {cy+0.38} Z")
        p(f"M {cx-0.22} {cy+0.38} L {cx+0.22} {cy+0.38} L {cx+0.18} {cy+0.60} L {cx-0.18} {cy+0.60} Z", accent=True)
        # Knuckle bumps
        for dx in [-0.20, -0.05, 0.10, 0.24]:
            p(f"M {cx+dx} {cy-0.42} L {cx+dx+0.10} {cy-0.42} L {cx+dx+0.08} {cy-0.55} L {cx+dx+0.02} {cy-0.55} Z", accent=True)

    elif part_area == "weapon_grip":
        # Beam rifle: barrel + body + scope + grip
        p(f"M {cx-0.72} {cy-0.12} L {cx+0.80} {cy-0.12} L {cx+0.80} {cy+0.18} L {cx-0.72} {cy+0.18} Z")
        # Extended barrel tip
        p(f"M {cx+0.80} {cy-0.08} L {cx+1.10} {cy-0.08} L {cx+1.10} {cy+0.14} L {cx+0.80} {cy+0.14} Z", accent=True)
        # Scope
        p(f"M {cx-0.10} {cy-0.32} L {cx+0.38} {cy-0.32} L {cx+0.38} {cy-0.12} L {cx-0.10} {cy-0.12} Z", accent=True)
        # Trigger guard / handle
        p(f"M {cx-0.22} {cy+0.18} L {cx+0.12} {cy+0.18} L {cx+0.05} {cy+0.52} L {cx-0.28} {cy+0.52} Z")

    elif part_area == "backpack":
        # Thruster block + two nozzles
        p(f"M {cx-0.75} {cy-0.38} L {cx+0.75} {cy-0.38} L {cx+0.75} {cy+0.30} L {cx-0.75} {cy+0.30} Z")
        # Detail panel
        p(f"M {cx-0.60} {cy-0.25} L {cx+0.60} {cy-0.25} L {cx+0.60} {cy+0.02} L {cx-0.60} {cy+0.02} Z", accent=True)
        # Left nozzle (hexagonal approximation)
        lx = cx - 0.38
        p(f"M {lx-0.18} {cy+0.30} L {lx+0.18} {cy+0.30} L {lx+0.28} {cy+0.52} L {lx+0.18} {cy+0.72} L {lx-0.18} {cy+0.72} L {lx-0.28} {cy+0.52} Z")
        # Right nozzle
        rx = cx + 0.38
        p(f"M {rx-0.18} {cy+0.30} L {rx+0.18} {cy+0.30} L {rx+0.28} {cy+0.52} L {rx+0.18} {cy+0.72} L {rx-0.18} {cy+0.72} L {rx-0.28} {cy+0.52} Z")

    elif part_area == "gate_area":
        # Chamfered armor panel
        p(f"M {cx-0.82} {cy-0.52} L {cx-0.50} {cy-0.68} L {cx+0.50} {cy-0.68} "
          f"L {cx+0.82} {cy-0.52} L {cx+0.82} {cy+0.52} "
          f"L {cx+0.50} {cy+0.68} L {cx-0.50} {cy+0.68} L {cx-0.82} {cy+0.52} Z")
        # Inner panel line
        p(f"M {cx-0.58} {cy-0.38} L {cx-0.30} {cy-0.52} L {cx+0.30} {cy-0.52} "
          f"L {cx+0.58} {cy-0.38} L {cx+0.58} {cy+0.38} "
          f"L {cx+0.30} {cy+0.52} L {cx-0.30} {cy+0.52} L {cx-0.58} {cy+0.38} Z", accent=True)

    else:
        # Fallback: simple angular shape
        p(f"M {cx-0.38} {cy-0.38} L {cx+0.38} {cy-0.38} L {cx+0.38} {cy+0.38} L {cx-0.38} {cy+0.38} Z")


def _draw_runner_antenna_overlay(fig: go.Figure, cx: float, cy: float, color: str) -> None:
    """Make the A1 part read as a V antenna when using a photo-like background."""

    outline = dict(color="#E2E8F0", width=1.25)
    risk_outline = dict(color="#FC8181", width=1.4, dash="dot")
    fill = _transparent_color(color, 0.24)
    base_fill = "rgba(40,45,60,0.72)"

    def add(path: str, *, accent: bool = False) -> None:
        fig.add_shape(
            type="path",
            path=path,
            fillcolor=base_fill if accent else fill,
            line=outline,
        )

    # A fictional V antenna: two long fins, a center nub, and a shared base.
    add(f"M {cx-0.25} {cy-0.58} L {cx-0.56} {cy+0.70} "
        f"L {cx-0.38} {cy+0.78} L {cx-0.05} {cy-0.12} "
        f"L {cx+0.04} {cy-0.58} Z")
    add(f"M {cx+0.25} {cy-0.58} L {cx+0.56} {cy+0.70} "
        f"L {cx+0.38} {cy+0.78} L {cx+0.05} {cy-0.12} "
        f"L {cx-0.04} {cy-0.58} Z")
    add(f"M {cx-0.08} {cy-0.16} L {cx} {cy+0.70} L {cx+0.08} {cy-0.16} Z")
    add(f"M {cx-0.34} {cy-0.72} L {cx+0.34} {cy-0.72} "
        f"L {cx+0.26} {cy-0.48} L {cx-0.26} {cy-0.48} Z", accent=True)

    # Highlight the thin fin tips independently from the gate markers.
    fig.add_shape(
        type="circle",
        x0=cx - 0.70,
        y0=cy + 0.52,
        x1=cx + 0.70,
        y1=cy + 0.95,
        fillcolor="rgba(0,0,0,0)",
        line=risk_outline,
    )


def _draw_gate_markers(
    fig: go.Figure,
    gate_points: list,
    issue_cats: list,
    gate_pos: str,
    *,
    active: bool = True,
) -> None:
    if not active:
        for gx, gy in gate_points:
            fig.add_shape(
                type="circle",
                x0=gx - 0.07,
                y0=gy - 0.07,
                x1=gx + 0.07,
                y1=gy + 0.07,
                fillcolor="rgba(113,128,150,0.45)",
                line=dict(color="rgba(226,232,240,0.45)", width=0.8),
            )
        return

    risky = bool(set(issue_cats) & {"gate_mark", "breakage_risk", "small_parts"})
    gate_color = "#E53E3E" if (risky or gate_pos in {"front", "tip"}) else "#718096"
    for gx, gy in gate_points:
        fig.add_shape(type="circle", x0=gx - 0.10, y0=gy - 0.10,
                      x1=gx + 0.10, y1=gy + 0.10,
                      fillcolor=gate_color, line=dict(color="#E2E8F0", width=1.0))
        if gate_color == "#E53E3E":
            fig.add_shape(type="circle", x0=gx - 0.26, y0=gy - 0.26,
                          x1=gx + 0.26, y1=gy + 0.26,
                          fillcolor="rgba(0,0,0,0)",
                          line=dict(color="#FC8181", width=1.0, dash="dash"))


def _draw_mecha_silhouette(fig: go.Figure) -> None:
    """Fictional mecha body silhouette. No real product or trademark is referenced."""

    def armor(path: str) -> None:
        fig.add_shape(type="path", path=path, layer="below",
                      fillcolor="rgba(36,50,95,0.82)",
                      line=dict(color="rgba(80,105,165,0.60)", width=0.9))

    def armor2(path: str) -> None:
        fig.add_shape(type="path", path=path, layer="below",
                      fillcolor="rgba(48,60,85,0.78)",
                      line=dict(color="rgba(80,105,165,0.50)", width=0.8))

    def gold(path: str) -> None:
        fig.add_shape(type="path", path=path, layer="below",
                      fillcolor="rgba(200,158,10,0.90)",
                      line=dict(color="rgba(230,190,40,0.65)", width=0.8))

    def red(path: str) -> None:
        fig.add_shape(type="path", path=path, layer="below",
                      fillcolor="rgba(200,45,45,0.84)",
                      line=dict(color="rgba(230,80,80,0.55)", width=0.7))

    def thruster(cx: float, cy: float, r: float) -> None:
        fig.add_shape(type="circle", x0=cx-r, y0=cy-r, x1=cx+r, y1=cy+r,
                      layer="below", fillcolor="rgba(25,35,55,0.90)",
                      line=dict(color="rgba(80,120,190,0.70)", width=1.1))
        fig.add_shape(type="circle", x0=cx-r*0.6, y0=cy-r*0.6,
                      x1=cx+r*0.6, y1=cy+r*0.6,
                      layer="below", fillcolor="rgba(180,95,20,0.50)",
                      line=dict(color="rgba(220,135,40,0.55)", width=0.7))

    # --- V-FIN ---
    gold("M 4.65 8.18 L 4.42 8.82 L 4.88 8.32 Z")
    gold("M 5.35 8.18 L 5.58 8.82 L 5.12 8.32 Z")
    gold("M 4.90 8.25 L 5.00 8.68 L 5.10 8.25 Z")

    # --- HEAD ---
    armor("M 4.52 7.60 L 5.48 7.60 L 5.40 8.18 L 4.60 8.18 Z")
    armor("M 4.65 7.44 L 5.35 7.44 L 5.48 7.60 L 4.52 7.60 Z")
    armor2("M 4.65 7.60 L 5.35 7.60 L 5.30 8.05 L 4.70 8.05 Z")
    red("M 4.68 7.80 L 5.32 7.80 L 5.32 7.95 L 4.68 7.95 Z")

    # --- NECK ---
    armor2("M 4.78 7.30 L 5.22 7.30 L 5.28 7.44 L 4.72 7.44 Z")

    # --- CHEST ---
    armor("M 3.65 7.30 L 6.35 7.30 L 6.10 6.52 L 3.90 6.52 Z")
    armor2("M 4.05 6.52 L 5.95 6.52 L 5.80 5.92 L 4.20 5.92 Z")
    red("M 4.85 6.78 L 5.15 6.78 L 5.15 7.10 L 4.85 7.10 Z")
    armor2("M 3.90 6.78 L 4.32 6.78 L 4.32 7.10 L 3.90 7.10 Z")
    armor2("M 5.68 6.78 L 6.10 6.78 L 6.10 7.10 L 5.68 7.10 Z")

    # --- LEFT SHOULDER ARMOR (around 3.15,6.6) ---
    armor("M 2.10 6.98 L 3.70 6.98 L 3.72 6.20 L 2.50 5.85 L 1.95 6.18 Z")
    armor2("M 2.18 6.18 L 3.62 6.18 L 3.62 6.98 L 2.18 6.98 Z")

    # --- RIGHT SHOULDER (6.85 area) ---
    armor("M 6.30 6.98 L 7.90 6.98 L 8.05 6.18 L 7.50 5.85 L 6.28 6.20 Z")

    # --- LEFT UPPER ARM (3.15,6.6 → 2.2,5.25) ---
    armor("M 2.75 6.45 L 3.45 6.22 L 3.25 5.38 L 2.55 5.38 Z")

    # --- LEFT FOREARM (2.2,5.25 → 1.75,4.05) ---
    armor("M 1.90 5.30 L 2.58 5.30 L 2.38 4.08 L 1.68 4.08 Z")
    armor2("M 1.95 4.82 L 2.48 4.82 L 2.45 5.10 L 1.98 5.10 Z")

    # --- LEFT HAND (1.75,4.05) ---
    armor("M 1.52 3.72 L 2.08 3.72 L 2.12 4.08 L 1.48 4.08 Z")

    # --- RIGHT UPPER ARM (→weapon direction) ---
    armor("M 6.55 6.42 L 7.28 6.22 L 7.58 5.08 L 6.85 5.08 Z")

    # --- RIGHT FOREARM ---
    armor("M 7.05 5.08 L 7.78 5.08 L 8.22 3.98 L 7.48 3.82 Z")

    # --- BEAM RIFLE (weapon_grip at 8.05,4.2) ---
    armor2("M 7.18 4.02 L 9.32 4.02 L 9.32 4.42 L 7.18 4.42 Z")
    fig.add_shape(type="path", path="M 9.32 4.08 L 9.68 4.08 L 9.68 4.36 L 9.32 4.36 Z",
                  layer="below", fillcolor="rgba(30,40,58,0.88)",
                  line=dict(color="rgba(80,105,165,0.55)", width=0.8))
    armor2("M 8.18 3.85 L 8.72 3.85 L 8.72 4.02 L 8.18 4.02 Z")
    armor("M 7.78 4.42 L 8.10 4.42 L 7.98 4.82 L 7.65 4.82 Z")

    # --- BACKPACK (6.92,6.12) ---
    armor("M 6.20 5.60 L 7.65 5.60 L 7.65 6.65 L 6.20 6.65 Z")
    thruster(6.60, 6.88, 0.24)
    thruster(7.25, 6.88, 0.24)

    # --- WAIST (5.0,5.75 in new layout) ---
    armor("M 4.18 5.55 L 5.82 5.55 L 5.82 5.98 L 4.18 5.98 Z")
    armor2("M 4.55 5.05 L 5.45 5.05 L 5.45 5.55 L 4.55 5.55 Z")
    armor("M 3.52 4.95 L 4.22 4.95 L 4.18 5.55 L 3.52 5.42 Z")
    armor("M 5.78 4.95 L 6.48 4.95 L 6.48 5.42 L 5.82 5.55 Z")

    # --- LEFT THIGH ---
    armor("M 3.75 4.95 L 4.52 4.95 L 4.48 3.88 L 3.80 3.88 Z")

    # --- LEFT KNEE ARMOR (4.15,3.62) ---
    armor2("M 3.70 3.95 L 4.58 3.95 L 4.62 3.52 L 3.66 3.52 Z")

    # --- LEFT LOWER LEG ---
    armor("M 3.78 3.52 L 4.52 3.52 L 4.45 2.30 L 3.85 2.30 Z")
    armor2("M 3.82 3.08 L 4.48 3.08 L 4.45 3.38 L 3.84 3.38 Z")

    # --- LEFT FOOT ---
    armor("M 3.52 2.30 L 4.78 2.30 L 4.92 1.78 L 3.45 1.78 Z")
    armor2("M 4.78 2.30 L 4.92 1.95 L 4.92 1.78 L 4.78 1.78 Z")

    # --- RIGHT THIGH ---
    armor("M 5.48 4.95 L 6.25 4.95 L 6.20 3.88 L 5.52 3.88 Z")

    # --- RIGHT KNEE ---
    armor2("M 5.42 3.95 L 6.30 3.95 L 6.34 3.52 L 5.38 3.52 Z")

    # --- RIGHT LOWER LEG ---
    armor("M 5.48 3.52 L 6.22 3.52 L 6.15 2.30 L 5.55 2.30 Z")

    # --- RIGHT FOOT ---
    armor("M 5.22 2.30 L 6.48 2.30 L 6.55 1.78 L 5.08 1.78 Z")


def _add_manual_note_card(fig: go.Figure, risk_score: float | None,
                          main_issue_category: str, *, highlight: bool = False) -> None:
    x, y, w, h = 6.55, 1.40, 2.65, 1.15
    facecolor  = _risk_color(risk_score)
    score_text = f"{risk_score:.0f}" if risk_score is not None else "n/a"

    fig.add_shape(type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
                  fillcolor=facecolor, line=dict(color=_BORDER, width=1.4), opacity=0.92)
    if highlight:
        fig.add_shape(type="rect", x0=x - 0.12, y0=y - 0.12,
                      x1=x + w + 0.12, y1=y + h + 0.12,
                      fillcolor="rgba(0,0,0,0)",
                      line=dict(color="#FC8181", width=2.2, dash="dash"))
        fig.add_annotation(x=x + w / 2, y=y + h + 0.24,
                           text="<span style='color:#FC8181'>▶ selected</span>",
                           showarrow=False, font=dict(size=8),
                           xref="x", yref="y", xanchor="center")

    fig.add_annotation(x=x + 0.18, y=y + h - 0.22, text="<b>Manual note</b>",
                       showarrow=False, xanchor="left",
                       font=dict(size=9, color=_TEXT), xref="x", yref="y")
    fig.add_annotation(x=x + 0.18, y=y + h - 0.58, text=f"risk {score_text}",
                       showarrow=False, xanchor="left",
                       font=dict(size=8.5, color=_TEXT), xref="x", yref="y")
    fig.add_annotation(x=x + 0.18, y=y + 0.25,
                       text=main_issue_category or "assembly_difficulty",
                       showarrow=False, xanchor="left",
                       font=dict(size=8, color=_MUTED), xref="x", yref="y")


def _add_assembled_external_card(
    fig: go.Figure,
    part_area: str,
    risk_score: float | None,
    main_issue_category: str,
    note: str,
    *,
    x: float,
    y: float,
    highlight: bool = False,
) -> None:
    w, h = 1.85, 0.92
    x0, y0 = x - w / 2, y - h / 2
    x1, y1 = x + w / 2, y + h / 2
    facecolor = _risk_color(risk_score)
    score_text = f"{risk_score:.0f}" if risk_score is not None else "n/a"

    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fillcolor=facecolor,
        opacity=0.25,
        line=dict(color=facecolor, width=1.5, dash="dot"),
    )
    if highlight:
        fig.add_shape(
            type="rect",
            x0=x0 - 0.10,
            y0=y0 - 0.10,
            x1=x1 + 0.10,
            y1=y1 + 0.10,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="#FC8181", width=2.2, dash="dash"),
        )

    fig.add_annotation(
        x=x0 + 0.12,
        y=y1 - 0.18,
        text=f"<b>{part_area}</b>",
        showarrow=False,
        xanchor="left",
        font=dict(size=8.5, color=_TEXT),
        xref="x",
        yref="y",
    )
    fig.add_annotation(
        x=x0 + 0.12,
        y=y1 - 0.45,
        text=f"risk <b>{score_text}</b> / {note}",
        showarrow=False,
        xanchor="left",
        font=dict(size=7.8, color=_TEXT),
        xref="x",
        yref="y",
    )
    fig.add_annotation(
        x=x0 + 0.12,
        y=y0 + 0.18,
        text=main_issue_category or "n/a",
        showarrow=False,
        xanchor="left",
        font=dict(size=7.5, color=_MUTED),
        xref="x",
        yref="y",
    )


def _add_risk_legend(fig: go.Figure) -> None:
    for level, color in [("High", RISK_COLORS["high"]),
                         ("Medium", RISK_COLORS["medium"]),
                         ("Low", RISK_COLORS["low"])]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=color,
                                             line=dict(color=_TEXT, width=1)),
                                 name=level, showlegend=True))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _summarize_by_part(
    records: Sequence[Mapping[str, Any]],
    inspection_phase: str | None = None,
) -> dict[str, dict[str, Any]]:
    scores     : dict[str, list[int]]       = defaultdict(list)
    categories : dict[str, Counter[str]]    = defaultdict(Counter)
    gates      : dict[str, Counter[str]]    = defaultdict(Counter)
    for record in records:
        record_phase = record.get("inspection_phase")
        if inspection_phase and record_phase and str(record_phase) != inspection_phase:
            continue
        pa = str(record.get("part_area", "unknown"))
        scores[pa].append(int(record.get("risk_score", 0)))
        categories[pa].update(_as_list(record.get("issue_categories", [])))
        gates[pa].update([str(record.get("gate_position", "hidden"))])

    summary: dict[str, dict[str, Any]] = {}
    for pa, sc in scores.items():
        main_issue = categories[pa].most_common(1)
        main_gate  = gates[pa].most_common(1)
        summary[pa] = {
            "average_risk_score":   sum(sc) / len(sc),
            "feedback_count":       len(sc),
            "main_issue_category":  main_issue[0][0] if main_issue else "n/a",
            "issue_categories":     list(categories[pa].keys()),
            "gate_position":        main_gate[0][0]  if main_gate  else "hidden",
        }
    return summary


def _risk_color(risk_score: float | None) -> str:
    if risk_score is None:       return RISK_COLORS["none"]
    if risk_score >= 70:         return RISK_COLORS["high"]
    if risk_score >= 40:         return RISK_COLORS["medium"]
    return RISK_COLORS["low"]


def _transparent_color(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return f"rgba(113,128,150,{alpha})"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _risk_level(risk_score: float | None) -> str:
    if risk_score is None: return "none"
    if risk_score >= 70:   return "high"
    if risk_score >= 40:   return "medium"
    return "low"


def _point(value: Any) -> tuple[float, float]:
    x, y = value
    return float(x), float(y)


def _assembled_layout_point(layout: Mapping[str, Any]) -> tuple[float, float]:
    if "fx" in layout and "fy" in layout:
        bounds = ASSEMBLED_BG_BOUNDS
        return (
            float(bounds["x"]) + float(layout["fx"]) * float(bounds["w"]),
            float(bounds["y"]) - float(layout["fy"]) * float(bounds["h"]),
        )
    return float(layout["x"]), float(layout["y"])


def _priority_label(risk_score: int) -> str:
    if risk_score >= 70: return "高"
    if risk_score >= 40: return "中"
    return "低"


def _verification_hint(viewpoint: str) -> str:
    hints = {
        "設計": "試作可動・保持力テスト",
        "金型": "ゲート位置・寸法ばらつき確認",
        "説明書": "初心者組み立てレビュー",
        "ユーザー体験": "組み立て後アンケート比較",
        "設計/金型": "試作切り出し・ゲート跡確認",
    }
    return hints.get(viewpoint, "追加レビューで確認")


def _as_list(value: Any) -> list[str]:
    if value is None:        return []
    if isinstance(value, str): return [value]
    return [str(item) for item in value]
