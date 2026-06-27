from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from textwrap import shorten
from typing import Any, Literal, overload

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from feedback_analyzer import analyze_feedback
from improvement_generator import (
    build_improvement_report,
    format_suggestions_markdown,
    generate_actionable_fix_plan,
    generate_before_after_validation_plan,
    generate_improvement_suggestions,
    generate_stakeholder_tags,
)
from image_based_analyzer import (
    build_cross_image_insights,
    build_image_based_findings,
    build_image_based_report,
)
from llm_adapter import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    generate_local_llm_analysis,
    get_local_llm_status,
)
from part_visualizer import (
    build_fix_point_rows,
    plot_assembled_inspection_map,
    plot_runner_inspection_map,
)
from risk_scorer import explain_risk_score
from schemas import FeedbackRecord, PartFeatures, RowAnalysisWarning


APP_TITLE = "Plamo Design Feedback AI"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_feedback.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "improvement_report.md"
IMAGE_ANALYSIS_OUTPUT_PATH = BASE_DIR / "outputs" / "image_based_analysis.md"
RUNNER_IMAGE_PATH = BASE_DIR / "assets" / "runner_sample.png"
ASSEMBLED_IMAGE_PATH = BASE_DIR / "assets" / "assembled_sample.png"
REQUIRED_COLUMNS = {
    "feedback_id",
    "kit_name",
    "part_area",
    "inspection_phase",
    "joint_type",
    "part_size",
    "material_type",
    "moving_part",
    "gate_position",
    "estimated_load",
    "assembly_step",
    "feedback_text",
    "user_level",
    "created_at",
}
PART_FEATURE_COLUMNS = (
    "joint_type",
    "part_size",
    "material_type",
    "moving_part",
    "gate_position",
    "estimated_load",
    "assembly_step",
)
PART_AREA_OPTIONS = (
    "shoulder_joint",
    "elbow_joint",
    "waist_joint",
    "antenna",
    "backpack",
    "hand_parts",
    "weapon_grip",
    "leg_joint",
    "gate_area",
    "instruction_step",
)
FEATURE_OPTIONS = {
    "joint_type": ("ball_joint", "peg_joint", "c_clip", "hinge", "none"),
    "part_size": ("small", "medium", "large"),
    "material_type": ("PS", "ABS", "soft_plastic"),
    "moving_part": ("true", "false"),
    "gate_position": ("front", "side", "back", "hidden", "tip"),
    "estimated_load": ("low", "medium", "high"),
}
INSPECTION_PHASE_LABELS = {
    "all": "全体",
    "runner_state": "ランナー状態・組立前",
    "assembled_state": "組み立て後",
}
PART_AREA_LABELS = {
    "shoulder_joint": "肩関節",
    "elbow_joint": "肘関節",
    "waist_joint": "腰接続",
    "antenna": "アンテナ",
    "backpack": "背面装備",
    "hand_parts": "手首・手パーツ",
    "weapon_grip": "武器グリップ",
    "leg_joint": "脚関節",
    "gate_area": "ゲート周辺",
    "instruction_step": "説明書工程",
}
ISSUE_CATEGORY_LABELS = {
    "breakage_risk": "破損リスク",
    "tight_joint": "関節が固い",
    "loose_joint": "保持力不足",
    "assembly_difficulty": "組立難度",
    "gate_mark": "ゲート跡",
    "instruction_unclear": "説明書不明瞭",
    "posing_stability": "ポーズ安定性",
    "small_parts": "小型部品",
    "satisfaction_positive": "好意的評価",
    "other": "その他",
}


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _configure_matplotlib()

    st.title(APP_TITLE)
    st.write("プラモデルのユーザーフィードバックから、壊れやすい箇所・組み立てづらい箇所・改善案を分析するプロトタイプ")
    st.info("公式データは使用していない学習用プロトタイプです。サンプルデータはすべて架空のものです。")

    try:
        feedback_df = load_feedback_csv(DATA_PATH, _file_mtime(DATA_PATH))
    except (FileNotFoundError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        st.error(f"CSVの読み込みに失敗しました: {error}")
        st.stop()

    records, row_warnings = analyze_feedback_dataframe(feedback_df, return_errors=True)
    if row_warnings:
        st.warning(f"分析できないCSV行を {len(row_warnings)} 件スキップしました。")
        with st.expander("スキップした行の詳細", expanded=False):
            st.dataframe(pd.DataFrame([asdict(warning) for warning in row_warnings]), width="stretch", hide_index=True)
    if not records:
        st.error("分析できるフィードバック行がありません。CSVの必須項目を確認してください。")
        st.stop()
    result_df = build_result_dataframe(records)

    render_portfolio_overview(records)
    render_summary_cards(records)
    render_single_feedback_demo()

    st.subheader("改善優先度ランキング")
    priority_df = build_priority_ranking_dataframe(records)
    st.dataframe(
        priority_df[
            [
                "part_area_label",
                "part_area",
                "feedback_count",
                "average_risk_score",
                "max_risk_score",
                "main_issue_category_label",
                "main_issue_category",
                "priority_level",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("サンプルCSV")
    st.dataframe(feedback_df, width="stretch", hide_index=True)

    st.subheader("フィードバックごとの分析結果")
    st.dataframe(result_df, width="stretch", hide_index=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("部位別の平均リスクスコア")
        st.pyplot(plot_part_risk_scores(result_df), width="stretch")

    with right_col:
        st.subheader("問題カテゴリ別の件数")
        st.pyplot(plot_category_counts(records), width="stretch")

    st.subheader("検査ビュー")
    runner_tab, assembled_tab = st.tabs(["ランナー検査ビュー", "完成後検査ビュー"])
    with runner_tab:
        st.caption("ゲート位置・小型部品・切り出し時の破損リスクを可視化します。")
        st.plotly_chart(plot_runner_inspection_map(filter_records_by_phase(records, "runner_state")), width="stretch")
    with assembled_tab:
        st.caption("関節の固さ・保持力不足・ポージング安定性を可視化します。")
        st.plotly_chart(plot_assembled_inspection_map(filter_records_by_phase(records, "assembled_state")), width="stretch")

    render_image_based_analysis(records)

    st.subheader("リスク上位5件")
    top5_df = result_df.sort_values("risk_score", ascending=False).head(5)
    st.dataframe(
        top5_df[
            [
                "feedback_id",
                "kit_name",
                "part_area_label",
                "part_area",
                "inspection_phase_label",
                "inspection_phase",
                "risk_score",
                "severity",
                "issue_categories_label",
                "issue_categories",
                "feedback_text",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("選択フィードバックの詳細分析")
    selected_record = select_feedback_record(records)
    render_detail_analysis(selected_record, records)

    st.subheader("改善案レポート")
    report_markdown = build_improvement_report(records)
    st.markdown(report_markdown)

    if st.button("outputs/improvement_report.md に保存", type="primary"):
        try:
            save_report(report_markdown, OUTPUT_PATH)
        except OSError as error:
            st.error(f"レポート保存に失敗しました: {error}")
        else:
            st.success(f"保存しました: {OUTPUT_PATH}")


@st.cache_data
def load_feedback_csv(path: Path, _file_mtime_value: float) -> pd.DataFrame:
    """Read the sample CSV. The mtime argument is part of Streamlit's cache key."""

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, encoding="utf-8")
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"必要な列が不足しています: {', '.join(sorted(missing_columns))}")
    return df


@overload
def analyze_feedback_dataframe(
    feedback_df: pd.DataFrame,
    *,
    return_errors: Literal[False] = False,
) -> list[FeedbackRecord]:
    ...


@overload
def analyze_feedback_dataframe(
    feedback_df: pd.DataFrame,
    *,
    return_errors: Literal[True],
) -> tuple[list[FeedbackRecord], list[RowAnalysisWarning]]:
    ...


def analyze_feedback_dataframe(
    feedback_df: pd.DataFrame,
    *,
    return_errors: bool = False,
) -> list[FeedbackRecord] | tuple[list[FeedbackRecord], list[RowAnalysisWarning]]:
    records: list[FeedbackRecord] = []
    row_warnings: list[RowAnalysisWarning] = []

    for row_number, (_, row) in enumerate(feedback_df.iterrows(), start=2):
        try:
            records.append(analyze_feedback_row(row))
        except (KeyError, TypeError, ValueError) as error:
            row_warnings.append(
                RowAnalysisWarning(
                    row_number=row_number,
                    feedback_id=_safe_row_value(row, "feedback_id"),
                    message=str(error),
                )
            )

    if return_errors:
        return records, row_warnings
    return records


def analyze_feedback_row(row: pd.Series) -> FeedbackRecord:
    feedback_text = _required_text(row, "feedback_text")
    feedback_id = _required_text(row, "feedback_id")
    kit_name = _required_text(row, "kit_name")
    part_area = _required_text(row, "part_area")
    inspection_phase = _required_text(row, "inspection_phase")
    if inspection_phase not in {"runner_state", "assembled_state"}:
        raise ValueError(f"inspection_phase が不正です: {inspection_phase}")

    classification = analyze_feedback(feedback_text)
    part_features = extract_part_features(row)
    risk_explanation = explain_risk_score(classification, part_features=part_features)
    risk_score = int(risk_explanation["risk_score"])
    suggestions = generate_improvement_suggestions(
        part_area=part_area,
        issue_categories=classification["issue_categories"],
        risk_score=risk_score,
        feedback_text=feedback_text,
    )

    return {
        "feedback_id": feedback_id,
        "kit_name": kit_name,
        "part_area": part_area,
        "inspection_phase": inspection_phase,
        **part_features,
        "feedback_text": feedback_text,
        "user_level": _required_text(row, "user_level"),
        "created_at": _required_text(row, "created_at"),
        "issue_categories": classification["issue_categories"],
        "detected_parts": classification["detected_parts"],
        "severity": classification["severity"],
        "risk_score": risk_score,
        "risk_explanation": risk_explanation,
        "cause_candidates": classification["cause_candidates"],
        "improvement_suggestions": suggestions,
    }


def build_result_dataframe(records: list[FeedbackRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "feedback_id": record["feedback_id"],
                "kit_name": record["kit_name"],
                "part_area": record["part_area"],
                "part_area_label": _part_area_label(str(record["part_area"])),
                "inspection_phase": record["inspection_phase"],
                "inspection_phase_label": _phase_label(str(record["inspection_phase"])),
                "joint_type": record["joint_type"],
                "part_size": record["part_size"],
                "material_type": record["material_type"],
                "moving_part": record["moving_part"],
                "gate_position": record["gate_position"],
                "estimated_load": record["estimated_load"],
                "assembly_step": record["assembly_step"],
                "user_level": record["user_level"],
                "created_at": record["created_at"],
                "issue_categories": ", ".join(record["issue_categories"]),
                "issue_categories_label": "、".join(_issue_category_label(category) for category in record["issue_categories"]),
                "detected_parts": ", ".join(record["detected_parts"]),
                "severity": record["severity"],
                "risk_score": record["risk_score"],
                "score_factors": summarize_score_factors(record),
                "cause_candidates": " / ".join(record["cause_candidates"][:3]),
                "feedback_text": record["feedback_text"],
            }
        )
    return pd.DataFrame(rows)


def extract_part_features(row: pd.Series) -> PartFeatures:
    return {column: _format_feature_value(row[column]) for column in PART_FEATURE_COLUMNS}


def render_portfolio_overview(records: list[FeedbackRecord]) -> None:
    priority_df = build_priority_ranking_dataframe(records)
    top_part = "-"
    top_reason = "-"
    if not priority_df.empty:
        top_part = str(priority_df.iloc[0]["part_area_label"])
        top_reason = str(priority_df.iloc[0]["main_issue_category_label"])

    with st.container(border=True):
        st.markdown("**このダッシュボードで最初に見ること**")
        cols = st.columns(3)
        cols[0].markdown(
            "**改善優先部位**  \n"
            f"{top_part} を起点に、リスクが集中する部位を確認します。"
        )
        cols[1].markdown(
            "**主なリスク要因**  \n"
            f"{top_reason} など、ユーザーの声と部品特徴を結びつけます。"
        )
        cols[2].markdown(
            "**検査フェーズ**  \n"
            "ランナー状態と組み立て後を分け、切り出し前/完成後の観点で見ます。"
        )


def render_summary_cards(records: list[FeedbackRecord]) -> None:
    total_count = len(records)
    high_risk_count = sum(1 for record in records if int(record["risk_score"]) >= 70)
    average_risk_score = _average([int(record["risk_score"]) for record in records])
    priority_df = build_priority_ranking_dataframe(records)
    top_part = str(priority_df.iloc[0]["part_area_label"]) if not priority_df.empty else "-"
    main_category = _issue_category_label(most_common_issue_category(records))

    st.subheader("開発改善ダッシュボード")
    metric_cols = st.columns(5)
    metric_cols[0].metric("総フィードバック件数", total_count)
    metric_cols[1].metric("高リスク件数", high_risk_count)
    metric_cols[2].metric("最優先改善部位", top_part)
    metric_cols[3].metric("最多問題カテゴリ", main_category)
    metric_cols[4].metric("平均リスクスコア", f"{average_risk_score:.1f}")


def render_single_feedback_demo() -> None:
    with st.expander("任意フィードバック分析デモ", expanded=False):
        feedback_text = st.text_area(
            "フィードバック文",
            value="肩の関節が固く、何度か動かすと白化しそうで不安だった。",
            height=90,
        )
        top_cols = st.columns(3)
        part_area = top_cols[0].selectbox("対象部位", PART_AREA_OPTIONS, index=0)
        inspection_phase = top_cols[1].selectbox(
            "検査フェーズ",
            ("runner_state", "assembled_state"),
            index=1,
        )
        assembly_step = top_cols[2].number_input("工程番号", min_value=1, max_value=99, value=18)

        feature_cols = st.columns(6)
        part_features = {
            "joint_type": feature_cols[0].selectbox("joint_type", FEATURE_OPTIONS["joint_type"], index=3),
            "part_size": feature_cols[1].selectbox("part_size", FEATURE_OPTIONS["part_size"], index=1),
            "material_type": feature_cols[2].selectbox("material_type", FEATURE_OPTIONS["material_type"], index=1),
            "moving_part": feature_cols[3].selectbox("moving_part", FEATURE_OPTIONS["moving_part"], index=0),
            "gate_position": feature_cols[4].selectbox("gate_position", FEATURE_OPTIONS["gate_position"], index=3),
            "estimated_load": feature_cols[5].selectbox("estimated_load", FEATURE_OPTIONS["estimated_load"], index=2),
            "assembly_step": str(int(assembly_step)),
        }

        if st.button("このフィードバックを分析", key="single_feedback_analyze"):
            classification = analyze_feedback(feedback_text)
            risk_explanation = explain_risk_score(classification, part_features=part_features)
            risk_score = int(risk_explanation["risk_score"])
            suggestions = generate_improvement_suggestions(
                part_area=part_area,
                issue_categories=classification["issue_categories"],
                risk_score=risk_score,
                feedback_text=feedback_text,
            )
            demo_record = {
                "feedback_id": "DEMO-001",
                "kit_name": "Demo Fictional Kit",
                "part_area": part_area,
                "inspection_phase": inspection_phase,
                **part_features,
                "feedback_text": feedback_text,
                "user_level": "demo",
                "created_at": "demo",
                "issue_categories": classification["issue_categories"],
                "detected_parts": classification["detected_parts"],
                "severity": classification["severity"],
                "risk_score": risk_score,
                "risk_explanation": risk_explanation,
                "cause_candidates": classification["cause_candidates"],
                "improvement_suggestions": suggestions,
            }
            render_detail_analysis(demo_record, [demo_record])


def build_priority_ranking_dataframe(records: list[FeedbackRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "part_area",
                "part_area_label",
                "feedback_count",
                "average_risk_score",
                "max_risk_score",
                "main_issue_category",
                "main_issue_category_label",
                "priority_level",
            ]
        )

    rows: list[dict[str, Any]] = []
    part_areas = sorted({str(record["part_area"]) for record in records})
    for part_area in part_areas:
        part_records = [record for record in records if str(record["part_area"]) == part_area]
        risk_scores = [int(record["risk_score"]) for record in part_records]
        average_score = _average(risk_scores)
        max_score = max(risk_scores)
        rows.append(
            {
                "part_area": part_area,
                "part_area_label": _part_area_label(part_area),
                "feedback_count": len(part_records),
                "average_risk_score": round(average_score, 1),
                "max_risk_score": max_score,
                "main_issue_category": most_common_issue_category(part_records),
                "main_issue_category_label": _issue_category_label(most_common_issue_category(part_records)),
                "priority_level": priority_level(average_score, max_score),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["average_risk_score", "max_risk_score", "feedback_count"],
        ascending=[False, False, False],
    )


def priority_level(average_risk_score: float, max_risk_score: int) -> str:
    if average_risk_score >= 75 or max_risk_score >= 90:
        return "High"
    if average_risk_score >= 50:
        return "Medium"
    return "Low"


def most_common_issue_category(records: list[FeedbackRecord]) -> str:
    counts: dict[str, int] = {}
    for record in records:
        for category in record["issue_categories"]:
            counts[category] = counts.get(category, 0) + 1
    if not counts:
        return "-"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def summarize_score_factors(record: FeedbackRecord) -> str:
    breakdown = record.get("risk_explanation", {}).get("breakdown", [])
    reasons = [
        f"+{item.get('points', 0)} {_score_factor_label(str(item.get('factor_type', '')), str(item.get('factor', '')))}"
        for item in breakdown
        if item.get("points", 0) > 0
    ]
    return " / ".join(reasons[:4])


def build_score_breakdown_dataframe(record: FeedbackRecord) -> pd.DataFrame:
    breakdown = record.get("risk_explanation", {}).get("breakdown", [])
    rows = [
        {
            "種別": item.get("factor_type", ""),
            "要因": item.get("factor", ""),
            "表示名": _score_factor_label(str(item.get("factor_type", "")), str(item.get("factor", ""))),
            "加点": item.get("points", 0),
            "根拠": item.get("reason", ""),
        }
        for item in breakdown
    ]
    rows.append(
        {
            "種別": "total",
            "要因": "risk_score",
            "加点": record.get("risk_score", 0),
            "根拠": "0から100に丸めた最終スコア",
        }
    )
    return pd.DataFrame(rows)


def render_part_carte(record: FeedbackRecord) -> None:
    labels = [
        ("feedback_id", "Feedback ID"),
        ("kit_name", "Kit"),
        ("part_area", "Part Area"),
        ("inspection_phase", "Inspection Phase"),
        ("joint_type", "Joint"),
        ("part_size", "Part Size"),
        ("material_type", "Material"),
        ("moving_part", "Moving"),
        ("gate_position", "Gate"),
        ("estimated_load", "Load"),
        ("assembly_step", "Step"),
    ]
    with st.container(border=True):
        for start in range(0, len(labels), 4):
            cols = st.columns(4)
            for col, (key, label) in zip(cols, labels[start : start + 4]):
                value = record.get(key, "-")
                if key == "part_area":
                    value = f"{_part_area_label(str(value))} ({value})"
                if key == "inspection_phase":
                    value = _phase_label(str(value))
                col.markdown(f"**{label}**  \n`{value}`")
        tags = " ".join(f"`{tag}`" for tag in generate_stakeholder_tags(record))
        st.markdown(f"**担当観点タグ**  \n{tags}")


def risk_level(risk_score: int) -> str:
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"


def main_score_reason(record: FeedbackRecord) -> str:
    breakdown = record.get("risk_explanation", {}).get("breakdown", [])
    positive_items = [item for item in breakdown if int(item.get("points", 0)) > 0]
    if not positive_items:
        return "-"
    top_item = max(positive_items, key=lambda item: int(item.get("points", 0)))
    return _score_factor_label(str(top_item.get("factor_type", "")), str(top_item.get("factor", "-")))


def build_contextual_cause_summary(record: FeedbackRecord) -> str:
    categories = "、".join(_issue_category_label(category) for category in record.get("issue_categories", []))
    return (
        f"このフィードバックは `{_part_area_label(str(record.get('part_area')))}` に対するもので、"
        f"`moving_part={record.get('moving_part')}`、"
        f"`estimated_load={record.get('estimated_load')}`、"
        f"`part_size={record.get('part_size')}`、"
        f"`material_type={record.get('material_type')}` です。"
        f"`{categories}` が検出されているため、部品形状、接続クリアランス、荷重方向、"
        "組み立て工程のどこでユーザー不安が生じるかを優先して確認します。"
    )


def build_actionable_fix_plan_dataframe(record: FeedbackRecord) -> pd.DataFrame:
    rows = []
    for plan in generate_actionable_fix_plan(record):
        rows.append(
            {
                "観点": plan["viewpoint"],
                "変更対象": plan["target"],
                "変更内容": plan["action"],
                "期待効果": plan["expected_effect"],
                "検証方法": plan["verification"],
            }
        )
    return pd.DataFrame(rows)


def build_validation_plan_dataframe(record: FeedbackRecord) -> pd.DataFrame:
    rows = []
    for item in generate_before_after_validation_plan(record):
        rows.append(
            {
                "段階": item["phase"],
                "確認内容": item["check_item"],
                "指標": item["metric"],
                "目安": item["target"],
            }
        )
    return pd.DataFrame(rows)


def render_image_based_analysis(records: list[FeedbackRecord]) -> None:
    st.subheader("画像ベース再分析")
    st.caption(
        "現在の架空ランナー画像と完成後検査図に描かれた検査点、色、ゲート点、ラベルをもとに、"
        "切り出し前と組み立て後のリスクを再整理します。公式画像や実在商品画像の画像認識ではありません。"
    )

    image_cols = st.columns(2)
    with image_cols[0]:
        if RUNNER_IMAGE_PATH.exists():
            st.image(str(RUNNER_IMAGE_PATH), caption="ランナー画像: 切り出し前リスク")
        else:
            st.info("ランナー画像はまだ生成されていません。`python scripts\\generate_portfolio_assets.py` で生成できます。")
    with image_cols[1]:
        if ASSEMBLED_IMAGE_PATH.exists():
            st.image(str(ASSEMBLED_IMAGE_PATH), caption="完成後検査図: 関節・接続部リスク")
        else:
            st.info("完成後検査図はまだ生成されていません。`python scripts\\generate_portfolio_assets.py` で生成できます。")

    runner_findings = build_image_based_findings(records, "runner_state")
    assembled_findings = build_image_based_findings(records, "assembled_state")
    image_tabs = st.tabs(["ランナー画像からの読み取り", "完成後図からの読み取り", "画像間の対応"])

    with image_tabs[0]:
        st.dataframe(build_image_findings_dataframe(runner_findings), width="stretch", hide_index=True)
    with image_tabs[1]:
        st.dataframe(build_image_findings_dataframe(assembled_findings), width="stretch", hide_index=True)
    with image_tabs[2]:
        st.dataframe(pd.DataFrame(build_cross_image_insights(records)), width="stretch", hide_index=True)

    image_report_markdown = build_image_based_report(
        records,
        RUNNER_IMAGE_PATH.relative_to(BASE_DIR),
        ASSEMBLED_IMAGE_PATH.relative_to(BASE_DIR),
    )
    with st.expander("画像ベース再分析レポート", expanded=False):
        st.markdown(image_report_markdown)

    if st.button("outputs/image_based_analysis.md に保存", key="save_image_based_report"):
        try:
            save_report(image_report_markdown, IMAGE_ANALYSIS_OUTPUT_PATH)
        except OSError as error:
            st.error(f"画像ベース再分析レポートの保存に失敗しました: {error}")
        else:
            st.success(f"保存しました: {IMAGE_ANALYSIS_OUTPUT_PATH}")


def build_image_findings_dataframe(findings: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for finding in findings:
        rows.append(
            {
                "優先度": finding["risk_level"],
                "対象": finding["visual_target"],
                "部位": _part_area_label(str(finding["part_area"])),
                "risk_score": finding["risk_score"],
                "主要カテゴリ": _issue_category_label(str(finding["main_issue_category"])),
                "画像上の読み取り": finding["visual_cue"],
                "解釈": finding["image_based_interpretation"],
                "改善方向": finding["recommended_action"],
            }
        )
    return pd.DataFrame(rows)


def render_local_llm_panel(record: FeedbackRecord) -> None:
    record_key = str(record.get("feedback_id", "unknown")).replace(" ", "_").replace("-", "_")
    result_key = f"local_llm_result_{record_key}"
    status_key = f"local_llm_connection_{record_key}"

    with st.expander("ローカルAI補強分析（Ollama・任意）", expanded=False):
        st.caption(
            "risk_score はルールベースのまま固定し、ローカルモデルでは原因説明・改善案・検証観点だけを補強します。"
            "Ollamaが起動していない場合も、通常の分析画面はそのまま使えます。"
        )
        config_cols = st.columns([2, 3, 1])
        model = config_cols[0].text_input(
            "モデル",
            value=DEFAULT_LOCAL_LLM_MODEL,
            key=f"local_llm_model_{record_key}",
        )
        endpoint = config_cols[1].text_input(
            "エンドポイント",
            value=DEFAULT_LOCAL_LLM_ENDPOINT,
            key=f"local_llm_endpoint_{record_key}",
        )
        timeout = config_cols[2].number_input(
            "timeout秒",
            min_value=5,
            max_value=120,
            value=int(DEFAULT_LOCAL_LLM_TIMEOUT),
            step=5,
            key=f"local_llm_timeout_{record_key}",
        )

        action_cols = st.columns([1, 1, 4])
        if action_cols[0].button("接続確認", key=f"local_llm_status_{record_key}"):
            status = get_local_llm_status(endpoint=endpoint, timeout=3)
            st.session_state[status_key] = status

        if action_cols[1].button("補強分析を生成", key=f"local_llm_generate_{record_key}"):
            with st.spinner("ローカルモデルで補強分析を生成しています..."):
                st.session_state[result_key] = generate_local_llm_analysis(
                    record,
                    endpoint=endpoint,
                    model=model,
                    timeout=float(timeout),
                )

        status_result = st.session_state.get(status_key)
        if isinstance(status_result, dict):
            if status_result.get("available"):
                models = ", ".join(status_result.get("models", [])[:6]) or "モデル一覧なし"
                st.success(f"Ollamaに接続できました: {models}")
            else:
                st.warning(f"接続できませんでした: {status_result.get('error', 'unknown error')}")

        result = st.session_state.get(result_key)
        if isinstance(result, dict):
            if result["ok"]:
                st.markdown(result["content"])
                st.caption(f"model: `{result['model']}` / endpoint: `{result['endpoint']}`")
            else:
                st.warning(f"ローカルAI補強分析を生成できませんでした: {result['error']}")

        st.markdown(
            "PowerShell例: `ollama serve` を起動し、別ターミナルで "
            "`ollama pull llama3.2:3b` など任意のモデルを用意します。"
        )


def build_priority_recommendation(record: FeedbackRecord) -> dict[str, str]:
    score = int(record.get("risk_score", 0))
    level = risk_level(score)
    plans = generate_actionable_fix_plan(record)
    viewpoint = plans[0]["viewpoint"] if plans else "追加レビュー"
    reason_map = {
        "High": "高リスクのため、設計変更または金型・説明書変更の検討対象として先に確認します。",
        "Medium": "中リスクのため、同種レビューの追加確認と試作検証で優先度を固めます。",
        "Low": "低リスクのため、現状は傾向監視と追加レビュー収集を優先します。",
    }
    return {
        "priority": level,
        "viewpoint": viewpoint,
        "short_reason": f"risk {score}",
        "reason": reason_map[level],
    }


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _required_text(row: pd.Series, column: str) -> str:
    value = row[column]
    if pd.isna(value):
        raise ValueError(f"{column} が空です")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{column} が空です")
    return text


def _safe_row_value(row: pd.Series, column: str) -> str:
    try:
        value = row[column]
    except KeyError:
        return "unknown"
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def _format_feature_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip()


def plot_part_risk_scores(result_df: pd.DataFrame) -> plt.Figure:
    part_scores = result_df.groupby("part_area")["risk_score"].mean().sort_values(ascending=False)
    part_scores.index = [_part_area_label(str(part_area)) for part_area in part_scores.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    part_scores.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_xlabel("対象部位")
    ax.set_ylabel("平均リスクスコア")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_category_counts(records: list[FeedbackRecord]) -> plt.Figure:
    category_counts: dict[str, int] = {}
    for record in records:
        for category in record["issue_categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1

    category_df = (
        pd.DataFrame(category_counts.items(), columns=["issue_category", "count"])
        .sort_values("count", ascending=False)
    )
    category_df["issue_category_label"] = category_df["issue_category"].map(_issue_category_label)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    category_df.plot(kind="bar", x="issue_category_label", y="count", ax=ax, color="#F58518", legend=False)
    ax.set_xlabel("問題カテゴリ")
    ax.set_ylabel("件数")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def select_feedback_record(records: list[FeedbackRecord]) -> FeedbackRecord:
    option_labels = {
        (
            f"{record['feedback_id']} | "
            f"{_phase_label(str(record['inspection_phase']))} | "
            f"{_part_area_label(str(record['part_area']))} | "
            f"{shorten(record['feedback_text'], width=36, placeholder='...')}"
        ): record
        for record in records
    }
    selected_label = st.selectbox("詳細を見るフィードバック", list(option_labels.keys()))
    return option_labels[selected_label]


def render_detail_analysis(record: FeedbackRecord, records: list[FeedbackRecord] | None = None) -> None:
    st.subheader("部品改善カルテ")
    render_part_carte(record)

    st.markdown("### ユーザーの声")
    st.markdown(f"> #### {record['feedback_text']}")

    st.markdown("### 対象部位マップ")
    phase_description = (
        "ランナー検査ビューでは、ゲート位置・小型部品・切り出し時の破損リスクを可視化します。"
        if record.get("inspection_phase") == "runner_state"
        else "完成後検査ビューでは、関節の固さ・保持力不足・ポージング安定性を可視化します。"
    )
    st.caption(phase_description)
    context_records = records or [record]
    inspection_phase = str(record.get("inspection_phase", "assembled_state"))
    if inspection_phase == "runner_state":
        map_records = filter_records_by_phase(context_records, "runner_state") or [record]
        figure = plot_runner_inspection_map(map_records, highlight_part_area=str(record.get("part_area")))
    else:
        map_records = filter_records_by_phase(context_records, "assembled_state") or [record]
        figure = plot_assembled_inspection_map(map_records, highlight_part_area=str(record.get("part_area")))
    st.plotly_chart(figure, width="stretch")

    st.markdown("### リスク判定の理由")
    risk_cols = st.columns(4)
    risk_cols[0].metric("risk_score", record["risk_score"])
    risk_cols[1].metric("risk_level", risk_level(int(record["risk_score"])))
    risk_cols[2].metric("severity", record["severity"])
    risk_cols[3].metric("主な加点理由", main_score_reason(record))
    st.caption("この判定は、架空データに基づくルールベース推定です。実製品の品質判定や公式評価ではありません。")
    st.dataframe(build_score_breakdown_dataframe(record), width="stretch", hide_index=True)

    st.markdown("### 具体的な変更案")
    fix_plan_df = build_actionable_fix_plan_dataframe(record)
    st.dataframe(fix_plan_df, width="stretch", hide_index=True)
    st.markdown("**Before/After 検証計画**")
    st.dataframe(build_validation_plan_dataframe(record), width="stretch", hide_index=True)
    render_local_llm_panel(record)

    st.markdown("### 推定原因")
    st.markdown(build_contextual_cause_summary(record))
    for cause in record["cause_candidates"][:3]:
        st.markdown(f"- {cause}")

    st.markdown("### 既存の一般改善案")
    st.markdown(format_suggestions_markdown(record["improvement_suggestions"]))

    with st.expander("補助的な修正ポイント", expanded=False):
        fix_point_rows = build_fix_point_rows(record)
        if fix_point_rows:
            st.dataframe(pd.DataFrame(fix_point_rows), width="stretch", hide_index=True)
        else:
            st.info("修正ポイントを生成できませんでした。")


def save_report(report_markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown, encoding="utf-8")


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def filter_records_by_phase(records: list[FeedbackRecord], inspection_phase: str) -> list[FeedbackRecord]:
    if inspection_phase == "all":
        return records
    return [record for record in records if str(record.get("inspection_phase")) == inspection_phase]


def _phase_value_from_label(label: str) -> str:
    for phase, phase_label in INSPECTION_PHASE_LABELS.items():
        if phase_label == label:
            return phase
    return "all"


def _phase_label(inspection_phase: str) -> str:
    return INSPECTION_PHASE_LABELS.get(inspection_phase, inspection_phase)


def _part_area_label(part_area: str) -> str:
    return PART_AREA_LABELS.get(part_area, part_area)


def _issue_category_label(category: str) -> str:
    return ISSUE_CATEGORY_LABELS.get(category, category)


def _score_factor_label(factor_type: str, factor: str) -> str:
    if factor_type == "issue_category":
        return _issue_category_label(factor)
    if factor_type == "severity":
        return {"low": "低", "medium": "中", "high": "高"}.get(factor, factor)
    return factor


def _configure_matplotlib() -> None:
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


if __name__ == "__main__":
    main()
