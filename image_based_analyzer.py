from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RUNNER_IMAGE_CUES = {
    "antenna": {
        "visual_target": "A1 antenna",
        "visual_cue": "細いV字形状、先端付近のゲート点、赤い点線囲み",
        "risk_interpretation": "切り出し時の破損、白化、細い根元への応力集中",
        "recommended_action": "ゲート位置を先端から逃がし、根元厚みと切り出し順序の注意表示を見直す",
    },
    "hand_parts": {
        "visual_target": "A2 hand_parts",
        "visual_cue": "小型部品が複数並び、ゲート点が近い",
        "risk_interpretation": "紛失、切り出し時の変形、交換時の扱いにくさ",
        "recommended_action": "つまみ代や仮保持用タブを追加し、保持面からゲートを外す",
    },
    "weapon_grip": {
        "visual_target": "A3 weapon_grip",
        "visual_cue": "小型の棒状パーツとグリップ接続点",
        "risk_interpretation": "ゲート跡が保持面や見た目に影響する可能性",
        "recommended_action": "グリップ接触面を避けてゲートを配置し、保持用の浅い凹凸を検討する",
    },
    "backpack": {
        "visual_target": "B2 backpack",
        "visual_cue": "大きめの装甲ブロック、低リスク色",
        "risk_interpretation": "大型部品のため破損リスクは低めだが、ゲート位置と切り出し時の保持姿勢を確認",
        "recommended_action": "外観面を避けたゲート配置と、切り出し時に持ちやすいランナー接続を確認する",
    },
    "gate_area": {
        "visual_target": "B3 gate_area",
        "visual_cue": "外観面に近い複数ゲート点",
        "risk_interpretation": "素組み時のゲート跡視認性、初心者の処理負担",
        "recommended_action": "ゲートを裏側・側面・装甲の影になる位置へ移動する",
    },
    "instruction_step": {
        "visual_target": "Manual note",
        "visual_cue": "ランナー外の説明書メモカード",
        "risk_interpretation": "切り出し順序や工程理解の迷い",
        "recommended_action": "注意アイコン、拡大図、切り出し順序の補足を追加する",
    },
}


ASSEMBLED_IMAGE_CUES = {
    "shoulder_joint": {
        "visual_target": "shoulder_joint",
        "visual_cue": "最大の赤マーカー、肩軸付近の強調リング",
        "risk_interpretation": "可動時の応力集中、固い軸、白化・ヒビ不安",
        "recommended_action": "肩軸の軸径、受け側内径、入口面取り/R、押し込み方向ガイドを優先確認する",
    },
    "elbow_joint": {
        "visual_target": "elbow_joint",
        "visual_cue": "腕側の中リスクマーカー",
        "risk_interpretation": "可動時の固さ、摩擦集中",
        "recommended_action": "肘軸の初期抵抗、軸受け形状、可動クリアランスを確認する",
    },
    "waist_joint": {
        "visual_target": "waist_joint",
        "visual_cue": "胴体中央の中リスクマーカー",
        "risk_interpretation": "腰接続の保持力不足、ポーズ時の傾き",
        "recommended_action": "差し込み深さ、摩擦面積、ロック形状を見直す",
    },
    "leg_joint": {
        "visual_target": "leg_joint",
        "visual_cue": "脚側の中リスクマーカー",
        "risk_interpretation": "自立・ポーズ保持の不安定さ",
        "recommended_action": "接地時荷重、脚関節保持力、可動後の保持力低下を比較する",
    },
    "hand_parts": {
        "visual_target": "hand_parts",
        "visual_cue": "手元の中リスクマーカー",
        "risk_interpretation": "小型手首の扱いにくさ、保持面の弱さ",
        "recommended_action": "交換時のつまみ代、保持面、グリップ接触形状を調整する",
    },
    "weapon_grip": {
        "visual_target": "weapon_grip check card",
        "visual_cue": "完成後画像には武器が写っていないため、別パーツ保持部の確認カードとして表示",
        "risk_interpretation": "武器保持時の落下、抜き差し後の保持力低下",
        "recommended_action": "グリップ径、手の開き量、浅いストッパー形状を調整する",
    },
    "backpack": {
        "visual_target": "backpack rear-view check card",
        "visual_cue": "正面画像では背面部品が見えにくいため、背面/別視点の確認カードとして表示",
        "risk_interpretation": "単体リスクは低めだが、背面荷重として肩・腰へ影響",
        "recommended_action": "単体ではなく重心・肩腰への荷重伝達を確認する",
    },
}


def build_image_based_findings(
    records: Sequence[Mapping[str, Any]],
    inspection_phase: str,
) -> list[dict[str, Any]]:
    """Return findings inferred from the generated inspection image annotations."""

    cue_map = RUNNER_IMAGE_CUES if inspection_phase == "runner_state" else ASSEMBLED_IMAGE_CUES
    summary = _summarize_records(records, inspection_phase)
    findings: list[dict[str, Any]] = []

    for part_area, cue in cue_map.items():
        item = summary.get(part_area, {})
        risk_score = int(round(float(item.get("average_risk_score", 0))))
        feedback_count = int(item.get("feedback_count", 0))
        findings.append(
            {
                "inspection_phase": inspection_phase,
                "part_area": part_area,
                "visual_target": cue["visual_target"],
                "risk_score": risk_score,
                "risk_level": _risk_level(risk_score, feedback_count),
                "main_issue_category": item.get("main_issue_category", "n/a"),
                "feedback_count": feedback_count,
                "visual_cue": cue["visual_cue"],
                "image_based_interpretation": cue["risk_interpretation"],
                "recommended_action": cue["recommended_action"],
            }
        )

    return sorted(findings, key=lambda row: (int(row["risk_score"]), int(row["feedback_count"])), reverse=True)


def build_cross_image_insights(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    runner = {item["part_area"]: item for item in build_image_based_findings(records, "runner_state")}
    assembled = {item["part_area"]: item for item in build_image_based_findings(records, "assembled_state")}

    return [
        {
            "観点": "切り出し破損",
            "ランナー状態": _summary_text(runner.get("antenna")),
            "完成後状態": "完成後図には直接出にくい",
            "示唆": "完成後レビューだけでは見落としやすいので、ランナー検査で先に拾う",
        },
        {
            "観点": "小型部品UX",
            "ランナー状態": _summary_text(runner.get("hand_parts")),
            "完成後状態": _summary_text(assembled.get("hand_parts")),
            "示唆": "切り出し、交換、保持の両方で不満につながるため一体で改善する",
        },
        {
            "観点": "ゲート跡",
            "ランナー状態": _summary_text(runner.get("gate_area")),
            "完成後状態": "完成後図には直接出にくい",
            "示唆": "外観品質と初心者の処理負担としてランナー側で管理する",
        },
        {
            "観点": "可動負荷",
            "ランナー状態": "画像上では弱い",
            "完成後状態": _summary_text(assembled.get("shoulder_joint")),
            "示唆": "組み立て後に初めて見える固さ・白化・保持リスクを別管理する",
        },
        {
            "観点": "荷重影響",
            "ランナー状態": _summary_text(runner.get("backpack")),
            "完成後状態": _summary_text(assembled.get("backpack")),
            "示唆": "部品単体の破損リスクと、完成後の重心・荷重リスクを分けて扱う",
        },
    ]


def build_image_based_report(
    records: Sequence[Mapping[str, Any]],
    runner_image_path: Path | str,
    assembled_image_path: Path | str,
) -> str:
    runner_findings = build_image_based_findings(records, "runner_state")
    assembled_findings = build_image_based_findings(records, "assembled_state")
    insights = build_cross_image_insights(records)
    top_runner = runner_findings[0]
    top_assembled = assembled_findings[0]

    lines = [
        "# 画像ベース再分析レポート",
        "",
        "対象画像:",
        "",
        f"- `{runner_image_path}`",
        f"- `{assembled_image_path}`",
        "",
        "この分析は、現在のアプリが生成している架空ランナー模式図と組み立て後検査図の検査点、色、ゲート点、リスクラベルを構造化して再整理したものです。実在商品画像、公式画像、公式設計データは使用していません。",
        "",
        "## 結論",
        "",
        f"- ランナー状態では `{top_runner['part_area']}` が最優先です。画像上の読み取り: {top_runner['visual_cue']}。",
        f"- 完成後状態では `{top_assembled['part_area']}` が最優先です。画像上の読み取り: {top_assembled['visual_cue']}。",
        "- `hand_parts` は両フェーズに出るため、切り出しやすさと保持力を同時に改善する価値があります。",
        "",
        "## ランナー画像からの分析",
        "",
        _markdown_table(
            runner_findings,
            ["risk_level", "visual_target", "risk_score", "main_issue_category", "visual_cue", "recommended_action"],
        ),
        "",
        "## 完成後画像からの分析",
        "",
        _markdown_table(
            assembled_findings,
            ["risk_level", "visual_target", "risk_score", "main_issue_category", "visual_cue", "recommended_action"],
        ),
        "",
        "## 画像間の対応関係",
        "",
        _markdown_table(insights, ["観点", "ランナー状態", "完成後状態", "示唆"]),
    ]
    return "\n".join(lines) + "\n"


def _summarize_records(records: Sequence[Mapping[str, Any]], inspection_phase: str) -> dict[str, dict[str, Any]]:
    scores: dict[str, list[int]] = defaultdict(list)
    categories: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        if str(record.get("inspection_phase")) != inspection_phase:
            continue
        part_area = str(record.get("part_area", "unknown"))
        scores[part_area].append(int(record.get("risk_score", 0)))
        categories[part_area].update(_as_list(record.get("issue_categories", [])))

    summary: dict[str, dict[str, Any]] = {}
    for part_area, values in scores.items():
        main_issue = categories[part_area].most_common(1)
        summary[part_area] = {
            "average_risk_score": sum(values) / len(values),
            "feedback_count": len(values),
            "main_issue_category": main_issue[0][0] if main_issue else "n/a",
        }
    return summary


def _summary_text(item: Mapping[str, Any] | None) -> str:
    if not item:
        return "該当なし"
    if int(item.get("feedback_count", 0)) == 0:
        return "該当フィードバックなし"
    return f"{item['part_area']} / risk {item['risk_score']} / {item['main_issue_category']}"


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")).replace("\n", " ") for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _risk_level(risk_score: int, feedback_count: int = 1) -> str:
    if feedback_count <= 0:
        return "No feedback"
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
