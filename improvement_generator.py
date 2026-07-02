"""Generate improvement suggestions and Markdown reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


VIEWPOINTS = ("設計", "金型", "説明書", "ユーザー体験")
INSPECTION_PHASE_LABELS = {
    "runner_state": "ランナー状態・組立前",
    "assembled_state": "組み立て後",
}


_CATEGORY_SUGGESTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "breakage_risk": {
        "設計": (
            "薄肉部や細い根元に補強形状を追加し、可動時の負荷集中を下げる。",
            "曲げ応力がかかる方向を想定し、断面形状やR形状を見直す。",
        ),
        "金型": (
            "切り出し時に力が伝わりにくいランナー接続位置を検討する。",
            "細部の成形安定性を確認し、薄すぎる先端形状を避ける。",
        ),
        "説明書": (
            "破損しやすい箇所に注意アイコンと拡大図を追加する。",
        ),
        "ユーザー体験": (
            "初心者でも安全に扱えるよう、切り出し順序や支え方を補足する。",
        ),
    },
    "tight_joint": {
        "設計": (
            "軸径と受け側のクリアランスを見直し、可動時の摩擦を適正化する。",
            "回転軸まわりの逃げ形状を追加し、白化や割れのリスクを下げる。",
        ),
        "金型": (
            "軸部の寸法ばらつきが可動感へ与える影響を確認する。",
        ),
        "説明書": (
            "固い場合に無理に動かさない注意文を追加する。",
        ),
        "ユーザー体験": (
            "組み立て直後でも動かしやすい可動抵抗を目標値として設定する。",
        ),
    },
    "loose_joint": {
        "設計": (
            "接続形状や摩擦面積を見直し、保持力不足を抑える。",
            "差し込み深さやロック形状を調整し、抜けや傾きを減らす。",
        ),
        "金型": (
            "保持に関わる面の成形精度と摩耗しやすい角の形状を確認する。",
        ),
        "説明書": (
            "差し込み不足が起きやすい箇所に断面図や完了状態の図を追加する。",
        ),
        "ユーザー体験": (
            "ポーズ変更を繰り返しても保持できる耐久確認項目を追加する。",
        ),
    },
    "assembly_difficulty": {
        "設計": (
            "左右や向きが直感的に分かる非対称形状やガイドを追加する。",
        ),
        "金型": (
            "似た形状の部品はランナー上の配置や番号で混同しにくくする。",
        ),
        "説明書": (
            "初心者向けに組み立て順序の補足説明と拡大図を追加する。",
            "間違えやすい向きを矢印や注意アイコンで強調する。",
        ),
        "ユーザー体験": (
            "初回組み立て時の迷いを減らすため、工程ごとの確認ポイントを設ける。",
        ),
    },
    "gate_mark": {
        "設計": (
            "外観で目立つ面にゲート跡が残らない部品分割を検討する。",
        ),
        "金型": (
            "ゲート位置を目立ちにくい面へ移動し、処理しやすい接続にする。",
            "切断後の白化が目立つ箇所はゲートの太さや向きを見直す。",
        ),
        "説明書": (
            "ゲート処理が必要な箇所に拡大図と推奨手順を追加する。",
        ),
        "ユーザー体験": (
            "初心者でも仕上がり差が出にくい位置へゲートを配置する。",
        ),
    },
    "instruction_unclear": {
        "設計": (
            "似た部品は形状差や取り付けガイドを強め、説明書依存を下げる。",
        ),
        "金型": (
            "ランナー番号や部品配置が工程順と大きく離れないよう確認する。",
        ),
        "説明書": (
            "注意アイコン、拡大図、左右比較図を追加して誤組みを防ぐ。",
            "完成状態の向きを工程ごとに明示する。",
        ),
        "ユーザー体験": (
            "初心者レビューで迷いが集中する工程を優先して改善する。",
        ),
    },
    "posing_stability": {
        "設計": (
            "重心位置と関節保持力を再評価し、長時間ポーズの安定性を高める。",
            "大型装備を支える補助ジョイントや支点の追加を検討する。",
        ),
        "金型": (
            "保持面の面粗度や寸法ばらつきが傾きに与える影響を確認する。",
        ),
        "説明書": (
            "安定しやすいポーズ例や支え方を補足する。",
        ),
        "ユーザー体験": (
            "遊びや展示で頻出するポーズを想定した保持試験を追加する。",
        ),
    },
    "small_parts": {
        "設計": (
            "小型部品に持ちやすいつまみ代や根元補強を追加する。",
            "極端に細い先端は安全な範囲で厚みを持たせる。",
        ),
        "金型": (
            "切り出し時に部品へ負荷が入りにくいランナー配置へ変更する。",
        ),
        "説明書": (
            "紛失や破損を避けるため、先に切り出さない注意を記載する。",
        ),
        "ユーザー体験": (
            "初心者でも扱いやすい部品サイズと作業順を優先する。",
        ),
    },
    "satisfaction_positive": {
        "設計": (
            "良い評価を受けた保持感や組みやすさを他部位へ展開する。",
        ),
        "ユーザー体験": (
            "満足点を継続評価し、改善対象とのバランスを確認する。",
        ),
    },
    "other": {
        "ユーザー体験": (
            "追加レビューを集め、分類ルールで拾えていない不満を確認する。",
        ),
    },
}


_PART_SUGGESTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "shoulder_joint": {
        "設計": ("肩の可動範囲と軸負荷を同時に確認し、白化しやすい角度を減らす。",),
    },
    "elbow_joint": {
        "設計": ("肘の曲げ方向に対して軸受けが片当たりしない形状へ調整する。",),
    },
    "waist_joint": {
        "設計": ("腰接続は回転と前後荷重の両方で保持力を確認する。",),
    },
    "antenna": {
        "設計": ("細長い小型パーツは根元補強と切り出し方向を優先して見直す。",),
    },
    "backpack": {
        "ユーザー体験": ("背面装備装着時の重心変化を想定した展示安定性を確認する。",),
    },
    "hand_parts": {
        "設計": ("手首や持ち手は交換頻度を考慮し、抜き差し時の保持と破損を両立する。",),
    },
    "weapon_grip": {
        "設計": ("武器保持部はグリップ径と手の開き量を合わせ、保持力を安定させる。",),
    },
    "leg_joint": {
        "設計": ("脚関節は自立時の荷重を基準に保持力と可動範囲を再評価する。",),
    },
    "gate_area": {
        "金型": ("外観面と切り出しやすさの両方からゲート位置を再検討する。",),
    },
    "instruction_step": {
        "説明書": ("工程ごとに完成状態の向きと取り付け前後の差分を示す。",),
    },
}


def generate_improvement_suggestions(
    part_area: str,
    issue_categories: Sequence[str] | str,
    risk_score: int,
    feedback_text: str,
) -> dict[str, list[str]]:
    """Return Japanese suggestions grouped by viewpoint."""

    suggestions: dict[str, list[str]] = {viewpoint: [] for viewpoint in VIEWPOINTS}
    categories = _as_list(issue_categories) or ["other"]

    for category in categories:
        for viewpoint, values in _CATEGORY_SUGGESTIONS.get(category, {}).items():
            suggestions.setdefault(viewpoint, []).extend(values)

    for viewpoint, values in _PART_SUGGESTIONS.get(str(part_area), {}).items():
        suggestions.setdefault(viewpoint, []).extend(values)

    if risk_score >= 70:
        suggestions["ユーザー体験"].append("リスクが高いため、試作段階で初心者と経験者の両方による組み立て確認を行う。")
    elif risk_score >= 40:
        suggestions["ユーザー体験"].append("中程度のリスクとして、該当部位の再レビューと組み立て手順の確認を行う。")
    else:
        suggestions["ユーザー体験"].append("現時点では大きなリスクは低いが、良い評価と軽微な不満を継続的に収集する。")

    return {
        viewpoint: _unique(values)[:4]
        for viewpoint, values in suggestions.items()
        if values
    }


def generate_actionable_fix_plan(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return concrete fix actions for a record as structured data."""

    part_area = str(record.get("part_area", "unknown"))
    categories = set(_as_list(record.get("issue_categories", [])))
    plans: list[dict[str, str]] = []

    if part_area == "shoulder_joint" and _has_any(categories, {"tight_joint", "breakage_risk"}):
        plans.append(
            {
                "viewpoint": "設計",
                "target": "肩軸と受け側の接続部",
                "action": "軸受け入口に面取り/R追加を行い、軸径と受け側内径のクリアランスを見直し、押し込み方向のガイドを追加する。",
                "expected_effect": "差し込み時や可動初期の応力集中を下げ、白化・ヒビ不安を減らす。",
                "verification": "試作サンプルで可動初期抵抗、白化発生率、初心者組立時の不安コメントを確認する。",
            }
        )

    if part_area == "antenna" and _has_any(categories, {"breakage_risk", "small_parts", "gate_mark"}):
        plans.append(
            {
                "viewpoint": "設計/金型",
                "target": "細長いA1パーツの根元とランナー接続部",
                "action": "根元厚みを確保し、ゲート位置を先端から目立ちにくい側へ移動し、切り出し順序の注意を追加する。",
                "expected_effect": "切り出し時の折れ不安とゲート跡の目立ちやすさを減らす。",
                "verification": "切り出し時の破損率、ゲート跡の目立ちやすさ、初心者レビューを確認する。",
            }
        )

    if part_area == "waist_joint" and _has_any(categories, {"loose_joint", "posing_stability"}):
        plans.append(
            {
                "viewpoint": "設計",
                "target": "腰接続軸と受け側",
                "action": "差し込み深さ、摩擦面積、ロック形状を見直し、前後荷重とひねり荷重の両方で保持できる形状に調整する。",
                "expected_effect": "上半身や背面装備の荷重で傾く、戻る、ぐらつく状態を減らす。",
                "verification": "装備あり/なしでの自立時間、ポーズ保持率、可動後の保持力変化を比較する。",
            }
        )

    if part_area in {"hand_parts", "weapon_grip"} and _has_any(categories, {"loose_joint", "small_parts"}):
        plans.append(
            {
                "viewpoint": "設計",
                "target": "持ち手とグリップ接触面",
                "action": "グリップ径と手の開き量を合わせ、保持用の浅い凹凸またはストッパーを追加する。",
                "expected_effect": "武器保持時のずれや落下を減らし、抜き差し後も保持力を維持する。",
                "verification": "武器保持時の落下回数、抜き差し回数後の保持力、長時間ポーズ時のずれ量を確認する。",
            }
        )

    if part_area == "gate_area" and "gate_mark" in categories:
        plans.append(
            {
                "viewpoint": "金型",
                "target": "外観面に近いゲート位置",
                "action": "ゲートを裏側・側面・装甲の影になる位置へ移動し、切断後に外観面へ跡が残りにくい接続へ変更する。",
                "expected_effect": "素組み状態で目立つゲート跡を減らし、初心者でも仕上がり差が出にくくなる。",
                "verification": "素組み状態でのゲート跡視認性、初心者の処理難度コメント、切断後の白化発生を確認する。",
            }
        )

    if part_area == "instruction_step" and _has_any(categories, {"instruction_unclear", "assembly_difficulty"}):
        plans.append(
            {
                "viewpoint": "説明書",
                "target": "該当する組み立て工程",
                "action": "拡大図、左右比較、取り付け後の完成状態図、注意アイコンを追加する。",
                "expected_effect": "左右や向きの誤認を減らし、初心者が説明書だけで正しく組める状態に近づける。",
                "verification": "初心者が説明書だけで正しく組めるか、誤組み回数、工程ごとの迷いコメントを確認する。",
            }
        )

    if not plans:
        plans.append(_generic_actionable_fix_plan(record))

    return plans


def generate_stakeholder_tags(record: Mapping[str, Any]) -> list[str]:
    """Return stakeholder tags for who should inspect the issue first."""

    tags: list[str] = []
    categories = set(_as_list(record.get("issue_categories", [])))
    part_area = str(record.get("part_area", ""))

    if categories & {"breakage_risk", "tight_joint", "loose_joint", "posing_stability", "small_parts"}:
        tags.append("設計者向け")
    if categories & {"gate_mark", "breakage_risk"} or part_area in {"gate_area", "antenna"}:
        tags.append("金型担当向け")
    if categories & {"instruction_unclear", "assembly_difficulty"} or part_area == "instruction_step":
        tags.append("説明書担当向け")
    if int(record.get("risk_score", 0)) >= 70 or categories & {"posing_stability", "satisfaction_positive"}:
        tags.append("品質/UX担当向け")

    return _unique(tags) or ["追加レビュー向け"]


def generate_before_after_validation_plan(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return a compact before/after validation plan for one feedback record."""

    fix_plan = generate_actionable_fix_plan(record)[0]
    risk_score = int(record.get("risk_score", 0))
    part_area = str(record.get("part_area", "対象部位"))
    categories = set(_as_list(record.get("issue_categories", [])))

    metric = _validation_metric(part_area, categories)
    return [
        {
            "phase": "Before",
            "check_item": f"{part_area} の現状サンプルを確認する。",
            "metric": metric,
            "target": "現状値とユーザー不安コメントを記録する。",
        },
        {
            "phase": "Change",
            "check_item": fix_plan["action"],
            "metric": "変更内容の反映有無",
            "target": "試作または模式検証で変更点を確認できる状態にする。",
        },
        {
            "phase": "After",
            "check_item": f"{part_area} の改善後サンプルを同条件で確認する。",
            "metric": metric,
            "target": _validation_target(risk_score),
        },
    ]


def format_suggestions_markdown(suggestions: Mapping[str, Sequence[str]]) -> str:
    """Format grouped suggestions as Markdown."""

    lines: list[str] = []
    for viewpoint in VIEWPOINTS:
        values = suggestions.get(viewpoint, [])
        if not values:
            continue
        lines.append(f"### {viewpoint}")
        lines.extend(f"- {value}" for value in values)
        lines.append("")
    return "\n".join(lines).strip()


def build_improvement_report(records: Sequence[Mapping[str, Any]]) -> str:
    """Build a Markdown improvement report from analyzed feedback records."""

    if not records:
        return "# Plamo Design Feedback AI 改善案レポート\n\n分析対象データがありません。"

    sorted_records = sorted(records, key=lambda item: int(item.get("risk_score", 0)), reverse=True)
    top_records = sorted_records[:5]
    average_score = sum(int(record.get("risk_score", 0)) for record in records) / len(records)
    phase_scores = _average_scores_by_phase(records)
    category_counts = _count_categories(records)
    part_scores = _average_scores_by_part(records)
    executive_summary = _build_executive_summary(records)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Plamo Design Feedback AI 改善案レポート",
        "",
        "> 公式データは使用していない学習用プロトタイプです。サンプルデータはすべて架空のものです。",
        "",
        "## エグゼクティブサマリー",
        "",
        f"- 最も改善優先度が高い部位: {executive_summary['top_part_area']}",
        f"- 主なリスク要因: {executive_summary['risk_factors']}",
        f"- 優先すべき改善観点: {executive_summary['priority_viewpoints']}",
        f"- この分析から得られる示唆: {executive_summary['insight']}",
        "",
        f"- 生成日時: {generated_at}",
        f"- 分析件数: {len(records)}",
        f"- 平均リスクスコア: {average_score:.1f}",
        "",
        "## 検査フェーズ別平均リスク",
        "",
        "| 検査フェーズ | 平均リスクスコア | 件数 |",
        "| --- | ---: | ---: |",
    ]

    for inspection_phase, summary in phase_scores:
        lines.append(f"| {_phase_label(inspection_phase)} | {summary['average']:.1f} | {summary['count']} |")

    lines.extend([
        "",
        "## 部位別平均リスク",
        "",
        "| 部位 | 平均リスクスコア | 件数 |",
        "| --- | ---: | ---: |",
    ])

    for part_area, summary in part_scores:
        lines.append(f"| {part_area} | {summary['average']:.1f} | {summary['count']} |")

    lines.extend([
        "",
        "## 問題カテゴリ別件数",
        "",
        "| カテゴリ | 件数 |",
        "| --- | ---: |",
    ])
    for category, count in category_counts.most_common():
        lines.append(f"| {category} | {count} |")

    lines.extend([
        "",
        "## リスク上位フィードバック",
        "",
    ])

    for index, record in enumerate(top_records, start=1):
        suggestions = record.get("improvement_suggestions", {})
        if not isinstance(suggestions, Mapping):
            suggestions = {}

        lines.extend([
            f"### {index}. {record.get('feedback_id', 'unknown')} / {record.get('part_area', 'unknown')}",
            "",
            f"- キット名: {record.get('kit_name', 'unknown')}",
            f"- 検査フェーズ: {_phase_label(str(record.get('inspection_phase', 'unknown')))}",
            f"- リスクスコア: {record.get('risk_score', 0)}",
            f"- 重要度: {record.get('severity', 'unknown')}",
            f"- 問題カテゴリ: {', '.join(_as_list(record.get('issue_categories', [])))}",
            f"- フィードバック: {record.get('feedback_text', '')}",
            "",
            "原因候補:",
        ])
        lines.extend(f"- {cause}" for cause in _as_list(record.get("cause_candidates", []))[:4])
        lines.extend([
            "",
            "改善案:",
            format_suggestions_markdown(suggestions),
            "",
            "具体的変更案:",
            "",
            "| 観点 | 変更対象 | 変更内容 | 期待効果 | 検証方法 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for plan in generate_actionable_fix_plan(record):
            lines.append(
                "| {viewpoint} | {target} | {action} | {expected_effect} | {verification} |".format(
                    viewpoint=plan["viewpoint"],
                    target=plan["target"],
                    action=plan["action"],
                    expected_effect=plan["expected_effect"],
                    verification=plan["verification"],
                )
            )
        lines.append("")
        lines.extend([
            "Before/After検証計画:",
            "",
            "| 段階 | 確認内容 | 指標 | 目安 |",
            "| --- | --- | --- | --- |",
        ])
        for validation in generate_before_after_validation_plan(record):
            lines.append(
                "| {phase} | {check_item} | {metric} | {target} |".format(
                    phase=validation["phase"],
                    check_item=validation["check_item"],
                    metric=validation["metric"],
                    target=validation["target"],
                )
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _count_categories(records: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(_as_list(record.get("issue_categories", [])))
    return counter


def _average_scores_by_part(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, dict[str, float]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("part_area", "unknown"))].append(int(record.get("risk_score", 0)))

    summaries = [
        (
            part_area,
            {"average": sum(scores) / len(scores), "count": len(scores)},
        )
        for part_area, scores in grouped.items()
    ]
    return sorted(summaries, key=lambda item: item[1]["average"], reverse=True)


def _average_scores_by_phase(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, dict[str, float]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("inspection_phase", "unknown"))].append(int(record.get("risk_score", 0)))

    summaries = [
        (
            inspection_phase,
            {"average": sum(scores) / len(scores), "count": len(scores)},
        )
        for inspection_phase, scores in grouped.items()
    ]
    return sorted(summaries, key=lambda item: item[1]["average"], reverse=True)


def _build_executive_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    top_part_area = _top_priority_part(records)
    top_records = [record for record in records if str(record.get("part_area")) == top_part_area]
    if not top_records:
        top_records = list(records)

    risk_factors = _summarize_risk_factors(top_records)
    priority_viewpoints = _summarize_priority_viewpoints(top_records)
    insight = _build_insight(top_part_area, risk_factors, priority_viewpoints)

    return {
        "top_part_area": top_part_area,
        "risk_factors": risk_factors,
        "priority_viewpoints": priority_viewpoints,
        "insight": insight,
    }


def _top_priority_part(records: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("part_area", "unknown"))].append(int(record.get("risk_score", 0)))
    if not grouped:
        return "unknown"

    def sort_key(item: tuple[str, list[int]]) -> tuple[float, int, int]:
        _, scores = item
        return (sum(scores) / len(scores), max(scores), len(scores))

    return max(grouped.items(), key=sort_key)[0]


def _summarize_risk_factors(records: Sequence[Mapping[str, Any]]) -> str:
    category_counter = _count_categories(records)
    feature_counter: Counter[str] = Counter()
    for record in records:
        categories = set(_as_list(record.get("issue_categories", [])))
        if record.get("part_size") == "small" and "breakage_risk" in categories:
            feature_counter["小型部品と破損リスクの組み合わせ"] += 1
        if str(record.get("moving_part")).lower() == "true" and "tight_joint" in categories:
            feature_counter["可動部の固さ"] += 1
        if str(record.get("moving_part")).lower() == "true" and "loose_joint" in categories:
            feature_counter["可動部の保持力不足"] += 1
        if record.get("estimated_load") == "high":
            feature_counter["高荷重部位"] += 1
        if record.get("gate_position") == "front" and "gate_mark" in categories:
            feature_counter["正面ゲート跡"] += 1

    factors = [category for category, _ in category_counter.most_common(3)]
    factors.extend(feature for feature, _ in feature_counter.most_common(3))
    return "、".join(_unique(factors)) if factors else "明確な集中要因なし"


def _summarize_priority_viewpoints(records: Sequence[Mapping[str, Any]]) -> str:
    viewpoint_counter: Counter[str] = Counter()
    for record in records:
        suggestions = record.get("improvement_suggestions", {})
        if isinstance(suggestions, Mapping):
            for viewpoint, values in suggestions.items():
                viewpoint_counter[str(viewpoint)] += len(_as_list(values))
    if not viewpoint_counter:
        return "追加レビュー"
    return "、".join(viewpoint for viewpoint, _ in viewpoint_counter.most_common(3))


def _build_insight(top_part_area: str, risk_factors: str, priority_viewpoints: str) -> str:
    return (
        f"今回のサンプルでは {top_part_area} に改善優先度が集中している。"
        f"主な要因は {risk_factors} であり、{priority_viewpoints} の観点から先に検証すると、"
        "設計変更とユーザー体験改善の接点を見つけやすい。"
    )


def _phase_label(inspection_phase: str) -> str:
    return INSPECTION_PHASE_LABELS.get(inspection_phase, inspection_phase)


def _generic_actionable_fix_plan(record: Mapping[str, Any]) -> dict[str, str]:
    part_area = str(record.get("part_area", "対象部位"))
    categories = set(_as_list(record.get("issue_categories", [])))
    if "tight_joint" in categories:
        return {
            "viewpoint": "設計",
            "target": f"{part_area} の可動接続部",
            "action": "軸径、受け側内径、摩擦面、差し込みガイドを確認し、可動初期抵抗が過度に高くならない寸法へ調整する。",
            "expected_effect": "可動時の白化不安や組み立て時の押し込み負荷を減らす。",
            "verification": "可動初期抵抗、白化発生率、組み立て時の不安コメントを比較する。",
        }
    if "loose_joint" in categories or "posing_stability" in categories:
        return {
            "viewpoint": "設計",
            "target": f"{part_area} の保持構造",
            "action": "差し込み深さ、摩擦面積、ロック形状、荷重方向を確認し、保持力が落ちにくい接続へ調整する。",
            "expected_effect": "ポーズ保持時の傾き、抜け、ぐらつきを減らす。",
            "verification": "装備あり/なしでの保持時間、ポーズ保持率、繰り返し可動後の保持力を確認する。",
        }
    if "gate_mark" in categories:
        return {
            "viewpoint": "金型",
            "target": f"{part_area} のゲート接続部",
            "action": "外観面を避けたゲート配置とし、切り出し方向と処理しやすさを見直す。",
            "expected_effect": "素組み時に目立つゲート跡と処理難度を減らす。",
            "verification": "ゲート跡視認性、白化発生、初心者の処理難度コメントを確認する。",
        }
    return {
        "viewpoint": "UX",
        "target": f"{part_area} のユーザー体験",
        "action": "該当部位のレビューを追加収集し、迷い・不安・満足点を工程別に整理する。",
        "expected_effect": "改善すべき設計要因と説明書要因を切り分けやすくする。",
        "verification": "追加レビューで同じ不満が再現するか、ユーザーレベル別に確認する。",
    }


def _has_any(categories: set[str], targets: set[str]) -> bool:
    return bool(categories & targets)


def _validation_metric(part_area: str, categories: set[str]) -> str:
    if part_area == "antenna" or "breakage_risk" in categories:
        return "破損率、白化発生率、不安コメント件数"
    if part_area == "gate_area" or "gate_mark" in categories:
        return "ゲート跡視認性、処理難度コメント、白化発生"
    if "tight_joint" in categories:
        return "可動初期抵抗、白化発生率、不安コメント件数"
    if "loose_joint" in categories or "posing_stability" in categories:
        return "ポーズ保持率、自立時間、繰り返し可動後の保持力"
    if "instruction_unclear" in categories or "assembly_difficulty" in categories:
        return "誤組み回数、工程ごとの迷いコメント、組み立て時間"
    return "レビュー再発率、満足度、不安コメント件数"


def _validation_target(risk_score: int) -> str:
    if risk_score >= 70:
        return "高リスク要因が明確に減り、同種不満コメントが減ることを確認する。"
    if risk_score >= 40:
        return "中リスク要因が悪化せず、改善前より不安コメントが減ることを確認する。"
    return "現状品質を維持しつつ、追加不満が増えないことを確認する。"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _unique(values: Sequence[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values
