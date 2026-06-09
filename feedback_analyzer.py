"""Rule-based feedback classification for Plamo Design Feedback AI.

The public function keeps a small, clean interface so the implementation can
later be replaced by an LLM or ML classifier without changing the Streamlit UI.
"""

from __future__ import annotations

import unicodedata

from schemas import ClassificationResult


ISSUE_CATEGORIES = (
    "breakage_risk",
    "tight_joint",
    "loose_joint",
    "assembly_difficulty",
    "gate_mark",
    "instruction_unclear",
    "posing_stability",
    "small_parts",
    "satisfaction_positive",
    "other",
)


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "breakage_risk": (
        "折れ",
        "折れそう",
        "割れ",
        "破損",
        "白化",
        "ヒビ",
        "ひび",
        "欠け",
        "もげ",
        "怖",
        "こわ",
    ),
    "tight_joint": (
        "固",
        "硬",
        "きつ",
        "渋",
        "動かしにく",
        "抜き差ししにく",
        "負荷",
    ),
    "loose_joint": (
        "ゆる",
        "緩",
        "外れ",
        "抜け",
        "保持",
        "垂れ",
        "傾",
        "ぐらつ",
        "ポロリ",
    ),
    "assembly_difficulty": (
        "組みにく",
        "はめにく",
        "合わせにく",
        "迷",
        "間違",
        "押し込",
        "噛み合わせ",
        "工程",
        "向き",
    ),
    "gate_mark": (
        "ゲート",
        "ゲート跡",
        "跡",
        "白く",
        "ニッパー",
        "処理",
        "切り出",
        "ランナー",
    ),
    "instruction_unclear": (
        "説明書",
        "手順",
        "工程",
        "図",
        "左右",
        "番号",
        "分かりにく",
        "わかりにく",
        "見づら",
        "拡大図",
    ),
    "posing_stability": (
        "ポーズ",
        "自立",
        "倒れ",
        "傾",
        "安定",
        "支え",
        "構え",
    ),
    "small_parts": (
        "細",
        "小さ",
        "小さい",
        "紛失",
        "ピンセット",
        "アンテナ",
        "指",
        "手首",
        "クリアパーツ",
    ),
    "satisfaction_positive": (
        "良い",
        "よい",
        "満足",
        "安心",
        "組みやす",
        "しっかり",
        "楽しい",
        "問題な",
        "保持力が高",
        "きれい",
    ),
}


_CATEGORY_NEGATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "breakage_risk": (
        "折れにく",
        "折れない",
        "割れにく",
        "割れない",
        "破損しにく",
        "破損しない",
        "白化しにく",
        "白化しない",
        "怖くない",
        "こわくない",
        "不安はない",
        "心配ない",
    ),
    "tight_joint": (
        "固くない",
        "硬くない",
        "きつくない",
        "渋くない",
        "動かしにくくない",
        "抜き差ししにくくない",
    ),
    "loose_joint": (
        "ゆるくない",
        "緩くない",
        "外れない",
        "抜けない",
        "ぐらつかない",
        "ポロリしない",
        "保持できる",
    ),
    "assembly_difficulty": (
        "組みにくくない",
        "はめにくくない",
        "合わせにくくない",
        "迷わない",
        "間違えない",
    ),
    "gate_mark": (
        "ゲート跡が目立たない",
        "ゲート跡は目立たない",
        "ゲート跡も目立たない",
        "跡が目立たない",
        "跡も目立たない",
        "処理しやすい",
    ),
    "instruction_unclear": (
        "わかりにくくない",
        "分かりにくくない",
        "見づらくない",
        "分かりやすい",
        "わかりやすい",
        "見やすい",
    ),
    "posing_stability": (
        "倒れない",
        "傾かない",
        "安定している",
        "安定する",
        "自立する",
        "支えなしで立つ",
    ),
    "small_parts": (
        "小さすぎない",
        "紛失しにくい",
        "持ちやすい",
    ),
}


_PART_KEYWORDS: dict[str, tuple[str, ...]] = {
    "shoulder_joint": ("肩", "ショルダー", "shoulder"),
    "elbow_joint": ("肘", "ひじ", "elbow"),
    "waist_joint": ("腰", "胴体", "waist"),
    "antenna": ("アンテナ", "角", "antenna"),
    "backpack": ("バックパック", "背中", "backpack"),
    "hand_parts": ("手首", "手", "ハンド", "指"),
    "weapon_grip": ("武器", "グリップ", "持ち手", "weapon"),
    "leg_joint": ("脚", "足", "膝", "ひざ", "leg"),
    "gate_area": ("ゲート", "ランナー", "切り出", "ニッパー"),
    "instruction_step": ("説明書", "工程", "手順", "図", "番号", "左右"),
}


_CAUSE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "breakage_risk": (
        "細い形状や薄肉部に可動負荷が集中している可能性",
        "ランナー切り出し時に部品へ応力が伝わりやすい可能性",
    ),
    "tight_joint": (
        "軸径と受け側のクリアランスが小さい可能性",
        "可動部の摩擦が一部に集中している可能性",
    ),
    "loose_joint": (
        "接続部の摩擦保持が不足している可能性",
        "差し込み深さや受け形状が保持力に対して不足している可能性",
    ),
    "assembly_difficulty": (
        "パーツの向きや左右差が視認しにくい可能性",
        "組み立て順序が直感的でない可能性",
    ),
    "gate_mark": (
        "ゲート位置が外観面に近い可能性",
        "切断面が目立つ面や色に配置されている可能性",
    ),
    "instruction_unclear": (
        "図の角度や拡大情報が不足している可能性",
        "似た形状の部品を工程内で識別しにくい可能性",
    ),
    "posing_stability": (
        "重心と可動部保持力のバランスが不足している可能性",
        "背面装備や武装の荷重が関節へ集中している可能性",
    ),
    "small_parts": (
        "小型部品のつまみ代や根元の厚みが不足している可能性",
        "細い先端形状に切り出し時の応力が残りやすい可能性",
    ),
    "satisfaction_positive": (
        "ユーザー体験上の良い評価が含まれている",
    ),
    "other": (
        "現在のルールでは原因を十分に特定できない",
    ),
}


_IMPROVEMENT_HINTS: dict[str, tuple[str, ...]] = {
    "breakage_risk": (
        "細いパーツの根元に補強形状を追加する",
        "可動時に負荷が集中する箇所の肉厚やR形状を見直す",
    ),
    "tight_joint": (
        "軸径と受け側のクリアランスを見直し可動時の負荷を下げる",
        "組み立て前の仮合わせ注意を説明書に追加する",
    ),
    "loose_joint": (
        "接続形状や摩擦設計を見直し保持力を高める",
        "重い装備を支える部位は差し込み深さを再検討する",
    ),
    "assembly_difficulty": (
        "左右や向きを識別しやすい形状差やマーキングを追加する",
        "初心者向けに組み立て順序の補足説明を追加する",
    ),
    "gate_mark": (
        "ゲート位置を目立ちにくい面へ移動する",
        "処理しやすい切り出し方向を説明書で示す",
    ),
    "instruction_unclear": (
        "説明書に注意アイコンや拡大図を追加する",
        "似た部品を並べて比較できる図を追加する",
    ),
    "posing_stability": (
        "関節保持力と重心バランスを再評価する",
        "大型装備を支える補助ジョイントを検討する",
    ),
    "small_parts": (
        "細いパーツの根元に補強形状を追加する",
        "小型部品は切り出しやすいランナー配置へ変更する",
    ),
    "satisfaction_positive": (
        "良い評価を受けた構造は他部位にも展開できるか検討する",
    ),
    "other": (
        "追加の自由記述データを集めて分類ルールを拡張する",
    ),
}


_SEVERITY_POINTS = {
    "breakage_risk": 3,
    "tight_joint": 2,
    "loose_joint": 2,
    "assembly_difficulty": 1,
    "gate_mark": 1,
    "instruction_unclear": 1,
    "posing_stability": 2,
    "small_parts": 2,
}


_HIGH_SEVERITY_KEYWORDS = (
    "かなり",
    "非常に",
    "すぐ",
    "折れそう",
    "折れた",
    "割れ",
    "破損",
    "白化",
    "倒れ",
    "保持できない",
    "外れやすい",
    "怖",
    "こわ",
)


def analyze_feedback(feedback_text: str) -> ClassificationResult:
    """Analyze one feedback text and return rule-based classification results."""

    normalized_text = _normalize_text(feedback_text)
    if not normalized_text:
        return {
            "issue_categories": ["other"],
            "detected_parts": ["unknown"],
            "severity": "low",
            "cause_candidates": list(_CAUSE_CANDIDATES["other"]),
            "improvement_suggestions": list(_IMPROVEMENT_HINTS["other"]),
        }

    issue_categories = _detect_categories(normalized_text)
    detected_parts = _detect_parts(normalized_text)
    severity = _estimate_severity(issue_categories, normalized_text)
    cause_candidates = _collect_by_category(issue_categories, _CAUSE_CANDIDATES)
    improvement_suggestions = _collect_by_category(issue_categories, _IMPROVEMENT_HINTS)

    return {
        "issue_categories": issue_categories,
        "detected_parts": detected_parts,
        "severity": severity,
        "cause_candidates": cause_candidates,
        "improvement_suggestions": improvement_suggestions,
    }


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).lower().strip()


def _detect_categories(text: str) -> list[str]:
    categories = []
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if _contains_any(text, keywords) and not _is_negated_category(text, category):
            categories.append(category)
    return categories or ["other"]


def _detect_parts(text: str) -> list[str]:
    parts = [
        part_name
        for part_name, keywords in _PART_KEYWORDS.items()
        if _contains_any(text, keywords)
    ]
    return parts or ["unknown"]


def _estimate_severity(issue_categories: list[str], text: str) -> str:
    negative_categories = [
        category
        for category in issue_categories
        if category not in {"satisfaction_positive", "other"}
    ]
    if "satisfaction_positive" in issue_categories and all(
        category == "posing_stability" for category in negative_categories
    ):
        return "low"
    if not negative_categories:
        return "low"

    if _contains_any(text, _HIGH_SEVERITY_KEYWORDS):
        return "high"

    severity_points = sum(_SEVERITY_POINTS.get(category, 0) for category in negative_categories)
    if severity_points >= 4 or len(negative_categories) >= 3:
        return "high"
    return "medium"


def _collect_by_category(
    issue_categories: list[str],
    mapping: dict[str, tuple[str, ...]],
) -> list[str]:
    collected: list[str] = []
    for category in issue_categories:
        collected.extend(mapping.get(category, ()))
    return _unique(collected) or list(mapping["other"])


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(_normalize_text(keyword) in text for keyword in keywords)


def _is_negated_category(text: str, category: str) -> bool:
    return _contains_any(text, _CATEGORY_NEGATION_KEYWORDS.get(category, ()))


def _unique(values: list[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values
