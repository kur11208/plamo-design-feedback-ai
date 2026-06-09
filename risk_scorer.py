"""Risk scoring rules for classified plamo feedback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from schemas import RiskBreakdownItem, RiskExplanation


CATEGORY_WEIGHTS = {
    "breakage_risk": 35,
    "tight_joint": 25,
    "loose_joint": 20,
    "small_parts": 20,
    "assembly_difficulty": 15,
    "instruction_unclear": 10,
    "gate_mark": 10,
    "posing_stability": 15,
    "satisfaction_positive": -10,
    "other": 0,
}


SEVERITY_WEIGHTS = {
    "low": 0,
    "medium": 10,
    "high": 20,
}


def calculate_risk_score(
    classification: Mapping[str, Any],
    part_features: Mapping[str, Any] | None = None,
) -> int:
    """Return a 0-100 risk score from classification and optional part features."""

    return int(explain_risk_score(classification, part_features)["risk_score"])


def explain_risk_score(
    classification: Mapping[str, Any],
    part_features: Mapping[str, Any] | None = None,
) -> RiskExplanation:
    """Return score and human-readable score breakdown."""

    categories = _as_list(classification.get("issue_categories", []))
    severity = str(classification.get("severity", "low")).lower()
    features = {key: _normalize_feature(value) for key, value in (part_features or {}).items()}

    breakdown: list[RiskBreakdownItem] = []
    for category in categories:
        points = CATEGORY_WEIGHTS.get(category, 0)
        if points:
            breakdown.append(
                {
                    "factor_type": "issue_category",
                    "factor": category,
                    "points": points,
                    "reason": f"問題カテゴリ: {category}",
                }
            )

    severity_points = SEVERITY_WEIGHTS.get(severity, 0)
    if severity_points:
        breakdown.append(
            {
                "factor_type": "severity",
                "factor": severity,
                "points": severity_points,
                "reason": f"重要度: {severity}",
            }
        )

    breakdown.extend(_feature_bonus_items(categories, features))
    raw_score = sum(int(item["points"]) for item in breakdown)
    risk_score = max(0, min(100, int(raw_score)))
    return {
        "risk_score": risk_score,
        "raw_score": raw_score,
        "breakdown": breakdown,
    }


def _feature_risk_bonus(categories: list[str], features: Mapping[str, str]) -> int:
    return sum(int(item["points"]) for item in _feature_bonus_items(categories, features))


def _feature_bonus_items(categories: list[str], features: Mapping[str, str]) -> list[RiskBreakdownItem]:
    bonus_items: list[RiskBreakdownItem] = []
    if features.get("part_size") == "small" and "breakage_risk" in categories:
        bonus_items.append(
            {
                "factor_type": "part_feature",
                "factor": "part_size=small",
                "points": 15,
                "reason": "小型部品かつ破損リスクあり",
            }
        )
    if features.get("moving_part") == "true" and "tight_joint" in categories:
        bonus_items.append(
            {
                "factor_type": "part_feature",
                "factor": "moving_part=true",
                "points": 15,
                "reason": "可動部かつ関節が固い",
            }
        )
    if features.get("moving_part") == "true" and "loose_joint" in categories:
        bonus_items.append(
            {
                "factor_type": "part_feature",
                "factor": "moving_part=true",
                "points": 10,
                "reason": "可動部かつ保持力不足",
            }
        )
    if features.get("estimated_load") == "high":
        bonus_items.append(
            {
                "factor_type": "part_feature",
                "factor": "estimated_load=high",
                "points": 10,
                "reason": "想定荷重が高い",
            }
        )
    if features.get("gate_position") == "front" and "gate_mark" in categories:
        bonus_items.append(
            {
                "factor_type": "part_feature",
                "factor": "gate_position=front",
                "points": 10,
                "reason": "正面ゲートかつゲート跡リスクあり",
            }
        )
    if features.get("material_type") == "abs" and "tight_joint" in categories:
        bonus_items.append(
            {
                "factor_type": "part_feature",
                "factor": "material_type=ABS",
                "points": 5,
                "reason": "ABS想定かつ固い可動部",
            }
        )
    return bonus_items


def _normalize_feature(value: Any) -> str:
    return str(value).strip().lower()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
