from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict


class ClassificationResult(TypedDict):
    issue_categories: list[str]
    detected_parts: list[str]
    severity: str
    cause_candidates: list[str]
    improvement_suggestions: dict[str, list[str]]


class PartFeatures(TypedDict):
    joint_type: str
    part_size: str
    material_type: str
    moving_part: str
    gate_position: str
    estimated_load: str
    assembly_step: str


class RiskBreakdownItem(TypedDict):
    factor_type: str
    factor: str
    points: int
    reason: str


class RiskExplanation(TypedDict):
    risk_score: int
    raw_score: int
    breakdown: list[RiskBreakdownItem]


class FeedbackRecord(TypedDict):
    feedback_id: str
    kit_name: str
    part_area: str
    inspection_phase: str
    joint_type: str
    part_size: str
    material_type: str
    moving_part: str
    gate_position: str
    estimated_load: str
    assembly_step: str
    feedback_text: str
    user_level: str
    created_at: str
    issue_categories: list[str]
    detected_parts: list[str]
    severity: str
    risk_score: int
    risk_explanation: RiskExplanation
    cause_candidates: list[str]
    improvement_suggestions: dict[str, list[str]]
    analysis_error: NotRequired[str]


class ActionableFixPlanItem(TypedDict):
    viewpoint: str
    target: str
    action: str
    expected_effect: str
    verification: str


class ValidationPlanItem(TypedDict):
    phase: str
    check_item: str
    metric: str
    target: str


@dataclass(frozen=True)
class RowAnalysisWarning:
    row_number: int
    feedback_id: str
    message: str
