from __future__ import annotations

import unittest

import pandas as pd

from app import (
    analyze_feedback_dataframe,
    build_priority_ranking_dataframe,
    build_result_dataframe,
    build_runner_image_report,
    build_runner_input_record,
    build_runner_input_report,
    most_impactful_issue_category,
    primary_issue_category,
)
from improvement_generator import build_improvement_report, generate_actionable_fix_plan


class AppDataflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.feedback_df = pd.read_csv("data/sample_feedback.csv", encoding="utf-8")

    def test_sample_data_has_required_portfolio_features(self) -> None:
        self.assertGreaterEqual(len(self.feedback_df), 30)
        for column in [
            "joint_type",
            "part_size",
            "material_type",
            "moving_part",
            "gate_position",
            "estimated_load",
            "assembly_step",
        ]:
            self.assertIn(column, self.feedback_df.columns)

    def test_analysis_table_contains_design_features(self) -> None:
        records = analyze_feedback_dataframe(self.feedback_df)
        result_df = build_result_dataframe(records)

        self.assertEqual(len(records), len(self.feedback_df))
        self.assertIn("score_factors", result_df.columns)
        self.assertTrue(result_df["risk_score"].between(0, 100).all())

    def test_priority_ranking_and_report_summary(self) -> None:
        records = analyze_feedback_dataframe(self.feedback_df)
        ranking_df = build_priority_ranking_dataframe(records)
        report = build_improvement_report(records)

        self.assertFalse(ranking_df.empty)
        self.assertIn("priority_level", ranking_df.columns)
        self.assertIn("エグゼクティブサマリー", report)
        self.assertIn("最も改善優先度が高い部位", report)
        self.assertIn("| 観点 | 変更対象 | 変更内容 | 期待効果 | 検証方法 |", report)

    def test_top_records_have_actionable_fix_plan(self) -> None:
        records = analyze_feedback_dataframe(self.feedback_df)
        top_record = max(records, key=lambda record: int(record["risk_score"]))

        self.assertGreaterEqual(len(generate_actionable_fix_plan(top_record)), 1)

    def test_row_level_analysis_errors_are_skipped_with_warning(self) -> None:
        broken_df = self.feedback_df.copy()
        broken_df.loc[0, "feedback_text"] = ""
        broken_df.loc[1, "inspection_phase"] = "invalid_phase"

        records, row_warnings = analyze_feedback_dataframe(broken_df, return_errors=True)

        self.assertEqual(len(records), len(self.feedback_df) - 2)
        self.assertEqual(len(row_warnings), 2)
        self.assertIn("feedback_text", row_warnings[0].message)
        self.assertIn("inspection_phase", row_warnings[1].message)

    def test_runner_input_record_scores_image_features(self) -> None:
        record = build_runner_input_record(
            part_area="antenna",
            part_size="small",
            material_type="PS",
            gate_position="tip",
            estimated_load="high",
            assembly_step="runner_check",
            observations=["thin_or_fragile", "tip_gate"],
        )

        self.assertEqual(record["inspection_phase"], "runner_state")
        self.assertIn("breakage_risk", record["issue_categories"])
        self.assertIn("gate_mark", record["issue_categories"])
        self.assertGreaterEqual(record["risk_score"], 70)
        self.assertGreaterEqual(len(generate_actionable_fix_plan(record)), 1)

    def test_primary_issue_uses_largest_score_contribution(self) -> None:
        record = build_runner_input_record(
            part_area="gate_area",
            part_size="small",
            material_type="PS",
            gate_position="front",
            estimated_load="medium",
            assembly_step="local_image_check",
            observations=["visible_gate_mark", "many_gate_points", "small_part"],
        )

        self.assertEqual(primary_issue_category(record), "small_parts")

    def test_safe_observation_does_not_cancel_detected_runner_risk(self) -> None:
        record = build_runner_input_record(
            part_area="gate_area",
            part_size="medium",
            material_type="PS",
            gate_position="front",
            estimated_load="medium",
            assembly_step="local_image_check",
            observations=["looks_safe", "visible_gate_mark"],
        )

        self.assertIn("gate_mark", record["issue_categories"])
        self.assertNotIn("satisfaction_positive", record["issue_categories"])
        self.assertGreaterEqual(record["risk_score"], 30)

    def test_priority_main_issue_uses_score_impact_not_category_order(self) -> None:
        records = [
            build_runner_input_record(
                part_area="gate_area",
                part_size="small",
                material_type="PS",
                gate_position="front",
                estimated_load="medium",
                assembly_step="local_image_check",
                observations=["visible_gate_mark", "many_gate_points", "small_part"],
            )
        ]

        self.assertEqual(most_impactful_issue_category(records), "small_parts")

    def test_runner_input_report_masks_local_image_name(self) -> None:
        record = build_runner_input_record(
            part_area="antenna",
            part_size="small",
            material_type="PS",
            gate_position="tip",
            estimated_load="high",
            assembly_step="runner_check",
            observations=["thin_or_fragile", "tip_gate"],
        )

        report = build_runner_input_report(record, image_reference="local_uploaded_image")

        self.assertIn("ランナー入力評価レポート", report)
        self.assertIn("local_uploaded_image", report)
        self.assertIn("画像ファイル名、画像本体、元パスは保存していません", report)
        self.assertIn("| 観点 | 変更対象 | 変更内容 | 期待効果 | 検証方法 |", report)
        self.assertNotIn("example.jpg", report)
        self.assertNotIn("local_inputs/", report)

    def test_runner_image_report_is_runner_focused(self) -> None:
        records = analyze_feedback_dataframe(self.feedback_df)
        findings = [
            {
                "risk_level": "High",
                "visual_target": "A1 thin_part",
                "risk_score": 77,
                "main_issue_category": "breakage_risk",
                "visual_cue": "細い部品と先端ゲート",
                "recommended_action": "ゲート位置と根元厚みを見直す",
            }
        ]
        report = build_runner_image_report(findings, "assets/runner_sample.png")

        self.assertIn("ランナー画像ベース評価レポート", report)
        self.assertIn("assets/runner_sample.png", report)
        self.assertIn("A1 thin_part", report)
        self.assertNotIn("完成後画像", report)
        self.assertGreater(len(records), 0)


if __name__ == "__main__":
    unittest.main()
