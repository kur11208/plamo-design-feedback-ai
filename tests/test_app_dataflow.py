from __future__ import annotations

import unittest

import pandas as pd

from app import analyze_feedback_dataframe, build_priority_ranking_dataframe, build_result_dataframe
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


if __name__ == "__main__":
    unittest.main()
