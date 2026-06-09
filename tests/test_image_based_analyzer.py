from __future__ import annotations

import unittest

import pandas as pd

from app import analyze_feedback_dataframe, build_image_findings_dataframe
from image_based_analyzer import (
    build_cross_image_insights,
    build_image_based_findings,
    build_image_based_report,
)


class ImageBasedAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        feedback_df = pd.read_csv("data/sample_feedback.csv", encoding="utf-8")
        self.records = analyze_feedback_dataframe(feedback_df)

    def test_runner_image_findings_prioritize_antenna(self) -> None:
        findings = build_image_based_findings(self.records, "runner_state")

        self.assertEqual(findings[0]["part_area"], "antenna")
        self.assertIn("A1", findings[0]["visual_target"])
        self.assertIn("ゲート", findings[0]["visual_cue"])

    def test_assembled_image_findings_prioritize_shoulder_joint(self) -> None:
        findings = build_image_based_findings(self.records, "assembled_state")

        self.assertEqual(findings[0]["part_area"], "shoulder_joint")
        self.assertEqual(findings[0]["risk_level"], "High")
        self.assertIn("肩", findings[0]["recommended_action"])

    def test_image_findings_dataframe_uses_portfolio_labels(self) -> None:
        findings = build_image_based_findings(self.records, "runner_state")
        display_df = build_image_findings_dataframe(findings)

        self.assertIn("画像上の読み取り", display_df.columns)
        self.assertIn("改善方向", display_df.columns)
        self.assertIn("アンテナ", set(display_df["部位"]))

    def test_cross_image_insights_include_small_parts_ux(self) -> None:
        insights = build_cross_image_insights(self.records)

        self.assertIn("小型部品UX", {item["観点"] for item in insights})

    def test_markdown_report_references_current_images(self) -> None:
        report = build_image_based_report(
            self.records,
            "assets/runner_sample.png",
            "assets/assembled_sample.png",
        )

        self.assertIn("画像ベース再分析レポート", report)
        self.assertIn("assets/runner_sample.png", report)
        self.assertIn("assets/assembled_sample.png", report)
        self.assertIn("antenna", report)
        self.assertIn("shoulder_joint", report)
        self.assertIn("| risk_level | visual_target | risk_score |", report)


if __name__ == "__main__":
    unittest.main()
