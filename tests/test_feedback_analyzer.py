from __future__ import annotations

import unittest

from feedback_analyzer import analyze_feedback


class FeedbackAnalyzerTest(unittest.TestCase):
    def test_detects_breakage_and_small_part(self) -> None:
        result = analyze_feedback("アンテナ部分が細く、切り出すときに折れそうだった。")

        self.assertIn("breakage_risk", result["issue_categories"])
        self.assertIn("small_parts", result["issue_categories"])
        self.assertIn("antenna", result["detected_parts"])
        self.assertIn(result["severity"], {"medium", "high"})

    def test_positive_feedback_lowers_severity(self) -> None:
        result = analyze_feedback("脚の関節はしっかりしていて自立も安定していたので満足した。")

        self.assertIn("satisfaction_positive", result["issue_categories"])
        self.assertEqual(result["severity"], "low")

    def test_negated_risk_phrases_are_not_over_detected(self) -> None:
        result = analyze_feedback("アンテナは細いが折れにくく、ゲート跡も目立たないので安心した。")

        self.assertNotIn("breakage_risk", result["issue_categories"])
        self.assertNotIn("gate_mark", result["issue_categories"])
        self.assertIn("small_parts", result["issue_categories"])
        self.assertIn("satisfaction_positive", result["issue_categories"])


if __name__ == "__main__":
    unittest.main()
