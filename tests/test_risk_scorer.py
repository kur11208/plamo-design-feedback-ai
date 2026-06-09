from __future__ import annotations

import unittest

from risk_scorer import calculate_risk_score, explain_risk_score


class RiskScorerTest(unittest.TestCase):
    def test_part_feature_bonus_is_reflected(self) -> None:
        classification = {
            "issue_categories": ["breakage_risk"],
            "severity": "medium",
        }

        base_score = calculate_risk_score(classification)
        feature_score = calculate_risk_score(classification, {"part_size": "small"})

        self.assertEqual(base_score, 45)
        self.assertEqual(feature_score, 60)
        self.assertGreater(feature_score, base_score)

    def test_score_is_clamped_to_100(self) -> None:
        classification = {
            "issue_categories": [
                "breakage_risk",
                "tight_joint",
                "loose_joint",
                "small_parts",
                "gate_mark",
            ],
            "severity": "high",
        }
        features = {
            "part_size": "small",
            "moving_part": "true",
            "estimated_load": "high",
            "gate_position": "front",
            "material_type": "ABS",
        }

        self.assertEqual(calculate_risk_score(classification, features), 100)

    def test_explanation_contains_feature_reason(self) -> None:
        explanation = explain_risk_score(
            {"issue_categories": ["tight_joint"], "severity": "medium"},
            {"moving_part": "true", "material_type": "ABS"},
        )

        reasons = [item["reason"] for item in explanation["breakdown"]]
        self.assertIn("可動部かつ関節が固い", reasons)
        self.assertIn("ABS想定かつ固い可動部", reasons)

    def test_score_is_clamped_to_0_for_positive_only_feedback(self) -> None:
        classification = {
            "issue_categories": ["satisfaction_positive"],
            "severity": "low",
        }

        self.assertEqual(calculate_risk_score(classification), 0)

    def test_front_gate_bonus_requires_gate_mark_category(self) -> None:
        features = {"gate_position": "front"}

        without_gate_mark = calculate_risk_score(
            {"issue_categories": ["breakage_risk"], "severity": "low"},
            features,
        )
        with_gate_mark = calculate_risk_score(
            {"issue_categories": ["gate_mark"], "severity": "low"},
            features,
        )

        self.assertEqual(without_gate_mark, 35)
        self.assertEqual(with_gate_mark, 20)


if __name__ == "__main__":
    unittest.main()
