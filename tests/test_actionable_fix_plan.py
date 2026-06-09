from __future__ import annotations

import unittest

from improvement_generator import generate_actionable_fix_plan, generate_before_after_validation_plan


class ActionableFixPlanTest(unittest.TestCase):
    def test_shoulder_joint_plan_is_specific(self) -> None:
        plans = generate_actionable_fix_plan(
            {
                "part_area": "shoulder_joint",
                "issue_categories": ["tight_joint", "breakage_risk"],
            }
        )
        joined = " ".join(" ".join(plan.values()) for plan in plans)

        self.assertIn("肩軸と受け側", joined)
        self.assertIn("面取り/R追加", joined)
        self.assertIn("クリアランス", joined)
        self.assertIn("可動初期抵抗", joined)
        self.assertIn("白化発生率", joined)

    def test_antenna_plan_is_specific(self) -> None:
        plans = generate_actionable_fix_plan(
            {
                "part_area": "antenna",
                "issue_categories": ["breakage_risk", "small_parts", "gate_mark"],
            }
        )
        joined = " ".join(" ".join(plan.values()) for plan in plans)

        self.assertIn("アンテナ根元", joined)
        self.assertIn("ゲート位置", joined)
        self.assertIn("切り出し時の破損率", joined)

    def test_part_specific_branches_cover_core_portfolio_cases(self) -> None:
        cases = [
            ({"part_area": "waist_joint", "issue_categories": ["loose_joint"]}, "腰接続軸と受け側"),
            ({"part_area": "hand_parts", "issue_categories": ["small_parts"]}, "持ち手とグリップ接触面"),
            ({"part_area": "weapon_grip", "issue_categories": ["loose_joint"]}, "持ち手とグリップ接触面"),
            ({"part_area": "gate_area", "issue_categories": ["gate_mark"]}, "外観面に近いゲート位置"),
            ({"part_area": "instruction_step", "issue_categories": ["instruction_unclear"]}, "該当する組み立て工程"),
        ]

        for record, expected_target in cases:
            with self.subTest(part_area=record["part_area"]):
                plans = generate_actionable_fix_plan(record)
                self.assertGreaterEqual(len(plans), 1)
                self.assertEqual(plans[0]["target"], expected_target)

    def test_before_after_validation_plan_has_comparable_metrics(self) -> None:
        validation_plan = generate_before_after_validation_plan(
            {
                "part_area": "shoulder_joint",
                "issue_categories": ["tight_joint"],
                "risk_score": 80,
            }
        )

        phases = {item["phase"] for item in validation_plan}
        self.assertIn("Before", phases)
        self.assertIn("After", phases)
        self.assertTrue(all(item["metric"] for item in validation_plan))


if __name__ == "__main__":
    unittest.main()
