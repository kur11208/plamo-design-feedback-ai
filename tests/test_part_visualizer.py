from __future__ import annotations

import unittest

import plotly.graph_objects as go

from part_visualizer import plot_runner_inspection_map


class PartVisualizerTest(unittest.TestCase):
    def test_runner_map_uses_runner_specific_labels_and_manual_note(self) -> None:
        fig = plot_runner_inspection_map(
            [
                {
                    "part_area": "antenna",
                    "risk_score": 77,
                    "issue_categories": ["breakage_risk", "small_parts", "gate_mark"],
                    "gate_position": "tip",
                },
                {
                    "part_area": "instruction_step",
                    "risk_score": 41,
                    "issue_categories": ["assembly_difficulty"],
                    "gate_position": "hidden",
                },
            ],
            highlight_part_area="antenna",
        )

        self.assertIsInstance(fig, go.Figure)

        # Title
        self.assertIn("切り出し前リスクマップ", fig.layout.title.text)

        # A1 thin_part label is on the scatter trace text; others are annotations
        annotation_texts = [ann.text for ann in fig.layout.annotations]
        trace_texts = [str(t) for trace in fig.data for t in (trace.text or []) if t]
        all_texts = annotation_texts + trace_texts
        self.assertTrue(any("A1" in t and "thin_part" in t for t in all_texts if t),
                        "A1 thin_part label missing from annotations or trace texts")
        self.assertTrue(any("Manual note" in t for t in annotation_texts if t))
        self.assertTrue(any("selected" in t for t in annotation_texts if t))
        self.assertTrue(any("架空の検査模式図" in t for t in annotation_texts if t))

    def test_runner_map_returns_plotly_figure(self) -> None:
        fig = plot_runner_inspection_map([])
        self.assertIsInstance(fig, go.Figure)
        self.assertIn("切り出し前リスクマップ", fig.layout.title.text)


if __name__ == "__main__":
    unittest.main()
