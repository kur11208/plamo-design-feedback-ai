from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from image_recognizer import (
    analysis_findings_dataframe_rows,
    analyze_runner_image,
    crop_runner_image,
    render_roi_preview,
    render_runner_detection_overlay,
    suggest_runner_roi,
)


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class ImageRecognizerTest(unittest.TestCase):
    def test_detects_dense_runner_like_features(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        draw = ImageDraw.Draw(image)
        for x in range(30, 220, 35):
            draw.line((x, 30, x, 150), fill="black", width=3)
        for y in range(35, 160, 28):
            draw.line((25, y, 220, y), fill="black", width=3)
        for index in range(14):
            x = 35 + (index % 7) * 28
            y = 45 + (index // 7) * 55
            draw.rectangle((x, y, x + 11, y + 8), outline="black", fill="lightgray")

        analysis = analyze_runner_image(_png_bytes(image))

        self.assertIn("many_gate_points", analysis["observations"])
        self.assertIn("small_part", analysis["observations"])
        self.assertIn(analysis["gate_position"], {"front", "side", "tip"})
        self.assertGreaterEqual(analysis["confidence"], 0.4)

    def test_blank_image_defaults_to_safe_low_signal(self) -> None:
        image = Image.new("RGB", (200, 140), "white")

        analysis = analyze_runner_image(_png_bytes(image))

        self.assertEqual(analysis["observations"], ["looks_safe"])
        self.assertEqual(analysis["part_size"], "large")
        self.assertEqual(analysis["estimated_load"], "low")
        self.assertEqual(analysis["quality_warnings"], [])

    def test_warns_when_label_like_color_patch_overlaps_runner(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        draw = ImageDraw.Draw(image)
        for x in range(35, 220, 38):
            draw.line((x, 30, x, 155), fill=(180, 180, 180), width=3)
        for y in range(40, 150, 32):
            draw.line((25, y, 220, y), fill=(180, 180, 180), width=3)
        draw.rectangle((60, 55, 145, 110), fill=(245, 245, 245), outline="black", width=2)
        draw.rectangle((70, 65, 92, 82), fill="red")
        draw.rectangle((100, 65, 125, 82), fill="green")
        draw.line((72, 96, 132, 96), fill="black", width=2)

        analysis = analyze_runner_image(_png_bytes(image))

        self.assertTrue(analysis["quality_warnings"])
        self.assertTrue(
            any(row["項目"] == "解析品質注意" for row in analysis_findings_dataframe_rows(analysis))
        )

    def test_warns_when_low_saturation_label_or_glare_overlaps_clear_runner(self) -> None:
        image = Image.new("RGB", (240, 180), (230, 235, 232))
        draw = ImageDraw.Draw(image)
        for x in range(30, 220, 32):
            draw.line((x, 25, x, 160), fill=(150, 155, 152), width=2)
        for y in range(40, 155, 28):
            draw.line((25, y, 220, y), fill=(150, 155, 152), width=2)
        draw.rectangle((55, 52, 150, 112), fill=(248, 248, 248), outline=(25, 25, 25), width=2)
        for offset in range(0, 36, 9):
            draw.line((65, 66 + offset, 135, 66 + offset), fill=(30, 30, 30), width=2)

        analysis = analyze_runner_image(_png_bytes(image))

        self.assertTrue(
            any("透明パーツ" in warning for warning in analysis["quality_warnings"])
        )

    def test_detection_overlay_draws_candidate_boxes(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        draw = ImageDraw.Draw(image)
        for x in range(30, 220, 35):
            draw.line((x, 30, x, 150), fill="black", width=3)
        for y in range(35, 160, 28):
            draw.line((25, y, 220, y), fill="black", width=3)
        image_bytes = _png_bytes(image)

        overlay = Image.open(BytesIO(render_runner_detection_overlay(image_bytes))).convert("RGB")
        highlight_pixels = sum(
            1
            for red, green, blue in overlay.getdata()
            if (red > 180 and green < 90 and blue < 90) or (red > 220 and 110 < green < 190 and blue < 80)
        )

        self.assertGreater(highlight_pixels, 50)

    def test_roi_crop_limits_analysis_to_selected_area(self) -> None:
        image = Image.new("RGB", (260, 160), "white")
        draw = ImageDraw.Draw(image)
        for x in range(20, 115, 18):
            draw.line((x, 20, x, 140), fill="black", width=3)
        for y in range(25, 145, 22):
            draw.line((15, y, 120, y), fill="black", width=3)
        image_bytes = _png_bytes(image)

        left_crop = crop_runner_image(image_bytes, (0.0, 0.0, 0.46, 1.0))
        right_crop = crop_runner_image(image_bytes, (0.55, 0.0, 1.0, 1.0))

        left_analysis = analyze_runner_image(left_crop)
        right_analysis = analyze_runner_image(right_crop)

        self.assertIn("small_part", left_analysis["observations"])
        self.assertEqual(right_analysis["observations"], ["looks_safe"])

    def test_roi_preview_draws_analysis_region(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        preview = Image.open(BytesIO(render_roi_preview(_png_bytes(image), (0.2, 0.2, 0.8, 0.8)))).convert("RGB")
        cyan_pixels = sum(
            1
            for red, green, blue in preview.getdata()
            if red < 40 and 130 < green < 200 and blue > 200
        )

        self.assertGreater(cyan_pixels, 50)

    def test_suggest_runner_roi_selects_dense_runner_region(self) -> None:
        image = Image.new("RGB", (320, 180), "white")
        draw = ImageDraw.Draw(image)
        for x in range(24, 142, 18):
            draw.line((x, 18, x, 160), fill="black", width=3)
        for y in range(28, 156, 22):
            draw.line((18, y, 150, y), fill="black", width=3)
        draw.rectangle((215, 48, 296, 126), fill=(245, 245, 245), outline="black", width=2)
        draw.rectangle((226, 60, 250, 82), fill="red")
        draw.line((225, 100, 286, 100), fill="black", width=2)

        suggestion = suggest_runner_roi(_png_bytes(image))

        left, _top, right, _bottom = suggestion["crop_box"]
        self.assertLess(left, 0.2)
        self.assertLess(right, 0.7)
        self.assertGreaterEqual(suggestion["confidence"], 0.5)

    def test_suggest_runner_roi_defaults_to_full_image_on_low_signal(self) -> None:
        image = Image.new("RGB", (240, 180), "white")

        suggestion = suggest_runner_roi(_png_bytes(image))

        self.assertEqual(suggestion["crop_box"], (0.0, 0.0, 1.0, 1.0))
        self.assertLessEqual(suggestion["confidence"], 0.25)


if __name__ == "__main__":
    unittest.main()
