from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from image_recognizer import analyze_runner_image


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


if __name__ == "__main__":
    unittest.main()
