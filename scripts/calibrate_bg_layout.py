"""
Detect part positions in background images using Claude vision API
and print updated layout dicts for part_visualizer.py.

Usage:
  ANTHROPIC_API_KEY=sk-... python scripts/calibrate_bg_layout.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import anthropic

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"

# Plot coordinate bounds (must match part_visualizer.py)
RUNNER_IMG_BOUNDS = {"x0": 0.55, "y0_top": 8.55, "width": 8.90, "height": 7.50}
ASSEMBLED_IMG_BOUNDS = {"x0": 0.00, "y0_top": 9.50, "width": 10.00, "height": 9.50}


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _frac_to_runner_coords(fx: float, fy: float) -> tuple[float, float]:
    b = RUNNER_IMG_BOUNDS
    x = round(b["x0"] + fx * b["width"], 2)
    y = round(b["y0_top"] - fy * b["height"], 2)
    return x, y


def _frac_to_assembled_coords(fx: float, fy: float) -> tuple[float, float]:
    b = ASSEMBLED_IMG_BOUNDS
    x = round(b["x0"] + fx * b["width"], 2)
    y = round(b["y0_top"] - fy * b["height"], 2)
    return x, y


RUNNER_PROMPT = """
This is a plastic model runner (sprue) sheet image.
Identify where each part group is located and return ONLY a JSON object.

The parts to locate are:
- "antenna": the V-shaped fin / antenna pieces (labeled A1/A2 area top-left)
- "hand_parts": arm or hand armor pieces
- "weapon_grip": gun barrel / rifle parts
- "backpack": thruster nozzle / cylindrical engine parts
- "gate_area": large flat armor panel pieces

For each part, give the approximate center as fractions of the full image:
  x: 0.0 = left edge, 1.0 = right edge
  y: 0.0 = top edge, 1.0 = bottom edge

Return exactly this JSON structure (no markdown, no explanation):
{
  "antenna":     {"x": 0.0, "y": 0.0},
  "hand_parts":  {"x": 0.0, "y": 0.0},
  "weapon_grip": {"x": 0.0, "y": 0.0},
  "backpack":    {"x": 0.0, "y": 0.0},
  "gate_area":   {"x": 0.0, "y": 0.0}
}
"""

ASSEMBLED_PROMPT = """
This is a mecha robot figure standing in a front-facing pose.
Identify approximate body joint/part locations and return ONLY a JSON object.

Parts to locate:
- "shoulder_joint": center of one shoulder (pick the left shoulder from viewer's right)
- "elbow_joint": center of one elbow
- "waist_joint": center of the waist/hip area
- "leg_joint": center of one knee
- "hand_parts": position of the hand / fist
- "weapon_grip": where the gun/rifle is held
- "backpack": center of the backpack thrusters behind the figure

For each part, give the center as fractions:
  x: 0.0 = left edge, 1.0 = right edge
  y: 0.0 = top edge, 1.0 = bottom edge

Return exactly this JSON (no markdown, no explanation):
{
  "shoulder_joint": {"x": 0.0, "y": 0.0},
  "elbow_joint":    {"x": 0.0, "y": 0.0},
  "waist_joint":    {"x": 0.0, "y": 0.0},
  "leg_joint":      {"x": 0.0, "y": 0.0},
  "hand_parts":     {"x": 0.0, "y": 0.0},
  "weapon_grip":    {"x": 0.0, "y": 0.0},
  "backpack":       {"x": 0.0, "y": 0.0}
}
"""


def _detect(client: anthropic.Anthropic, image_path: Path, prompt: str) -> dict:
    b64 = _b64(image_path)
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/webp",
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw = msg.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def main() -> None:
    client = anthropic.Anthropic()

    runner_path   = ASSETS_DIR / "runner_bg.webp"
    assembled_path = ASSETS_DIR / "assembled_bg.webp"

    print("=== Detecting runner part positions ===")
    runner_fracs = _detect(client, runner_path, RUNNER_PROMPT)
    print("Raw fractions:", json.dumps(runner_fracs, indent=2))

    print("\nRUNNER_PART_LAYOUT center coordinates (paste into part_visualizer.py):")
    for part, frac in runner_fracs.items():
        x, y = _frac_to_runner_coords(frac["x"], frac["y"])
        print(f'  "{part}": center ({x}, {y})')

    print("\n=== Detecting assembled part positions ===")
    assembled_fracs = _detect(client, assembled_path, ASSEMBLED_PROMPT)
    print("Raw fractions:", json.dumps(assembled_fracs, indent=2))

    print("\nASSEMBLED_PART_LAYOUT coordinates (paste into part_visualizer.py):")
    for part, frac in assembled_fracs.items():
        x, y = _frac_to_assembled_coords(frac["x"], frac["y"])
        print(f'  "{part}": x={x}, y={y}')


if __name__ == "__main__":
    main()
