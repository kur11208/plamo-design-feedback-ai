"""Lightweight local image heuristics for runner inspection inputs."""

from __future__ import annotations

from collections import deque
from io import BytesIO
from typing import TypedDict

from PIL import Image


class RunnerImageAnalysis(TypedDict):
    part_area: str
    part_size: str
    material_type: str
    gate_position: str
    estimated_load: str
    observations: list[str]
    confidence: float
    metrics: dict[str, float | int]
    findings: list[dict[str, str]]


def analyze_runner_image(image_bytes: bytes) -> RunnerImageAnalysis:
    """Infer runner inspection features from an uploaded image.

    This is intentionally a lightweight heuristic, not a learned vision model.
    It reads local image structure and returns editable suggestions for the UI.
    """

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((160, 160))
    width, height = image.size
    pixels = list(image.getdata())
    brightness = [_brightness(pixel) for pixel in pixels]
    saturation = [_saturation(pixel) for pixel in pixels]

    edge_mask = _edge_mask(brightness, width, height)
    edge_density = sum(edge_mask) / max(1, len(edge_mask))
    central_edge_ratio = _region_edge_ratio(edge_mask, width, height, region="center")
    border_edge_ratio = _region_edge_ratio(edge_mask, width, height, region="border")
    active_cells = _active_edge_cells(edge_mask, width, height, grid_size=12)
    component_count = _count_components(active_cells)
    thin_line_score = _thin_line_score(active_cells)
    high_brightness_ratio = sum(1 for value in brightness if value >= 210) / max(1, len(brightness))
    low_saturation_ratio = sum(1 for value in saturation if value <= 35) / max(1, len(saturation))

    observations = _infer_observations(
        edge_density=edge_density,
        central_edge_ratio=central_edge_ratio,
        border_edge_ratio=border_edge_ratio,
        component_count=component_count,
        thin_line_score=thin_line_score,
    )
    part_size = _infer_part_size(edge_density, component_count, thin_line_score)
    gate_position = _infer_gate_position(central_edge_ratio, border_edge_ratio, thin_line_score)
    estimated_load = _infer_estimated_load(edge_density, component_count, thin_line_score)
    material_type = _infer_material_type(high_brightness_ratio, low_saturation_ratio)
    part_area = _infer_part_area(observations)
    confidence = _confidence(edge_density, component_count, observations)

    return {
        "part_area": part_area,
        "part_size": part_size,
        "material_type": material_type,
        "gate_position": gate_position,
        "estimated_load": estimated_load,
        "observations": observations,
        "confidence": confidence,
        "metrics": {
            "edge_density": round(edge_density, 4),
            "central_edge_ratio": round(central_edge_ratio, 4),
            "border_edge_ratio": round(border_edge_ratio, 4),
            "active_cell_components": component_count,
            "thin_line_score": round(thin_line_score, 4),
            "high_brightness_ratio": round(high_brightness_ratio, 4),
            "low_saturation_ratio": round(low_saturation_ratio, 4),
        },
        "findings": _build_findings(
            part_size=part_size,
            gate_position=gate_position,
            material_type=material_type,
            observations=observations,
            edge_density=edge_density,
            component_count=component_count,
            thin_line_score=thin_line_score,
        ),
    }


def analysis_findings_dataframe_rows(analysis: RunnerImageAnalysis) -> list[dict[str, str]]:
    rows = list(analysis["findings"])
    rows.append(
        {
            "項目": "推定信頼度",
            "推定": f"{analysis['confidence']:.2f}",
            "理由": "画像のエッジ密度、小領域数、観察タグ数から算出した簡易指標です。",
        }
    )
    return rows


def _brightness(pixel: tuple[int, int, int]) -> int:
    red, green, blue = pixel
    return int(0.299 * red + 0.587 * green + 0.114 * blue)


def _saturation(pixel: tuple[int, int, int]) -> int:
    return max(pixel) - min(pixel)


def _edge_mask(brightness: list[int], width: int, height: int) -> list[bool]:
    mask = [False] * (width * height)
    threshold = 34
    for y in range(height - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width - 1):
            index = row + x
            current = brightness[index]
            if abs(current - brightness[index + 1]) >= threshold or abs(current - brightness[next_row + x]) >= threshold:
                mask[index] = True
    return mask


def _region_edge_ratio(edge_mask: list[bool], width: int, height: int, *, region: str) -> float:
    total = 0
    edges = 0
    x_min = int(width * 0.25)
    x_max = int(width * 0.75)
    y_min = int(height * 0.25)
    y_max = int(height * 0.75)
    border = max(3, int(min(width, height) * 0.12))

    for y in range(height):
        for x in range(width):
            is_target = (
                x_min <= x <= x_max and y_min <= y <= y_max
                if region == "center"
                else x < border or x >= width - border or y < border or y >= height - border
            )
            if not is_target:
                continue
            total += 1
            if edge_mask[y * width + x]:
                edges += 1
    return edges / max(1, total)


def _active_edge_cells(edge_mask: list[bool], width: int, height: int, *, grid_size: int) -> list[list[bool]]:
    cells: list[list[bool]] = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    counts: list[list[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    totals: list[list[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    for y in range(height):
        cell_y = min(grid_size - 1, int(y * grid_size / max(1, height)))
        for x in range(width):
            cell_x = min(grid_size - 1, int(x * grid_size / max(1, width)))
            totals[cell_y][cell_x] += 1
            if edge_mask[y * width + x]:
                counts[cell_y][cell_x] += 1

    for y in range(grid_size):
        for x in range(grid_size):
            cells[y][x] = counts[y][x] / max(1, totals[y][x]) >= 0.035
    return cells


def _count_components(cells: list[list[bool]]) -> int:
    grid_size = len(cells)
    visited = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    components = 0
    for y in range(grid_size):
        for x in range(grid_size):
            if visited[y][x] or not cells[y][x]:
                continue
            components += 1
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y][x] = True
            while queue:
                current_x, current_y = queue.popleft()
                for next_x, next_y in _neighbors(current_x, current_y, grid_size):
                    if not visited[next_y][next_x] and cells[next_y][next_x]:
                        visited[next_y][next_x] = True
                        queue.append((next_x, next_y))
    return components


def _thin_line_score(cells: list[list[bool]]) -> float:
    grid_size = len(cells)
    active = 0
    sparse_neighbors = 0
    for y in range(grid_size):
        for x in range(grid_size):
            if not cells[y][x]:
                continue
            active += 1
            neighbor_count = sum(1 for next_x, next_y in _neighbors(x, y, grid_size) if cells[next_y][next_x])
            if neighbor_count <= 2:
                sparse_neighbors += 1
    return sparse_neighbors / max(1, active)


def _neighbors(x: int, y: int, grid_size: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        next_x = x + offset_x
        next_y = y + offset_y
        if 0 <= next_x < grid_size and 0 <= next_y < grid_size:
            output.append((next_x, next_y))
    return output


def _infer_observations(
    *,
    edge_density: float,
    central_edge_ratio: float,
    border_edge_ratio: float,
    component_count: int,
    thin_line_score: float,
) -> list[str]:
    observations: list[str] = []
    if component_count >= 4 or edge_density >= 0.115:
        observations.append("many_gate_points")
    if component_count >= 3 or edge_density >= 0.095:
        observations.append("small_part")
    if central_edge_ratio >= 0.055 and edge_density >= 0.045:
        observations.append("visible_gate_mark")
    if thin_line_score >= 0.44 and edge_density >= 0.035:
        observations.append("thin_or_fragile")
    if thin_line_score >= 0.58 and border_edge_ratio >= central_edge_ratio:
        observations.append("tip_gate")
    if not observations:
        observations.append("looks_safe")
    return observations


def _infer_part_size(edge_density: float, component_count: int, thin_line_score: float) -> str:
    if component_count >= 4 or edge_density >= 0.11 or thin_line_score >= 0.58:
        return "small"
    if component_count >= 2 or edge_density >= 0.045:
        return "medium"
    return "large"


def _infer_gate_position(central_edge_ratio: float, border_edge_ratio: float, thin_line_score: float) -> str:
    if thin_line_score >= 0.6 and border_edge_ratio >= central_edge_ratio:
        return "tip"
    if central_edge_ratio >= border_edge_ratio * 1.15 and central_edge_ratio >= 0.045:
        return "front"
    if border_edge_ratio >= central_edge_ratio * 1.3 and border_edge_ratio >= 0.05:
        return "side"
    return "hidden"


def _infer_estimated_load(edge_density: float, component_count: int, thin_line_score: float) -> str:
    if component_count >= 6 or edge_density >= 0.14 or thin_line_score >= 0.65:
        return "high"
    if component_count >= 2 or edge_density >= 0.05:
        return "medium"
    return "low"


def _infer_material_type(high_brightness_ratio: float, low_saturation_ratio: float) -> str:
    if high_brightness_ratio >= 0.45 and low_saturation_ratio >= 0.45:
        return "soft_plastic"
    return "PS"


def _infer_part_area(observations: list[str]) -> str:
    if "many_gate_points" in observations or "visible_gate_mark" in observations:
        return "gate_area"
    if "thin_or_fragile" in observations or "tip_gate" in observations:
        return "antenna"
    if "small_part" in observations:
        return "hand_parts"
    return "gate_area"


def _confidence(edge_density: float, component_count: int, observations: list[str]) -> float:
    signal_strength = min(0.5, edge_density * 2.2) + min(0.35, component_count * 0.05)
    observation_bonus = min(0.15, max(0, len(observations) - 1) * 0.05)
    return round(max(0.2, min(0.95, signal_strength + observation_bonus)), 2)


def _build_findings(
    *,
    part_size: str,
    gate_position: str,
    material_type: str,
    observations: list[str],
    edge_density: float,
    component_count: int,
    thin_line_score: float,
) -> list[dict[str, str]]:
    return [
        {
            "項目": "部品サイズ候補",
            "推定": part_size,
            "理由": f"エッジ密度 {edge_density:.3f}、小領域候補 {component_count} 件から推定。",
        },
        {
            "項目": "ゲート位置候補",
            "推定": gate_position,
            "理由": "中央/外周のエッジ分布と細線らしさから推定。",
        },
        {
            "項目": "材料候補",
            "推定": material_type,
            "理由": "明るさと彩度の分布から、透明/淡色パーツらしさを簡易推定。",
        },
        {
            "項目": "観察タグ候補",
            "推定": ", ".join(observations),
            "理由": f"細線スコア {thin_line_score:.3f} と小領域候補から自動提案。",
        },
    ]
