"""Lightweight local image heuristics for runner inspection inputs."""

from __future__ import annotations

from collections import deque
from io import BytesIO
from typing import TypedDict

from PIL import Image, ImageDraw

CropBoxRatio = tuple[float, float, float, float]


class RunnerRoiSuggestion(TypedDict):
    crop_box: CropBoxRatio
    confidence: float
    reason: str
    quality_notes: list[str]
    metrics: dict[str, float | int]


class RunnerImageAnalysis(TypedDict):
    part_area: str
    part_size: str
    material_type: str
    gate_position: str
    estimated_load: str
    observations: list[str]
    quality_warnings: list[str]
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
    high_saturation_ratio = sum(1 for value in saturation if value >= 80) / max(1, len(saturation))

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
    quality_warnings = _infer_quality_warnings(
        edge_density=edge_density,
        component_count=component_count,
        high_brightness_ratio=high_brightness_ratio,
        low_saturation_ratio=low_saturation_ratio,
        high_saturation_ratio=high_saturation_ratio,
    )

    return {
        "part_area": part_area,
        "part_size": part_size,
        "material_type": material_type,
        "gate_position": gate_position,
        "estimated_load": estimated_load,
        "observations": observations,
        "quality_warnings": quality_warnings,
        "confidence": confidence,
        "metrics": {
            "edge_density": round(edge_density, 4),
            "central_edge_ratio": round(central_edge_ratio, 4),
            "border_edge_ratio": round(border_edge_ratio, 4),
            "active_cell_components": component_count,
            "thin_line_score": round(thin_line_score, 4),
            "high_brightness_ratio": round(high_brightness_ratio, 4),
            "low_saturation_ratio": round(low_saturation_ratio, 4),
            "high_saturation_ratio": round(high_saturation_ratio, 4),
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
    for warning in analysis["quality_warnings"]:
        rows.append(
            {
                "項目": "解析品質注意",
                "推定": "要確認",
                "理由": warning,
            }
        )
    rows.append(
        {
            "項目": "推定信頼度",
            "推定": f"{analysis['confidence']:.2f}",
            "理由": "画像のエッジ密度、小領域数、観察タグ数から算出した簡易指標です。",
        }
    )
    return rows


def crop_runner_image(image_bytes: bytes, crop_box: CropBoxRatio) -> bytes:
    """Return a PNG containing only the selected region of interest."""

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    left, top, right, bottom = _crop_box_to_pixels(crop_box, image.width, image.height)
    cropped = image.crop((left, top, right, bottom))
    output = BytesIO()
    cropped.save(output, format="PNG")
    return output.getvalue()


def suggest_runner_roi(image_bytes: bytes) -> RunnerRoiSuggestion:
    """Suggest a runner-like region of interest from an uploaded image."""

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((260, 260))
    width, height = image.size
    pixels = list(image.getdata())
    brightness = [_brightness(pixel) for pixel in pixels]
    saturation = [_saturation(pixel) for pixel in pixels]
    edge_mask = _edge_mask(brightness, width, height)
    scores = _cell_edge_scores(edge_mask, width, height, grid_size=20)
    saturation_scores = _cell_saturation_scores(saturation, width, height, grid_size=20)
    active_cells = _roi_active_cells(scores)
    components = _active_components(active_cells)

    if not components:
        return {
            "crop_box": (0.0, 0.0, 1.0, 1.0),
            "confidence": 0.2,
            "reason": "ランナーらしい連続エッジ領域が弱いため、画像全体を候補にしました。",
            "quality_notes": ["対象ランナーが小さい、背景と同化している、またはピントが甘い可能性があります。"],
            "metrics": {"component_count": 0, "selected_cell_count": 0, "coverage_ratio": 1.0},
        }

    selected_component = max(components, key=len)
    crop_box = _component_crop_box(selected_component, grid_size=20, margin_cells=1)
    coverage_ratio = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
    selected_scores = [scores[y][x] for x, y in selected_component]
    selected_saturation_scores = [saturation_scores[y][x] for x, y in selected_component]
    mean_edge_score = sum(selected_scores) / max(1, len(selected_scores))
    mean_high_saturation = sum(selected_saturation_scores) / max(1, len(selected_saturation_scores))

    confidence = 0.35 + min(0.3, mean_edge_score * 2.6) + min(0.2, len(selected_component) / 60)
    quality_notes: list[str] = []
    if len(components) >= 3:
        confidence -= 0.15
        confidence = min(confidence, 0.54)
        quality_notes.append("複数のエッジ塊があるため、箱・説明書・別ランナーが混在している可能性があります。")
    if coverage_ratio >= 0.85:
        confidence -= 0.12
        confidence = min(confidence, 0.55)
        quality_notes.append("候補範囲が画像の大部分を占めています。対象ランナーだけに寄せると精度が上がります。")
    if mean_high_saturation >= 0.18:
        confidence -= 0.12
        confidence = min(confidence, 0.52)
        quality_notes.append("候補範囲に色の強い印刷物やシールが含まれている可能性があります。")

    confidence = round(max(0.2, min(0.95, confidence)), 2)
    if not quality_notes:
        quality_notes.append("最大の連続エッジ領域をランナー候補として提案しました。必要ならスライダーで微調整してください。")

    return {
        "crop_box": crop_box,
        "confidence": confidence,
        "reason": "画像内のエッジ密度が高く、連続している領域をランナー候補として抽出しました。",
        "quality_notes": quality_notes,
        "metrics": {
            "component_count": len(components),
            "selected_cell_count": len(selected_component),
            "coverage_ratio": round(coverage_ratio, 4),
            "mean_edge_score": round(mean_edge_score, 4),
            "mean_high_saturation": round(mean_high_saturation, 4),
        },
    }


def render_roi_preview(image_bytes: bytes, crop_box: CropBoxRatio) -> bytes:
    """Return a preview image with the analysis ROI drawn on top of the original image."""

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    original_width, original_height = image.size
    left, top, right, bottom = _crop_box_to_pixels(crop_box, original_width, original_height)

    preview = image.copy()
    preview.thumbnail((900, 650))
    scale_x = preview.width / max(1, original_width)
    scale_y = preview.height / max(1, original_height)
    preview_box = (
        int(left * scale_x),
        int(top * scale_y),
        int(right * scale_x),
        int(bottom * scale_y),
    )

    draw = ImageDraw.Draw(preview)
    for offset in range(4):
        draw.rectangle(
            (
                preview_box[0] - offset,
                preview_box[1] - offset,
                preview_box[2] + offset,
                preview_box[3] + offset,
            ),
            outline=(14, 165, 233),
        )
    draw.rectangle((preview_box[0], max(0, preview_box[1] - 20), preview_box[0] + 96, preview_box[1]), fill=(20, 20, 20))
    draw.text((preview_box[0] + 4, max(0, preview_box[1] - 17)), "analysis ROI", fill=(255, 255, 255))

    output = BytesIO()
    preview.save(output, format="PNG")
    return output.getvalue()


def render_runner_detection_overlay(image_bytes: bytes, analysis: RunnerImageAnalysis | None = None) -> bytes:
    """Return a PNG preview with heuristic risk-candidate boxes drawn over the image."""

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    preview = image.copy()
    preview.thumbnail((900, 650))
    draw = ImageDraw.Draw(preview)
    width, height = preview.size
    current_analysis = analysis or analyze_runner_image(image_bytes)
    observations = set(current_analysis["observations"])

    boxes = _overlay_boxes(preview, observations, str(current_analysis["gate_position"]))
    for box in boxes:
        _draw_labeled_box(draw, box)

    output = BytesIO()
    preview.save(output, format="PNG")
    return output.getvalue()


def _overlay_boxes(
    image: Image.Image,
    observations: set[str],
    gate_position: str,
) -> list[dict[str, int | str]]:
    width, height = image.size
    regions = _image_candidate_regions(image)
    boxes: list[dict[str, int | str]] = []
    if "visible_gate_mark" in observations or gate_position == "front":
        region = _select_gate_region(regions, width, height, gate_position)
        boxes.append(
            {
                "x1": region["x1"],
                "y1": region["y1"],
                "x2": region["x2"],
                "y2": region["y2"],
                "label": "gate",
                "color": "red",
            }
        )
    if "many_gate_points" in observations or "small_part" in observations:
        for region in regions[:2] or [_fallback_region(width, height)]:
            boxes.append(
                {
                    "x1": region["x1"],
                    "y1": region["y1"],
                    "x2": region["x2"],
                    "y2": region["y2"],
                    "label": "small parts",
                    "color": "orange",
                }
            )
    if "thin_or_fragile" in observations or "tip_gate" in observations:
        region = _select_thin_region(regions, width, height)
        boxes.append(
            {
                "x1": region["x1"],
                "y1": region["y1"],
                "x2": region["x2"],
                "y2": region["y2"],
                "label": "thin area",
                "color": "red",
            }
        )
    if not boxes:
        boxes.append(
            {
                "x1": int(width * 0.18),
                "y1": int(height * 0.18),
                "x2": int(width * 0.82),
                "y2": int(height * 0.82),
                "label": "low risk",
                "color": "green",
            }
        )
    return boxes


def _crop_box_to_pixels(crop_box: CropBoxRatio, width: int, height: int) -> tuple[int, int, int, int]:
    left_ratio, top_ratio, right_ratio, bottom_ratio = crop_box
    left_ratio = _clamp_ratio(left_ratio)
    top_ratio = _clamp_ratio(top_ratio)
    right_ratio = _clamp_ratio(right_ratio)
    bottom_ratio = _clamp_ratio(bottom_ratio)

    if right_ratio - left_ratio < 0.05:
        left_ratio, right_ratio = 0.0, 1.0
    if bottom_ratio - top_ratio < 0.05:
        top_ratio, bottom_ratio = 0.0, 1.0

    left = int(width * left_ratio)
    top = int(height * top_ratio)
    right = max(left + 1, int(width * right_ratio))
    bottom = max(top + 1, int(height * bottom_ratio))
    return left, top, min(width, right), min(height, bottom)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _image_candidate_regions(image: Image.Image, *, grid_size: int = 24) -> list[dict[str, int | float]]:
    """Find image-specific high-edge regions that can be used as visual candidates."""

    width, height = image.size
    brightness = [_brightness(pixel) for pixel in image.getdata()]
    edge_mask = _edge_mask(brightness, width, height)
    scores = _cell_edge_scores(edge_mask, width, height, grid_size=grid_size)
    return _candidate_regions_from_scores(scores, width, height, max_regions=5)


def _cell_edge_scores(edge_mask: list[bool], width: int, height: int, *, grid_size: int) -> list[list[float]]:
    counts: list[list[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    totals: list[list[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    for y in range(height):
        cell_y = min(grid_size - 1, int(y * grid_size / max(1, height)))
        for x in range(width):
            cell_x = min(grid_size - 1, int(x * grid_size / max(1, width)))
            totals[cell_y][cell_x] += 1
            if edge_mask[y * width + x]:
                counts[cell_y][cell_x] += 1
    return [
        [counts[y][x] / max(1, totals[y][x]) for x in range(grid_size)]
        for y in range(grid_size)
    ]


def _cell_saturation_scores(saturation: list[int], width: int, height: int, *, grid_size: int) -> list[list[float]]:
    saturated_counts: list[list[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    totals: list[list[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    for y in range(height):
        cell_y = min(grid_size - 1, int(y * grid_size / max(1, height)))
        for x in range(width):
            cell_x = min(grid_size - 1, int(x * grid_size / max(1, width)))
            totals[cell_y][cell_x] += 1
            if saturation[y * width + x] >= 80:
                saturated_counts[cell_y][cell_x] += 1
    return [
        [saturated_counts[y][x] / max(1, totals[y][x]) for x in range(grid_size)]
        for y in range(grid_size)
    ]


def _roi_active_cells(scores: list[list[float]]) -> list[list[bool]]:
    grid_size = len(scores)
    flat_scores = [score for row in scores for score in row]
    sorted_scores = sorted(flat_scores)
    percentile_index = int(len(sorted_scores) * 0.74)
    threshold = max(0.028, sorted_scores[min(percentile_index, len(sorted_scores) - 1)])
    return [
        [scores[y][x] >= threshold for x in range(grid_size)]
        for y in range(grid_size)
    ]


def _active_components(cells: list[list[bool]]) -> list[list[tuple[int, int]]]:
    grid_size = len(cells)
    visited = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    components: list[list[tuple[int, int]]] = []
    for y in range(grid_size):
        for x in range(grid_size):
            if visited[y][x] or not cells[y][x]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y][x] = True
            while queue:
                current_x, current_y = queue.popleft()
                component.append((current_x, current_y))
                for next_x, next_y in _neighbors(current_x, current_y, grid_size):
                    if not visited[next_y][next_x] and cells[next_y][next_x]:
                        visited[next_y][next_x] = True
                        queue.append((next_x, next_y))
            components.append(component)
    return components


def _component_crop_box(
    component: list[tuple[int, int]],
    *,
    grid_size: int,
    margin_cells: int,
) -> CropBoxRatio:
    xs = [x for x, _ in component]
    ys = [y for _, y in component]
    left = max(0, min(xs) - margin_cells) / grid_size
    top = max(0, min(ys) - margin_cells) / grid_size
    right = min(grid_size, max(xs) + margin_cells + 1) / grid_size
    bottom = min(grid_size, max(ys) + margin_cells + 1) / grid_size
    if right - left < 0.12:
        center = (left + right) / 2
        left = max(0.0, center - 0.06)
        right = min(1.0, center + 0.06)
    if bottom - top < 0.12:
        center = (top + bottom) / 2
        top = max(0.0, center - 0.06)
        bottom = min(1.0, center + 0.06)
    return (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))


def _candidate_regions_from_scores(
    scores: list[list[float]],
    width: int,
    height: int,
    *,
    max_regions: int,
) -> list[dict[str, int | float]]:
    grid_size = len(scores)
    flat_scores = [score for row in scores for score in row]
    sorted_scores = sorted(flat_scores)
    percentile_index = int(len(sorted_scores) * 0.82)
    threshold = max(0.035, sorted_scores[min(percentile_index, len(sorted_scores) - 1)])
    cells: list[tuple[int, int, float]] = [
        (x, y, scores[y][x])
        for y in range(grid_size)
        for x in range(grid_size)
        if scores[y][x] >= threshold
    ]
    cells.sort(key=lambda item: item[2], reverse=True)

    regions: list[dict[str, int | float]] = []
    cell_width = width / grid_size
    cell_height = height / grid_size
    for cell_x, cell_y, score in cells:
        center_x = int((cell_x + 0.5) * cell_width)
        center_y = int((cell_y + 0.5) * cell_height)
        if any(abs(center_x - int(region["cx"])) < width * 0.18 and abs(center_y - int(region["cy"])) < height * 0.18 for region in regions):
            continue

        box_width = int(max(width * 0.16, cell_width * 4.5))
        box_height = int(max(height * 0.16, cell_height * 4.5))
        regions.append(
            {
                "x1": max(0, center_x - box_width // 2),
                "y1": max(0, center_y - box_height // 2),
                "x2": min(width - 1, center_x + box_width // 2),
                "y2": min(height - 1, center_y + box_height // 2),
                "cx": center_x,
                "cy": center_y,
                "score": score,
            }
        )
        if len(regions) >= max_regions:
            break

    return regions


def _select_gate_region(
    regions: list[dict[str, int | float]],
    width: int,
    height: int,
    gate_position: str,
) -> dict[str, int | float]:
    if not regions:
        return _fallback_region(width, height)
    center_x = width / 2
    center_y = height / 2
    if gate_position == "front":
        return min(
            regions,
            key=lambda region: abs(float(region["cx"]) - center_x) + abs(float(region["cy"]) - center_y),
        )
    if gate_position in {"side", "tip"}:
        return min(
            regions,
            key=lambda region: min(
                float(region["cx"]),
                abs(width - float(region["cx"])),
                float(region["cy"]),
                abs(height - float(region["cy"])),
            ),
        )
    return max(regions, key=lambda region: float(region["score"]))


def _select_thin_region(
    regions: list[dict[str, int | float]],
    width: int,
    height: int,
) -> dict[str, int | float]:
    if not regions:
        return _fallback_region(width, height)
    return max(
        regions,
        key=lambda region: (
            abs((float(region["x2"]) - float(region["x1"])) - (float(region["y2"]) - float(region["y1"]))),
            float(region["score"]),
        ),
    )


def _fallback_region(width: int, height: int) -> dict[str, int | float]:
    return {
        "x1": int(width * 0.22),
        "y1": int(height * 0.22),
        "x2": int(width * 0.78),
        "y2": int(height * 0.78),
        "cx": int(width * 0.5),
        "cy": int(height * 0.5),
        "score": 0.0,
    }


def _draw_labeled_box(draw: ImageDraw.ImageDraw, box: dict[str, int | str]) -> None:
    color_name = str(box["color"])
    color = {"red": (230, 45, 45), "orange": (245, 158, 11), "green": (34, 197, 94)}.get(color_name, (230, 45, 45))
    x1 = int(box["x1"])
    y1 = int(box["y1"])
    x2 = int(box["x2"])
    y2 = int(box["y2"])
    label = str(box["label"])
    for offset in range(3):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    label_width = min(x2 - x1, max(72, len(label) * 8))
    label_box = (x1, max(0, y1 - 18), x1 + label_width, y1)
    draw.rectangle(label_box, fill=(20, 20, 20))
    draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(255, 255, 255))


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


def _infer_quality_warnings(
    *,
    edge_density: float,
    component_count: int,
    high_brightness_ratio: float,
    low_saturation_ratio: float,
    high_saturation_ratio: float,
) -> list[str]:
    warnings: list[str] = []
    if high_saturation_ratio >= 0.025 and low_saturation_ratio >= 0.35 and edge_density >= 0.04:
        warnings.append("シール、ラベル、説明書など色の強い印刷物が写り込み、ゲート候補として誤検出される可能性があります。")
    if component_count >= 6 and edge_density >= 0.08:
        warnings.append("複数ランナー、箱、背景物が混在している可能性があります。単体ランナーをトリミングすると精度が上がります。")
    if high_brightness_ratio >= 0.35 and edge_density >= 0.045:
        warnings.append("透明袋や強い反射が写っている可能性があります。袋から出すか、反射の少ない角度で撮影してください。")
    if low_saturation_ratio >= 0.85 and edge_density >= 0.055:
        warnings.append("透明パーツ、白いシール、台紙、袋反射の輪郭が混ざっている可能性があります。対象ランナーだけを切り出して確認してください。")
    return warnings


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
