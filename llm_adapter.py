"""Optional local LLM adapter for Ollama-based analysis support."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, TypedDict


DEFAULT_LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:11434/api/generate")
DEFAULT_LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")
DEFAULT_LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "60"))


class LocalLLMResult(TypedDict):
    ok: bool
    model: str
    endpoint: str
    content: str
    error: str


class LocalLLMStatus(TypedDict):
    available: bool
    endpoint: str
    models: list[str]
    error: str


def generate_local_llm_analysis(
    record: Mapping[str, Any],
    *,
    endpoint: str = DEFAULT_LOCAL_LLM_ENDPOINT,
    model: str = DEFAULT_LOCAL_LLM_MODEL,
    timeout: float = DEFAULT_LOCAL_LLM_TIMEOUT,
) -> LocalLLMResult:
    """Call a local Ollama-compatible endpoint and return a Markdown analysis.

    The LLM is intentionally used only for explanation and ideation. Risk score
    calculation remains rule-based so the dashboard stays explainable.
    """

    prompt = build_local_llm_prompt(record)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    try:
        response_data = _post_json(endpoint, payload, timeout=timeout)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
        return {
            "ok": False,
            "model": model,
            "endpoint": endpoint,
            "content": "",
            "error": _friendly_error(error),
        }

    content = str(response_data.get("response", "")).strip()
    if not content:
        return {
            "ok": False,
            "model": model,
            "endpoint": endpoint,
            "content": "",
            "error": "ローカルモデルから空の応答が返りました。",
        }

    return {
        "ok": True,
        "model": model,
        "endpoint": endpoint,
        "content": content,
        "error": "",
    }


def get_local_llm_status(
    *,
    endpoint: str = DEFAULT_LOCAL_LLM_ENDPOINT,
    timeout: float = 3.0,
) -> LocalLLMStatus:
    """Check whether the local Ollama server responds and list local models."""

    tags_endpoint = _ollama_tags_endpoint(endpoint)
    request = urllib.request.Request(tags_endpoint, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
        return {"available": False, "endpoint": tags_endpoint, "models": [], "error": _friendly_error(error)}

    models = [
        str(item.get("name", ""))
        for item in data.get("models", [])
        if isinstance(item, Mapping) and item.get("name")
    ]
    return {"available": True, "endpoint": tags_endpoint, "models": models, "error": ""}


def build_local_llm_prompt(record: Mapping[str, Any]) -> str:
    issue_categories = ", ".join(_as_list(record.get("issue_categories")))
    cause_candidates = "\n".join(f"- {item}" for item in _as_list(record.get("cause_candidates"))[:4])
    score_breakdown = _format_score_breakdown(record)
    improvement_suggestions = _format_suggestions(record.get("improvement_suggestions"))

    return f"""あなたはプラモデル開発改善を支援するレビュー分析アシスタントです。

前提:
- 実在する企業名、商品名、キャラクター名、公式データ、実際の金型データには触れない。
- 入力はすべて架空のプラモデルと架空レビューである。
- risk_score はルールベースで計算済みなので、再採点しない。
- 断定しすぎず、「確認対象」「仮説」「検証すべき観点」として書く。
- 出力は日本語Markdown。長すぎないが、設計者が次に見るポイントが分かる具体性にする。

対象レコード:
- feedback_id: {record.get("feedback_id", "-")}
- kit_name: {record.get("kit_name", "-")}
- inspection_phase: {record.get("inspection_phase", "-")}
- part_area: {record.get("part_area", "-")}
- joint_type: {record.get("joint_type", "-")}
- part_size: {record.get("part_size", "-")}
- material_type: {record.get("material_type", "-")}
- moving_part: {record.get("moving_part", "-")}
- gate_position: {record.get("gate_position", "-")}
- estimated_load: {record.get("estimated_load", "-")}
- assembly_step: {record.get("assembly_step", "-")}
- user_level: {record.get("user_level", "-")}
- feedback_text: {record.get("feedback_text", "-")}

ルールベース分析:
- issue_categories: {issue_categories or "-"}
- severity: {record.get("severity", "-")}
- risk_score: {record.get("risk_score", "-")}

スコア加点理由:
{score_breakdown}

既存の原因候補:
{cause_candidates or "-"}

既存の一般改善案:
{improvement_suggestions or "-"}

出力フォーマット:
## ローカルAI補足
### 1. 読み取り
### 2. 追加の推定原因
### 3. 具体的な改善案
### 4. 検証方法
### 5. ルールベース分析との差分
"""


def _post_json(endpoint: str, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ollama_tags_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    path = "/api/tags"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _format_score_breakdown(record: Mapping[str, Any]) -> str:
    breakdown = record.get("risk_explanation", {}).get("breakdown", [])
    if not isinstance(breakdown, list):
        return "-"
    rows = []
    for item in breakdown[:8]:
        if not isinstance(item, Mapping):
            continue
        points = item.get("points", 0)
        if int(points) == 0:
            continue
        rows.append(
            f"- +{points} / {item.get('factor_type', '-')} / {item.get('factor', '-')} / {item.get('reason', '-')}"
        )
    return "\n".join(rows) or "-"


def _format_suggestions(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "-"
    rows = []
    for viewpoint, suggestions in value.items():
        for suggestion in _as_list(suggestions)[:3]:
            rows.append(f"- {viewpoint}: {suggestion}")
    return "\n".join(rows[:10]) or "-"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _friendly_error(error: BaseException) -> str:
    message = str(error)
    if "Connection refused" in message or "対象のコンピューターによって拒否" in message:
        return "Ollamaに接続できません。`ollama serve` が起動しているか確認してください。"
    if isinstance(error, urllib.error.HTTPError):
        return f"Ollama APIエラー: HTTP {error.code}"
    return message or error.__class__.__name__
