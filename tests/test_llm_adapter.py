from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from llm_adapter import (
    LocalLLMEndpointError,
    build_local_llm_prompt,
    generate_local_llm_analysis,
    get_local_llm_status,
    validate_local_llm_endpoint,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class LocalLLMAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "feedback_id": "F-001",
            "kit_name": "Star Frame Alpha",
            "inspection_phase": "assembled_state",
            "part_area": "shoulder_joint",
            "joint_type": "hinge",
            "part_size": "medium",
            "material_type": "ABS",
            "moving_part": "true",
            "gate_position": "hidden",
            "estimated_load": "high",
            "assembly_step": "18",
            "user_level": "beginner",
            "feedback_text": "肩の関節が固く、白化しそうで怖かった。",
            "issue_categories": ["tight_joint", "breakage_risk"],
            "severity": "high",
            "risk_score": 94,
            "risk_explanation": {
                "breakdown": [
                    {
                        "factor_type": "issue_category",
                        "factor": "tight_joint",
                        "points": 25,
                        "reason": "tight_joint が検出されたため",
                    }
                ]
            },
            "cause_candidates": ["軸径と受け側クリアランスの確認が必要"],
            "improvement_suggestions": {"設計": ["軸受け入口にRを追加する"]},
        }

    def test_prompt_keeps_score_rule_based(self) -> None:
        prompt = build_local_llm_prompt(self.record)

        self.assertIn("risk_score はルールベースで計算済みなので、再採点しない", prompt)
        self.assertIn("shoulder_joint", prompt)
        self.assertIn("肩の関節が固く", prompt)

    def test_generate_local_llm_analysis_posts_to_ollama_endpoint(self) -> None:
        captured_payload = {}

        def fake_urlopen(request, timeout):
            captured_payload.update(json.loads(request.data.decode("utf-8")))
            return _FakeResponse({"response": "## ローカルAI補足\n確認対象を整理します。"})

        with patch("llm_adapter.urllib.request.urlopen", side_effect=fake_urlopen):
            result = generate_local_llm_analysis(
                self.record,
                endpoint="http://localhost:11434/api/generate",
                model="test-model",
                timeout=1,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(captured_payload["model"], "test-model")
        self.assertFalse(captured_payload["stream"])
        self.assertIn("ローカルAI補足", result["content"])

    def test_generate_local_llm_analysis_returns_error_without_exception(self) -> None:
        with patch("llm_adapter.urllib.request.urlopen", side_effect=OSError("connection failed")):
            result = generate_local_llm_analysis(self.record, timeout=1)

        self.assertFalse(result["ok"])
        self.assertIn("connection failed", result["error"])

    def test_get_local_llm_status_lists_models(self) -> None:
        with patch(
            "llm_adapter.urllib.request.urlopen",
            return_value=_FakeResponse({"models": [{"name": "llama3.1:8b"}]}),
        ):
            status = get_local_llm_status(endpoint="http://localhost:11434/api/generate", timeout=1)

        self.assertTrue(status["available"])
        self.assertEqual(status["models"], ["llama3.1:8b"])
        self.assertEqual(status["endpoint"], "http://localhost:11434/api/tags")

    def test_rejects_non_loopback_or_nonstandard_ollama_endpoints(self) -> None:
        endpoints = [
            "https://example.com/api/generate",
            "http://169.254.169.254:11434/api/generate",
            "http://127.0.0.2:11434/api/generate",
            "http://localhost:8080/api/generate",
            "http://localhost:11434/other",
            "http://localhost:11434/api/generate/",
            "http://user@localhost:11434/api/generate",
            "http://localhost:11434/api/generate?model=test",
        ]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(LocalLLMEndpointError):
                    validate_local_llm_endpoint(endpoint)

    def test_accepts_only_explicit_standard_loopback_endpoints(self) -> None:
        endpoints = [
            "http://localhost:11434/api/generate",
            "http://127.0.0.1:11434/api/generate",
            "http://[::1]:11434/api/generate",
        ]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(validate_local_llm_endpoint(endpoint), endpoint)

    def test_invalid_endpoint_never_starts_network_request(self) -> None:
        with patch("llm_adapter.urllib.request.urlopen") as mocked_urlopen:
            result = generate_local_llm_analysis(
                self.record,
                endpoint="http://169.254.169.254:11434/api/generate",
                timeout=1,
            )

        self.assertFalse(result["ok"])
        self.assertIn("localhost:11434", result["error"])
        mocked_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
