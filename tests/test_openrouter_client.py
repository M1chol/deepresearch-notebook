from __future__ import annotations

import json
import unittest

import httpx

from research_pipeline.config import default_config
from research_pipeline.models import ChatMessage
from research_pipeline.openrouter_client import OpenRouterClient


class OpenRouterClientTests(unittest.TestCase):
    def test_records_usage_cost_latency_and_reasoning_configuration(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "test/mid",
                    "choices": [{"message": {"content": "summary"}}],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                        "total_tokens": 18,
                        "cost": 0.004,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

        client = OpenRouterClient("key", default_config())
        client.client.close()
        client.client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            content = client.chat(
                model="test/mid",
                messages=[ChatMessage(role="user", content="summarize")],
                reasoning={"effort": "none"},
                operation="summarize_step",
            )
        finally:
            client.close()

        self.assertEqual(content, "summary")
        self.assertEqual(captured["reasoning"], {"effort": "none"})
        self.assertEqual(client.stats[0].total_tokens, 18)
        self.assertEqual(client.stats[0].cost, 0.004)
        self.assertGreater(client.stats[0].response_seconds, 0)
        self.assertGreater(client.stats[0].tokens_per_second, 0)


if __name__ == "__main__":
    unittest.main()
