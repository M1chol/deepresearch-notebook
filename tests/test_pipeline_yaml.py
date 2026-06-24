from __future__ import annotations

import unittest

from research_pipeline.config import default_config
from research_pipeline.pipeline import ResearchPipeline


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def chat(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return next(self.responses)


class PipelineYamlTests(unittest.TestCase):
    def test_yaml_workflow_uses_unselected_questions_when_limit_is_absent(self) -> None:
        client = FakeClient(
            [
                "Notes\n```text\nTopic intro\n```",
                "Notes\n```text\n1. First question\n2. Second question\n```",
                "# Plan\n\n## Step\n- Point",
            ]
        )

        result = ResearchPipeline(client, default_config()).run("Topic")

        self.assertIsNone(result.selected_questions)
        self.assertEqual(result.topic_intro.extracted, "Topic intro")
        self.assertIn(
            "1. First question",
            client.calls[-1]["messages"][1].content,
        )

    def test_yaml_workflow_runs_declared_fallback_and_selection(self) -> None:
        client = FakeClient(
            [
                "Topic response without a fence",
                "```text\nFallback intro\n```",
                "Question response without a fence",
                "```text\n1. One\n2. Two\n```",
                "```text\n2. Two\n```",
                "# Plan\n\n## Step\n- Point",
            ]
        )
        config = default_config(max_questions=1)

        result = ResearchPipeline(client, config).run("Topic")

        self.assertTrue(result.topic_intro.metadata["fallback_used"])
        self.assertTrue(result.questions.metadata["fallback_used"])
        self.assertEqual(result.selected_questions.extracted, "2. Two")
        self.assertIn("2. Two", client.calls[-1]["messages"][1].content)

    def test_mid_step_explicitly_disables_reasoning(self) -> None:
        client = FakeClient(["summary"])
        pipeline = ResearchPipeline(client, default_config())

        pipeline.call_step(
            "summarize_step",
            step="Step",
            point_results="Evidence",
        )

        self.assertEqual(client.calls[0]["reasoning"], {"effort": "none"})

    def test_research_config_declares_retries_and_final_report(self) -> None:
        pipeline = ResearchPipeline(FakeClient([]), default_config())
        config = pipeline.prompts.get_research_config()

        self.assertEqual(config["max_query_attempts"], 3)
        self.assertEqual(config["query_prompt"], "construct_search_query")
        self.assertEqual(config["broaden_query_prompt"], "broaden_search_query")
        self.assertEqual(config["final_step"]["name"], "final_report")
        self.assertEqual(config["final_step"]["prompt"], "final_report")


if __name__ == "__main__":
    unittest.main()
