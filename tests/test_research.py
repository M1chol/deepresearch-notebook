from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_pipeline.models import SearchResponse, SearchResult
from research_pipeline.research import ResearchExecutor, parse_research_plan
from research_pipeline.search import extract_html_text


class ResearchPlanParsingTests(unittest.TestCase):
    def test_parses_h2_steps_and_bullets(self) -> None:
        plan = """# Plan

## First step
- First point
* Second point

### Detail ignored as a step

## Second step
+ Third point
"""
        self.assertEqual(
            [(step.title, step.points) for step in parse_research_plan(plan)],
            [
                ("First step", ["First point", "Second point"]),
                ("Second step", ["Third point"]),
            ],
        )

    def test_extract_html_text_omits_script_and_normalizes_content(self) -> None:
        html = """
        <html><head><style>.hidden {display:none}</style></head>
        <body><main><h1>Title &amp; subject</h1><p>Hello   world.</p>
        <script>doNotInclude()</script></main></body></html>
        """
        text = extract_html_text(html)
        self.assertIn("Title & subject", text)
        self.assertIn("Hello world.", text)
        self.assertNotIn("doNotInclude", text)


class ResearchExecutorTests(unittest.TestCase):
    def test_broadens_query_when_selector_finds_nothing_useful(self) -> None:
        plan = """## Step
- Narrow point
"""
        searches = []
        broaden_calls = []

        def search(query: str) -> SearchResponse:
            searches.append(query)
            return SearchResponse(
                query,
                [SearchResult(title="Source", url="https://example.test/source")],
            )

        def select(step, point, query, results):
            return [] if query == "overly specific query" else [0]

        def broaden(step, point, query, reason):
            broaden_calls.append((query, reason))
            return "broad query"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ResearchExecutor(
                search=search,
                scrape=lambda url: "page",
                build_query=lambda step, point: "overly specific query",
                broaden_query=broaden,
                select_results=select,
                summarize_page=lambda *args: "summary",
                summarize_step=lambda *args: "step summary",
                max_query_attempts=3,
            ).run_markdown(plan, temp_dir)

            point = result.steps[0].points[0]
            self.assertEqual(searches, ["overly specific query", "broad query"])
            self.assertEqual(point.selected_indices, [0])
            self.assertEqual(len(point.query_attempts), 2)
            self.assertIn("zero useful", broaden_calls[0][1])

    def test_executes_all_points_persists_artifacts_and_continues_fetch_failure(
        self,
    ) -> None:
        plan = """## Sources
- Alpha evidence
- Beta evidence
"""
        calls: dict[str, list] = {
            "queries": [],
            "selections": [],
            "pages": [],
            "steps": [],
        }

        def build_query(step: str, point: str) -> str:
            calls["queries"].append((step, point))
            return f"{point} query"

        def search(query: str) -> SearchResponse:
            return SearchResponse(
                query=query,
                results=[
                    SearchResult(title="Useful", url=f"https://example.test/{query}"),
                    SearchResult(title="Broken", url="https://example.test/broken"),
                    SearchResult(title="Unused", url="https://example.test/unused"),
                ],
            )

        def select(step, point, query, results):
            calls["selections"].append((step, point, query, len(results)))
            return json.dumps(
                {"selected_indices": [0, 1], "rationale": "first two are relevant"}
            )

        def scrape(url: str) -> str:
            if url.endswith("/broken"):
                raise RuntimeError("fetch failed")
            return f"Full page for {url}"

        def summarize_page(step, point, query, result, text):
            calls["pages"].append((point, result.url, text))
            return f"summary for {point}"

        def summarize_step(step, points):
            calls["steps"].append((step, len(points)))
            return "combined step summary"

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results" / "topic" / "searches"
            result = ResearchExecutor(
                search=search,
                scrape=scrape,
                build_query=build_query,
                select_results=select,
                summarize_page=summarize_page,
                summarize_step=summarize_step,
            ).run_markdown(plan, output)

            self.assertEqual(len(result.steps), 1)
            self.assertEqual(len(result.steps[0].points), 2)
            self.assertEqual(len(calls["queries"]), 2)
            self.assertEqual(len(calls["pages"]), 2)
            self.assertEqual(calls["steps"], [("Sources", 2)])
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "result.json").exists())

            for point in result.steps[0].points:
                self.assertEqual(point.selected_indices, [0, 1])
                self.assertEqual(len(point.pages), 2)
                self.assertIsNone(point.pages[0].error)
                self.assertIn("fetch failed", point.pages[1].error or "")
                self.assertTrue((point.artifact_dir / "search.json").exists())
                self.assertTrue((point.artifact_dir / "point.json").exists())

            summary = result.steps[0].artifact_dir / "step-summary.md"
            self.assertIn("combined step summary", summary.read_text())

            search_artifact = json.loads(
                (result.steps[0].points[0].artifact_dir / "search.json").read_text()
            )
            self.assertIn("rationale", search_artifact["selection"])

    def test_accepts_selected_urls_and_survives_query_failure(self) -> None:
        plan = """## Step
- Good
- Bad
"""

        def build_query(step: str, point: str) -> str:
            if point == "Bad":
                raise ValueError("cannot build")
            return point

        result_item = SearchResult(title="One", url="https://example.test/one")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ResearchExecutor(
                search=lambda query: SearchResponse(query, [result_item]),
                scrape=lambda url: "page",
                build_query=build_query,
                select_results=lambda step, point, query, results: [results[0].url],
                summarize_page=lambda *args: "page summary",
                summarize_step=lambda step, points: "step summary",
            ).run_markdown(plan, temp_dir)

            self.assertEqual(result.steps[0].points[0].selected_indices, [0])
            self.assertIn("cannot build", result.steps[0].points[1].error or "")
            self.assertEqual(result.steps[0].summary, "step summary")

    def test_persists_scrape_when_page_summary_fails(self) -> None:
        item = SearchResult(title="Page", url="https://example.test/page")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ResearchExecutor(
                search=lambda query: SearchResponse(query, [item]),
                scrape=lambda url: "important extracted text",
                build_query=lambda step, point: point,
                select_results=lambda step, point, query, results: [0],
                summarize_page=lambda *args: (_ for _ in ()).throw(
                    RuntimeError("model failed")
                ),
                summarize_step=lambda step, points: "step summary",
            ).run_markdown("## Step\n- Point\n", temp_dir)

            page = result.steps[0].points[0].pages[0]
            artifact = page.artifact_path.read_text()
            self.assertIn("important extracted text", artifact)
            self.assertIn("model failed", artifact)


if __name__ == "__main__":
    unittest.main()
