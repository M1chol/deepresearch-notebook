from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_pipeline.cli import (
    _stats_totals,
    _write_nerd_stats,
    extract_research_title,
    slugify_research_name,
)
from research_pipeline.models import LLMCallStats


class CliTests(unittest.TestCase):
    def test_result_name_comes_from_research_title(self) -> None:
        title = extract_research_title(
            "# Distributed Version Control Systems\n\n## First step\n- Point"
        )
        self.assertEqual(title, "Distributed Version Control Systems")
        self.assertEqual(
            slugify_research_name(title or ""),
            "distributed-version-control-systems",
        )
        self.assertEqual(
            slugify_research_name(Path("custom_report.md").stem),
            "custom-report",
        )

    def test_nerd_stats_are_aggregated_and_persisted(self) -> None:
        stats = [
            LLMCallStats(
                operation="one",
                model="model",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                total_tokens=30,
                cost=0.01,
                response_seconds=2.0,
                tokens_per_second=10.0,
            ),
            LLMCallStats(
                operation="two",
                model="model",
                prompt_tokens=5,
                completion_tokens=10,
                reasoning_tokens=2,
                total_tokens=15,
                cost=0.02,
                response_seconds=1.0,
                tokens_per_second=10.0,
            ),
        ]
        totals = _stats_totals(stats)
        self.assertEqual(totals["total_tokens"], 45)
        self.assertEqual(totals["reasoning_tokens"], 2)
        self.assertEqual(totals["tokens_per_second"], 10.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            _write_nerd_stats(run_dir, stats)
            self.assertTrue((run_dir / "nerd_stats.json").exists())
            self.assertIn("Nerd stats", (run_dir / "nerd_stats.md").read_text())


if __name__ == "__main__":
    unittest.main()
