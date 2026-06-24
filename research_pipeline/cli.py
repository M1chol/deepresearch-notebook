from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path

from .config import default_config, load_api_key
from .models import LLMCallStats, PipelineResult
from .openrouter_client import OpenRouterClient
from .pipeline import ResearchPipeline
from .research import (
    PointResearchResult,
    ResearchExecutionResult,
    ResearchExecutor,
)
from .search import HTTPPageScraper, SearxNGClient


class TerminalStatus:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def task(self, message: str) -> None:
        elapsed = time.perf_counter() - self.started
        print(f"[{elapsed:7.1f}s] {message}", flush=True)

    def stats(self, stat: LLMCallStats) -> None:
        self.task(
            f"Nerd stats · {stat.operation} · {stat.model} · "
            f"{stat.prompt_tokens:,} in / {stat.completion_tokens:,} out · "
            f"${stat.cost:.6f} · {stat.response_seconds:.2f}s · "
            f"{stat.tokens_per_second:.1f} tok/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deep research pipeline")
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic. If omitted, --topic-file must be provided.",
    )
    parser.add_argument(
        "--topic-file",
        help="Path to a text or markdown file containing the topic.",
    )
    parser.add_argument(
        "--prompts",
        default="prompts.yaml",
        help="Path to prompts YAML file.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Select only the top N questions.",
    )
    parser.add_argument(
        "--human-in-loop",
        action="store_true",
        help="Allow manual edits after topic and question generation.",
    )
    parser.add_argument(
        "--name",
        "--output",
        dest="run_name",
        default=None,
        help=(
            "Optional results folder name. By default it is derived from the "
            "generated research title. --output is retained as a legacy alias."
        ),
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory for per-research artifacts.",
    )
    parser.add_argument(
        "--searxng-url",
        default="http://localhost:8080/search",
        help="SearxNG JSON search endpoint.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Generate the plan without search or final-report generation.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming for AI responses.",
    )
    args = parser.parse_args()

    if args.topic_file:
        topic = Path(args.topic_file).read_text(encoding="utf-8")
    elif args.topic:
        topic = args.topic
    else:
        raise SystemExit("Provide either a topic argument or --topic-file")

    status = TerminalStatus()
    status.task("Starting research planning")
    api_key = load_api_key()
    config = default_config(
        prompt_file=args.prompts,
        max_questions=args.max_questions,
        human_in_loop=args.human_in_loop,
        stream=args.stream,
    )

    with OpenRouterClient(api_key=api_key, config=config) as client:
        client.on_stats = status.stats
        pipeline = ResearchPipeline(client=client, config=config)
        pipeline.on_status = status.task
        planning_result = pipeline.run(topic)
        research_plan = planning_result.final_plan.extracted or ""

        research_title = extract_research_title(research_plan) or topic
        requested_name = Path(args.run_name).stem if args.run_name else research_title
        run_name = slugify_research_name(requested_name)
        run_dir = Path(args.results_root) / run_name
        planning_dir = run_dir / "planning"
        searches_dir = run_dir / "searches"
        run_dir.mkdir(parents=True, exist_ok=True)
        planning_dir.mkdir(parents=True, exist_ok=True)

        plan_path = run_dir / "research_plan.md"
        plan_path.write_text(research_plan, encoding="utf-8")
        _write_planning_artifacts(planning_dir, planning_result)
        status.task(f"Saved research plan: {plan_path}")

        research_result: ResearchExecutionResult | None = None
        final_report_path: Path | None = None
        if not args.skip_search:
            searches_dir.mkdir(parents=True, exist_ok=True)
            research_config = pipeline.prompts.get_research_config()
            broaden_prompt = str(
                research_config.get("broaden_query_prompt", "broaden_search_query")
            )
            query_prompt = str(
                research_config.get("query_prompt", "construct_search_query")
            )
            result_selection_prompt = str(
                research_config.get(
                    "result_selection_prompt",
                    "select_useful_results",
                )
            )
            page_summary_prompt = str(
                research_config.get("page_summary_prompt", "summarize_page")
            )
            step_summary_prompt = str(
                research_config.get("step_summary_prompt", "summarize_step")
            )
            max_query_attempts = int(
                research_config.get("max_query_attempts", 3)
            )

            with (
                SearxNGClient(base_url=args.searxng_url) as search_client,
                HTTPPageScraper() as scraper,
            ):
                executor = ResearchExecutor(
                    search=search_client.search,
                    scrape=scraper.scrape,
                    build_query=lambda step, point: pipeline.call_step(
                        query_prompt,
                        step=step,
                        point=point,
                    ),
                    broaden_query=lambda step, point, query, reason: (
                        pipeline.call_step(
                            broaden_prompt,
                            step=step,
                            point=point,
                            query=query,
                            reason=reason,
                        )
                    ),
                    select_results=lambda step, point, query, results: (
                        pipeline.call_step(
                            result_selection_prompt,
                            step=step,
                            point=point,
                            query=query,
                            results=json.dumps(
                                [asdict(item) for item in results],
                                ensure_ascii=False,
                                default=str,
                            ),
                        )
                    ),
                    summarize_page=lambda step, point, query, search_result, text: (
                        pipeline.call_step(
                            page_summary_prompt,
                            step=step,
                            point=point,
                            query=query,
                            url=search_result.url,
                            page_text=text,
                        )
                    ),
                    summarize_step=lambda step, points: pipeline.call_step(
                        step_summary_prompt,
                        step=step,
                        point_results=_format_point_results(points),
                    ),
                    max_query_attempts=max_query_attempts,
                    on_status=status.task,
                )
                research_result = executor.run_file(plan_path, searches_dir)

            status.task(
                f"Collected research for {len(research_result.steps)} steps"
            )
            final_step = dict(research_config.get("final_step", {}))
            report_prompt = str(final_step.get("prompt", "final_report"))
            report_filename = str(final_step.get("output", "final_report.md"))
            status.task("Generating final report from all collected research")
            final_report = pipeline.call_step(
                report_prompt,
                user_topic=topic,
                research_plan=research_plan,
                research_data=_format_research_data(research_result),
            )
            final_report_path = run_dir / report_filename
            final_report_path.write_text(final_report.strip() + "\n", encoding="utf-8")
            status.task(f"Saved final report: {final_report_path}")

        _write_nerd_stats(run_dir, client.stats)
        _print_totals(status, client.stats)

    print(f"\nResearch directory: {run_dir}")
    print(f"Research plan:      {plan_path}")
    if final_report_path:
        print(f"Final report:       {final_report_path}")


def extract_research_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        match = re.match(r"^#(?!#)\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def slugify_research_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return slug[:180].rstrip("-") or "research"


def _write_planning_artifacts(
    planning_dir: Path,
    result: PipelineResult,
) -> None:
    write_debug(planning_dir / "01_topic_intro_raw.md", result.topic_intro.raw)
    write_debug(
        planning_dir / "01_topic_intro_extracted.txt",
        result.topic_intro.extracted or "",
    )
    write_debug(planning_dir / "02_questions_raw.md", result.questions.raw)
    write_debug(
        planning_dir / "02_questions_extracted.txt",
        result.questions.extracted or "",
    )
    if result.selected_questions:
        write_debug(
            planning_dir / "03_selected_questions_raw.md",
            result.selected_questions.raw,
        )
        write_debug(
            planning_dir / "03_selected_questions_extracted.txt",
            result.selected_questions.extracted or "",
        )
    write_debug(planning_dir / "04_final_plan.md", result.final_plan.extracted or "")


def write_debug(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _format_point_results(points: list[PointResearchResult]) -> str:
    sections: list[str] = []
    for point in points:
        sections.append(
            json.dumps(
                {
                    "point": point.point,
                    "query_attempts": point.query_attempts,
                    "pages": [
                        {
                            "title": page.title,
                            "url": page.url,
                            "summary": page.summary,
                            "error": page.error,
                        }
                        for page in point.pages
                    ],
                    "error": point.error,
                },
                ensure_ascii=False,
            )
        )
    return "\n\n".join(sections)


def _format_research_data(result: ResearchExecutionResult) -> str:
    sections: list[str] = []
    for step in result.steps:
        point_data = []
        for point in step.points:
            point_data.append(
                {
                    "point": point.point,
                    "queries": [
                        attempt["query"] for attempt in point.query_attempts
                    ],
                    "sources": [
                        {
                            "title": page.title,
                            "url": page.url,
                            "summary": page.summary,
                            "error": page.error,
                        }
                        for page in point.pages
                    ],
                    "error": point.error,
                }
            )
        sections.append(
            json.dumps(
                {
                    "step": step.title,
                    "step_summary": step.summary,
                    "points": point_data,
                    "error": step.error,
                },
                ensure_ascii=False,
            )
        )
    return "\n\n".join(sections)


def _write_nerd_stats(run_dir: Path, stats: list[LLMCallStats]) -> None:
    totals = _stats_totals(stats)
    payload = {
        "calls": [asdict(stat) for stat in stats],
        "totals": totals,
    }
    (run_dir / "nerd_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# Nerd stats",
        "",
        f"- Calls: {totals['calls']}",
        f"- Prompt tokens: {totals['prompt_tokens']:,}",
        f"- Completion tokens: {totals['completion_tokens']:,}",
        f"- Reasoning tokens: {totals['reasoning_tokens']:,}",
        f"- Total tokens: {totals['total_tokens']:,}",
        f"- Cost: ${totals['cost']:.6f}",
        f"- LLM response time: {totals['response_seconds']:.2f}s",
        f"- Completion throughput: {totals['tokens_per_second']:.1f} tok/s",
        "",
        "| Task | Model | Input | Output | Cost | Time | tok/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    rows.extend(
        (
            f"| {stat.operation} | {stat.model} | {stat.prompt_tokens:,} | "
            f"{stat.completion_tokens:,} | ${stat.cost:.6f} | "
            f"{stat.response_seconds:.2f}s | {stat.tokens_per_second:.1f} |"
        )
        for stat in stats
    )
    (run_dir / "nerd_stats.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _stats_totals(stats: list[LLMCallStats]) -> dict[str, int | float]:
    completion_tokens = sum(item.completion_tokens for item in stats)
    response_seconds = sum(item.response_seconds for item in stats)
    return {
        "calls": len(stats),
        "prompt_tokens": sum(item.prompt_tokens for item in stats),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": sum(item.reasoning_tokens for item in stats),
        "total_tokens": sum(item.total_tokens for item in stats),
        "cost": sum(item.cost for item in stats),
        "response_seconds": response_seconds,
        "tokens_per_second": (
            completion_tokens / response_seconds if response_seconds else 0.0
        ),
    }


def _print_totals(status: TerminalStatus, stats: list[LLMCallStats]) -> None:
    totals = _stats_totals(stats)
    status.task(
        f"Total nerd stats · {totals['total_tokens']:,} tokens · "
        f"${totals['cost']:.6f} · {totals['response_seconds']:.2f}s LLM time · "
        f"{totals['tokens_per_second']:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
