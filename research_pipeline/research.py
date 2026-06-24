from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import SearchResponse, SearchResult


class SearchCallable(Protocol):
    def __call__(self, query: str) -> SearchResponse: ...


class ScrapeCallable(Protocol):
    def __call__(self, url: str) -> str: ...


QueryBuilder = Callable[[str, str], str]
QueryBroadener = Callable[[str, str, str, str], str]
ResultSelector = Callable[
    [str, str, str, Sequence[SearchResult]],
    Sequence[int] | Sequence[str] | dict[str, Any] | str,
]
PageSummarizer = Callable[[str, str, str, SearchResult, str], str]
StepSummarizer = Callable[[str, Sequence["PointResearchResult"]], str]


@dataclass(frozen=True)
class ResearchPlanStep:
    title: str
    points: list[str]


@dataclass
class PageResearchResult:
    result_index: int
    title: str
    url: str
    summary: str | None = None
    artifact_path: Path | None = None
    error: str | None = None


@dataclass
class PointResearchResult:
    point: str
    query: str
    query_attempts: list[dict[str, Any]] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    selection: Sequence[int] | Sequence[str] | dict[str, Any] | str | None = None
    selected_indices: list[int] = field(default_factory=list)
    pages: list[PageResearchResult] = field(default_factory=list)
    artifact_dir: Path | None = None
    error: str | None = None


@dataclass
class StepResearchResult:
    title: str
    points: list[PointResearchResult]
    summary: str
    artifact_dir: Path
    error: str | None = None


@dataclass
class ResearchExecutionResult:
    plan_path: Path | None
    output_dir: Path
    steps: list[StepResearchResult]


_STEP_RE = re.compile(r"^##(?!#)\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-+*]\s+(.+?)\s*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def parse_research_plan(markdown: str) -> list[ResearchPlanStep]:
    """Parse level-two headings and the bullet points belonging to each heading."""
    steps: list[ResearchPlanStep] = []
    current_title: str | None = None
    current_points: list[str] = []

    for line in markdown.splitlines():
        heading = _STEP_RE.match(line)
        if heading:
            if current_title is not None:
                steps.append(
                    ResearchPlanStep(title=current_title, points=current_points)
                )
            current_title = heading.group(1).strip()
            current_points = []
            continue

        if current_title is not None:
            bullet = _BULLET_RE.match(line)
            if bullet:
                current_points.append(bullet.group(1).strip())

    if current_title is not None:
        steps.append(ResearchPlanStep(title=current_title, points=current_points))

    return steps


def safe_artifact_name(value: str, *, max_slug_length: int = 48) -> str:
    normalized = _SLUG_RE.sub("-", value.lower()).strip("-") or "item"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:max_slug_length].rstrip('-')}-{digest}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _search_result_dict(result: SearchResult) -> dict[str, Any]:
    return asdict(result)


def _parse_selector_response(
    selection: Sequence[int] | Sequence[str] | dict[str, Any] | str,
    results: Sequence[SearchResult],
) -> list[int]:
    raw: Any = selection
    if isinstance(selection, str):
        try:
            raw = json.loads(selection)
        except json.JSONDecodeError:
            raw = [
                line.strip(" -*\t")
                for line in selection.splitlines()
                if line.strip(" -*\t")
            ]

    if isinstance(raw, dict):
        raw = (
            raw.get("selected_indices")
            or raw.get("indices")
            or raw.get("selected_urls")
            or raw.get("urls")
            or []
        )

    indices: list[int] = []
    urls = {result.url: index for index, result in enumerate(results)}
    for item in raw:
        index: int | None = None
        if isinstance(item, int):
            index = item
        elif isinstance(item, str):
            stripped = item.strip()
            if stripped.isdigit():
                index = int(stripped)
            else:
                index = urls.get(stripped)
        if index is not None and 0 <= index < len(results) and index not in indices:
            indices.append(index)
    return indices


class ResearchExecutor:
    """Execute each point of a Markdown research plan and persist all artifacts."""

    def __init__(
        self,
        *,
        search: SearchCallable,
        scrape: ScrapeCallable,
        build_query: QueryBuilder,
        broaden_query: QueryBroadener | None = None,
        select_results: ResultSelector,
        summarize_page: PageSummarizer,
        summarize_step: StepSummarizer,
        max_query_attempts: int = 3,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.search = search
        self.scrape = scrape
        self.build_query = build_query
        self.broaden_query = broaden_query
        self.select_results = select_results
        self.summarize_page = summarize_page
        self.summarize_step = summarize_step
        self.max_query_attempts = max(1, max_query_attempts)
        self.on_status = on_status

    def run_file(
        self,
        plan_path: str | Path,
        searches_dir: str | Path,
    ) -> ResearchExecutionResult:
        path = Path(plan_path)
        return self.run_markdown(
            path.read_text(encoding="utf-8"),
            searches_dir,
            plan_path=path,
        )

    def run_markdown(
        self,
        markdown: str,
        searches_dir: str | Path,
        *,
        plan_path: str | Path | None = None,
    ) -> ResearchExecutionResult:
        output_dir = Path(searches_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        steps = parse_research_plan(markdown)

        _write_json(
            output_dir / "manifest.json",
            {
                "plan_path": str(plan_path) if plan_path is not None else None,
                "steps": [asdict(step) for step in steps],
            },
        )

        step_results = [
            self._run_step(step_index, step, output_dir)
            for step_index, step in enumerate(steps, start=1)
        ]
        result = ResearchExecutionResult(
            plan_path=Path(plan_path) if plan_path is not None else None,
            output_dir=output_dir,
            steps=step_results,
        )
        _write_json(output_dir / "result.json", asdict(result))
        return result

    def _run_step(
        self,
        step_index: int,
        step: ResearchPlanStep,
        output_dir: Path,
    ) -> StepResearchResult:
        step_dir = output_dir / (
            f"{step_index:02d}-{safe_artifact_name(step.title)}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        self._status(f"Researching step {step_index}: {step.title}")
        point_results = [
            self._run_point(step, point_index, point, step_dir)
            for point_index, point in enumerate(step.points, start=1)
        ]

        summary_error = None
        try:
            self._status(f"Synthesizing step: {step.title}")
            summary = self.summarize_step(step.title, point_results).strip()
        except Exception as exc:
            summary_error = f"{type(exc).__name__}: {exc}"
            summary = ""

        summary_path = step_dir / "step-summary.md"
        summary_path.write_text(
            f"# {step.title}\n\n{summary or '_Step summary unavailable._'}\n",
            encoding="utf-8",
        )
        step_result = StepResearchResult(
            title=step.title,
            points=point_results,
            summary=summary,
            artifact_dir=step_dir,
            error=summary_error,
        )
        _write_json(step_dir / "step.json", asdict(step_result))
        return step_result

    def _run_point(
        self,
        step: ResearchPlanStep,
        point_index: int,
        point: str,
        step_dir: Path,
    ) -> PointResearchResult:
        point_dir = step_dir / (
            f"{point_index:02d}-{safe_artifact_name(point)}"
        )
        point_dir.mkdir(parents=True, exist_ok=True)
        point_result = PointResearchResult(
            point=point,
            query="",
            artifact_dir=point_dir,
        )

        try:
            self._status(f"Building query: {point}")
            point_result.query = self.build_query(step.title, point).strip()
            if not point_result.query:
                raise ValueError("Query builder returned an empty query")

            retry_reason = ""
            for attempt in range(1, self.max_query_attempts + 1):
                if attempt > 1:
                    if not self.broaden_query:
                        break
                    self._status(
                        f"Broadening query ({attempt}/{self.max_query_attempts}): "
                        f"{point}"
                    )
                    point_result.query = self.broaden_query(
                        step.title,
                        point,
                        point_result.query,
                        retry_reason,
                    ).strip()
                    if not point_result.query:
                        raise ValueError("Query broadener returned an empty query")

                self._status(
                    f"Searching ({attempt}/{self.max_query_attempts}): "
                    f"{point_result.query}"
                )
                response = self.search(point_result.query)
                point_result.search_results = list(response.results)
                point_result.selection = None
                point_result.selected_indices = []

                if point_result.search_results:
                    self._status(
                        f"Selecting useful results: {len(point_result.search_results)} "
                        f"candidates"
                    )
                    point_result.selection = self.select_results(
                        step.title,
                        point,
                        point_result.query,
                        point_result.search_results,
                    )
                    point_result.selected_indices = _parse_selector_response(
                        point_result.selection,
                        point_result.search_results,
                    )

                retry_reason = (
                    "search returned zero results"
                    if not point_result.search_results
                    else "relevance selection returned zero useful results"
                )
                point_result.query_attempts.append(
                    {
                        "attempt": attempt,
                        "query": point_result.query,
                        "result_count": len(point_result.search_results),
                        "results": [
                            _search_result_dict(result)
                            for result in point_result.search_results
                        ],
                        "selected_count": len(point_result.selected_indices),
                        "selection": point_result.selection,
                        "retry_reason": (
                            None if point_result.selected_indices else retry_reason
                        ),
                    }
                )
                if point_result.selected_indices:
                    break
        except Exception as exc:
            point_result.error = f"{type(exc).__name__}: {exc}"
            self._write_point_artifacts(point_result)
            return point_result

        for result_index in point_result.selected_indices:
            search_result = point_result.search_results[result_index]
            page = PageResearchResult(
                result_index=result_index,
                title=search_result.title,
                url=search_result.url,
            )
            page_number = len(point_result.pages) + 1
            page_path = point_dir / (
                f"page-{page_number:02d}-{safe_artifact_name(search_result.title)}.md"
            )
            page.artifact_path = page_path
            try:
                self._status(f"Scraping: {search_result.url}")
                scraped_text = self.scrape(search_result.url)
            except Exception as exc:
                page.error = f"{type(exc).__name__}: {exc}"
                page_path.write_text(
                    self._page_error_markdown(
                        point,
                        point_result.query,
                        search_result,
                        page.error,
                    ),
                    encoding="utf-8",
                )
                point_result.pages.append(page)
                continue

            try:
                self._status(f"Summarizing page: {search_result.title}")
                page.summary = self.summarize_page(
                    step.title,
                    point,
                    point_result.query,
                    search_result,
                    scraped_text,
                ).strip()
            except Exception as exc:
                page.error = f"{type(exc).__name__}: {exc}"
            page_path.write_text(
                self._page_markdown(
                    step.title,
                    point,
                    point_result.query,
                    search_result,
                    scraped_text,
                    page.summary,
                    page.error,
                ),
                encoding="utf-8",
            )
            point_result.pages.append(page)

        self._write_point_artifacts(point_result)
        return point_result

    @staticmethod
    def _page_markdown(
        step_title: str,
        point: str,
        query: str,
        result: SearchResult,
        scraped_text: str,
        summary: str | None,
        summary_error: str | None,
    ) -> str:
        summary_content = summary or "_No summary returned._"
        if summary_error:
            summary_content += f"\n\nSummary error: {summary_error}"
        return (
            f"# {result.title or result.url}\n\n"
            f"- Step: {step_title}\n"
            f"- Point: {point}\n"
            f"- Query: {query}\n"
            f"- URL: {result.url}\n\n"
            "## Cheap-model summary\n\n"
            f"{summary_content}\n\n"
            "## Extracted page text\n\n"
            f"{scraped_text}\n"
        )

    @staticmethod
    def _page_error_markdown(
        point: str,
        query: str,
        result: SearchResult,
        error: str,
    ) -> str:
        return (
            f"# {result.title or result.url}\n\n"
            f"- Point: {point}\n"
            f"- Query: {query}\n"
            f"- URL: {result.url}\n"
            f"- Error: {error}\n"
        )

    @staticmethod
    def _write_point_artifacts(point: PointResearchResult) -> None:
        assert point.artifact_dir is not None
        _write_json(
            point.artifact_dir / "search.json",
            {
                "point": point.point,
                "query": point.query,
                "query_attempts": point.query_attempts,
                "results": [
                    _search_result_dict(result) for result in point.search_results
                ],
                "selection": point.selection,
                "selected_indices": point.selected_indices,
                "selected_urls": [
                    point.search_results[index].url
                    for index in point.selected_indices
                ],
                "error": point.error,
            },
        )
        _write_json(point.artifact_dir / "point.json", asdict(point))

    def _status(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)
