from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ModelConfig:
    reasoning: str
    general: str
    extractor: str


@dataclass
class PipelineConfig:
    models: ModelConfig
    prompt_file: str = "prompts.yaml"
    max_questions: int | None = None
    human_in_loop: bool = False
    stream: bool = False
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    site_url: str | None = None
    app_name: str = "research-pipeline"


@dataclass
class StepResult:
    raw: str
    extracted: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    topic_intro: StepResult
    questions: StepResult
    selected_questions: StepResult | None
    final_plan: StepResult


@dataclass
class SearchResult:
    title: str
    url: str
    content: str = ""
    engine: str | None = None
    score: float | None = None
    category: str | None = None
    published_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    suggestions: list[str] = field(default_factory=list)
    infoboxes: list[dict[str, Any]] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)