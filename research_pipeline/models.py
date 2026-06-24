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
    cheap: str
    mid: str


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
class LLMCallStats:
    operation: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    response_seconds: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class PipelineResult:
    topic_intro: StepResult
    questions: StepResult
    selected_questions: StepResult | None
    final_plan: StepResult
    steps: dict[str, StepResult] = field(default_factory=dict)


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
