from __future__ import annotations

import os

from dotenv import load_dotenv

from .models import ModelConfig, PipelineConfig


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env or environment")
    return api_key


def default_config(
    *,
    prompt_file: str = "prompts.yaml",
    max_questions: int | None = None,
    human_in_loop: bool = False,
    stream: bool = False
) -> PipelineConfig:
    return PipelineConfig(
        models=ModelConfig(
            reasoning="moonshotai/kimi-k2.6",
            general="moonshotai/kimi-k2.6",
            extractor="google/gemma-4-26b-a4b-it",
            cheap="google/gemma-4-26b-a4b-it",
            mid="google/gemma-4-26b-a4b-it",
        ),
        prompt_file=prompt_file,
        max_questions=max_questions,
        human_in_loop=human_in_loop,
        stream=stream
    )
