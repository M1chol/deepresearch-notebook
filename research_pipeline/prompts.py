from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import ChatMessage, ModelConfig


@dataclass
class PromptStep:
    name: str
    model_role: str
    temperature: float
    system: str
    user: str
    reasoning: dict[str, Any] | None


@dataclass
class WorkflowStep:
    name: str
    prompt: str
    variables: dict[str, str]
    extract: list[str]
    condition: str | None
    fallback: dict[str, Any] | None
    human_edit_title: str | None
    metadata: dict[str, str]


class PromptRegistry:
    def __init__(self, prompt_file: str):
        self.path = Path(prompt_file)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get_step(self, name: str) -> PromptStep:
        try:
            raw = self.data["steps"][name]
        except KeyError as exc:
            raise KeyError(f"Prompt step not found: {name}") from exc

        return PromptStep(
            name=name,
            model_role=raw["model_role"],
            temperature=float(raw.get("temperature", 0.2)),
            system=raw.get("system") or self.data.get("system", {}).get("default", ""),
            user=raw["user"],
            reasoning=raw.get("reasoning"),
        )

    def get_research_config(self) -> dict[str, Any]:
        return dict(self.data.get("research", {}))

    def get_workflow(self) -> list[WorkflowStep]:
        try:
            raw_steps = self.data["pipeline"]["steps"]
        except KeyError as exc:
            raise KeyError("Pipeline workflow not found in prompt YAML") from exc

        return [
            WorkflowStep(
                name=raw["name"],
                prompt=raw["prompt"],
                variables=dict(raw.get("variables", {})),
                extract=list(raw.get("extract", [])),
                condition=raw.get("condition"),
                fallback=raw.get("fallback"),
                human_edit_title=raw.get("human_edit_title"),
                metadata=dict(raw.get("metadata", {})),
            )
            for raw in raw_steps
        ]

    def render_messages(self, step_name: str, **variables: str) -> list[ChatMessage]:
        step = self.get_step(step_name)

        try:
            user_content = step.user.format(**variables)
            system_content = step.system.format(**variables)
        except KeyError as exc:
            missing = str(exc)
            raise KeyError(
                f"Missing variable {missing} for prompt step {step_name}"
            ) from exc

        return [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]

    @staticmethod
    def resolve_model(model_role: str, models: ModelConfig) -> str:
        try:
            return getattr(models, model_role)
        except AttributeError as exc:
            raise ValueError(f"Unknown model role: {model_role}") from exc
