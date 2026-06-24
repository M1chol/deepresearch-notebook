from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import PipelineConfig, PipelineResult, StepResult
from .openrouter_client import OpenRouterClient
from .parsing import extract_last_fenced_block, normalize_numbered_lines
from .prompts import PromptRegistry, WorkflowStep

ChunkCallback = Callable[[str, str], None]


class ResearchPipeline:
    """Generic interpreter for the workflow declared in prompts.yaml."""

    def __init__(self, client: OpenRouterClient, config: PipelineConfig):
        self.client = client
        self.config = config
        self.prompts = PromptRegistry(config.prompt_file)
        self.on_status: Callable[[str], None] | None = None

    def run(self, user_topic: str) -> PipelineResult:
        context: dict[str, Any] = {
            "user_topic": user_topic,
            "max_questions": self.config.max_questions,
        }
        results: dict[str, StepResult] = {}

        for definition in self.prompts.get_workflow():
            if not self._condition_matches(definition.condition, context):
                continue

            result = self._run_workflow_step(definition, context)
            results[definition.name] = result
            context[definition.name] = result.extracted or ""
            context[f"{definition.name}_raw"] = result.raw

        try:
            return PipelineResult(
                topic_intro=results["topic_intro"],
                questions=results["questions"],
                selected_questions=results.get("selected_questions"),
                final_plan=results["final_plan"],
                steps=results,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Pipeline YAML did not produce required output: {exc.args[0]}"
            ) from exc

    def call_step(
        self,
        step_name: str,
        on_chunk: ChunkCallback | None = None,
        **variables: str,
    ) -> str:
        step = self.prompts.get_step(step_name)
        if self.on_status:
            self.on_status(f"LLM: {step_name}")
        model = self.prompts.resolve_model(step.model_role, self.config.models)
        messages = self.prompts.render_messages(step_name, **variables)

        if self.config.stream:
            chunks: list[str] = []
            for chunk in self.client.chat_stream(
                model=model,
                messages=messages,
                temperature=step.temperature,
                reasoning=step.reasoning,
                operation=step_name,
            ):
                chunks.append(chunk)
                if on_chunk:
                    on_chunk(step_name, chunk)
            return "".join(chunks)

        return self.client.chat(
            model=model,
            messages=messages,
            temperature=step.temperature,
            reasoning=step.reasoning,
            operation=step_name,
        )

    def _run_workflow_step(
        self,
        definition: WorkflowStep,
        context: dict[str, Any],
    ) -> StepResult:
        variables = self._resolve_mapping(definition.variables, context)
        raw = self.call_step(
            definition.prompt,
            on_chunk=self.print_chunk,
            **variables,
        )
        extracted = self._apply_extractors(raw, definition.extract)
        metadata = self._resolve_mapping(definition.metadata, context)
        metadata["prompt"] = definition.prompt
        metadata["fallback_used"] = False

        if self._is_empty(extracted) and definition.fallback:
            fallback_prompt = definition.fallback["prompt"]
            fallback_context = {**context, "raw": raw, "extracted": extracted or ""}
            fallback_variables = self._resolve_mapping(
                definition.fallback.get("variables", {}),
                fallback_context,
            )
            fallback_raw = self.call_step(fallback_prompt, **fallback_variables)
            extracted = self._apply_extractors(
                fallback_raw,
                list(definition.fallback.get("extract", [])),
            )
            metadata.update(
                fallback_used=True,
                fallback_prompt=fallback_prompt,
                fallback_raw=fallback_raw,
            )

        if definition.human_edit_title and self.config.human_in_loop:
            extracted = self.ask_human_edit(
                title=definition.human_edit_title,
                content=extracted or "",
            )

        return StepResult(raw=raw, extracted=extracted, metadata=metadata)

    @staticmethod
    def _resolve_mapping(
        mapping: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        string_context = {
            key: "" if value is None else str(value)
            for key, value in context.items()
        }
        for key, value in mapping.items():
            template = str(value)
            if template.startswith("{") and template.endswith("}") and "|" in template:
                candidates = template[1:-1].split("|")
                resolved[key] = next(
                    (
                        string_context[candidate]
                        for candidate in candidates
                        if string_context.get(candidate)
                    ),
                    "",
                )
            else:
                resolved[key] = template.format_map(string_context)
        return resolved

    @staticmethod
    def _condition_matches(condition: str | None, context: dict[str, Any]) -> bool:
        if not condition:
            return True
        if condition.startswith("present:"):
            return context.get(condition.removeprefix("present:")) is not None
        if condition.startswith("missing:"):
            return context.get(condition.removeprefix("missing:")) is None
        raise ValueError(f"Unsupported pipeline condition: {condition}")

    @staticmethod
    def _apply_extractors(text: str, extractors: list[str]) -> str | None:
        value: str | None = text
        for extractor in extractors:
            if extractor == "fenced_block":
                value = extract_last_fenced_block(value or "")
            elif extractor == "strip":
                value = (value or "").strip()
            elif extractor == "numbered_lines":
                value = normalize_numbered_lines(value or "")
            else:
                raise ValueError(f"Unsupported pipeline extractor: {extractor}")
        return value

    @staticmethod
    def _is_empty(value: str | None) -> bool:
        return value is None or not value.strip()

    @staticmethod
    def ask_human_edit(title: str, content: str) -> str:
        print()
        print("=" * 80)
        print(title)
        print("=" * 80)
        print(content)
        print()
        print("Press Enter to accept, or type/paste replacement text.")
        print("Finish replacement with a single line containing only: END")
        first = input("> ")

        if first.strip() == "":
            print("Continuing without an edit...")
            return content

        lines = [first]
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        print("Edit finished...")
        return "\n".join(lines).strip()

    @staticmethod
    def print_chunk(step_name: str, chunk: str) -> None:
        print(chunk, end="", flush=True)
