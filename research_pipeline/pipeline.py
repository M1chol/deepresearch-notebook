from __future__ import annotations

from .openrouter_client import OpenRouterClient
from .parsing import extract_last_fenced_block, normalize_numbered_lines
from .prompts import PromptRegistry
from .models import PipelineConfig, PipelineResult, StepResult

from collections.abc import Callable

ChunkCallback = Callable[[str, str], None]

class ResearchPipeline:
    def __init__(self, client: OpenRouterClient, config: PipelineConfig):
        self.client = client
        self.config = config
        self.prompts = PromptRegistry(config.prompt_file)

    def run(self, user_topic: str) -> PipelineResult:
        topic_intro = self.generate_topic_intro(user_topic)

        if self.config.human_in_loop:
            topic_intro.extracted = self.ask_human_edit(
                title="Generated topic introduction",
                content=topic_intro.extracted or "",
            )

        questions = self.generate_questions(
            user_topic=user_topic,
            topic_intro=topic_intro.extracted or "",
        )

        if self.config.human_in_loop:
            questions.extracted = self.ask_human_edit(
                title="Generated research questions",
                content=questions.extracted or "",
            )

        selected_questions = None
        final_questions = questions.extracted or ""

        if self.config.max_questions is not None:
            selected_questions = self.select_questions(
                questions=final_questions,
                max_questions=self.config.max_questions,
            )
            final_questions = selected_questions.extracted or ""

            if self.config.human_in_loop:
                final_questions = self.ask_human_edit(
                    title="Selected research questions",
                    content=final_questions,
                )
                selected_questions.extracted = final_questions

        final_plan = self.generate_final_plan(
            user_topic=user_topic,
            topic_intro=topic_intro.extracted or "",
            questions=final_questions,
        )

        return PipelineResult(
            topic_intro=topic_intro,
            questions=questions,
            selected_questions=selected_questions,
            final_plan=final_plan,
        )

    def call_step(
        self,
        step_name: str,
        on_chunk: ChunkCallback | None = None,
        **variables: str,
    ) -> str:
        step = self.prompts.get_step(step_name)
        model = self.prompts.resolve_model(step.model_role, self.config.models)
        messages = self.prompts.render_messages(step_name, **variables)

        if self.config.stream:
            chunks: list[str] = []
            for chunk in self.client.chat_stream(
                model=model,
                messages=messages,
                temperature=step.temperature,
            ):
                chunks.append(chunk)
                if on_chunk:
                    on_chunk(step_name, chunk)
            return "".join(chunks)

        return self.client.chat(
            model=model,
            messages=messages,
            temperature=step.temperature,
        )

    def generate_topic_intro(self, user_topic: str) -> StepResult:
        raw = self.call_step("topic_summary", user_topic=user_topic, on_chunk=self.print_chunk)
        extracted = extract_last_fenced_block(raw)

        if not extracted:
            fallback_raw = self.call_step(
                "extract_wikipedia_intro",
                source_text=raw,
            )
            extracted = extract_last_fenced_block(fallback_raw) or fallback_raw.strip()

            return StepResult(
                raw=raw,
                extracted=extracted,
                metadata={"fallback_used": True, "fallback_raw": fallback_raw},
            )

        return StepResult(
            raw=raw,
            extracted=extracted,
            metadata={"fallback_used": False},
        )

    def generate_questions(self, user_topic: str, topic_intro: str) -> StepResult:
        raw = self.call_step(
            "research_questions",
            user_topic=user_topic,
            topic_intro=topic_intro,
            on_chunk=self.print_chunk
        )
        extracted = extract_last_fenced_block(raw)

        if not extracted:
            fallback_raw = self.call_step(
                "extract_questions",
                source_text=raw,
            )
            extracted = extract_last_fenced_block(fallback_raw) or fallback_raw.strip()

            return StepResult(
                raw=raw,
                extracted=normalize_numbered_lines(extracted),
                metadata={"fallback_used": True, "fallback_raw": fallback_raw},
            )

        return StepResult(
            raw=raw,
            extracted=normalize_numbered_lines(extracted),
            metadata={"fallback_used": False},
        )

    def select_questions(self, questions: str, max_questions: int) -> StepResult:
        raw = self.call_step(
            "select_questions",
            questions=questions,
            max_questions=str(max_questions),
            on_chunk=self.print_chunk
        )
        extracted = extract_last_fenced_block(raw) or raw.strip()

        return StepResult(
            raw=raw,
            extracted=normalize_numbered_lines(extracted),
            metadata={"max_questions": max_questions},
        )

    def generate_final_plan(
        self,
        *,
        user_topic: str,
        topic_intro: str,
        questions: str,
    ) -> StepResult:
        raw = self.call_step(
            "final_plan",
            user_topic=user_topic,
            topic_intro=topic_intro,
            questions=questions,
            on_chunk=self.print_chunk
        )

        return StepResult(raw=raw, extracted=raw)

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