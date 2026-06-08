from __future__ import annotations

import argparse
from pathlib import Path

from .config import default_config, load_api_key
from .openrouter_client import OpenRouterClient
from .pipeline import ResearchPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research planning pipeline")
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
        "--output",
        default="research_plan.md",
        help="Where to save the final research plan.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming for AI response",
    )

    args = parser.parse_args()

    if args.topic_file:
        topic = Path(args.topic_file).read_text(encoding="utf-8")
    elif args.topic:
        topic = args.topic
    else:
        raise SystemExit("Provide either a topic argument or --topic-file")

    api_key = load_api_key()
    config = default_config(
        prompt_file=args.prompts,
        max_questions=args.max_questions,
        human_in_loop=args.human_in_loop,
        stream=args.stream,
    )

    with OpenRouterClient(api_key=api_key, config=config) as client:
        pipeline = ResearchPipeline(client=client, config=config)
        result = pipeline.run(topic)

    output_path = Path(args.output)
    output_path.write_text(result.final_plan.extracted or "", encoding="utf-8")

    print(f"Saved final research plan to: {output_path}")

    debug_dir = output_path.with_suffix("")
    debug_dir.mkdir(exist_ok=True)

    write_debug(debug_dir / "01_topic_intro_raw.md", result.topic_intro.raw)
    write_debug(
        debug_dir / "01_topic_intro_extracted.txt",
        result.topic_intro.extracted or "",
    )
    write_debug(debug_dir / "02_questions_raw.md", result.questions.raw)
    write_debug(
        debug_dir / "02_questions_extracted.txt",
        result.questions.extracted or "",
    )

    if result.selected_questions:
        write_debug(
            debug_dir / "03_selected_questions_raw.md",
            result.selected_questions.raw,
        )
        write_debug(
            debug_dir / "03_selected_questions_extracted.txt",
            result.selected_questions.extracted or "",
        )

    write_debug(debug_dir / "04_final_plan.md", result.final_plan.extracted or "")


def write_debug(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()