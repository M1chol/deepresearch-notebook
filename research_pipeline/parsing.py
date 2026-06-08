from __future__ import annotations

import re


FENCED_BLOCK_RE = re.compile(
    r"```(?:text|markdown|md)?\s*\n(?P<body>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def extract_last_fenced_block(text: str) -> str | None:
    matches = list(FENCED_BLOCK_RE.finditer(text.strip()))
    if not matches:
        return None

    return matches[-1].group("body").strip()


def normalize_numbered_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    cleaned = []

    for line in lines:
        if not line.strip():
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def ensure_fenced_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped
    return f"```text\n{stripped}\n```"