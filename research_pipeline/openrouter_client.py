from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import httpx

from .models import ChatMessage, LLMCallStats, PipelineConfig

class OpenRouterClient:
    def __init__(self, api_key: str, config: PipelineConfig):
        self.api_key = api_key
        self.config = config
        self.base_url = config.openrouter_base_url.rstrip("/")
        self.client = httpx.Client(timeout=300)
        self.stats: list[LLMCallStats] = []
        self.on_stats: Callable[[LLMCallStats], None] | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url

        if self.config.app_name:
            headers["X-Title"] = self.config.app_name

        return headers

    def _payload(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage],
        temperature: float,
        stream: bool,
        reasoning: dict[str, Any] | None = None,
    ) -> dict:
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
            "stream": stream,
        }
        if reasoning is not None:
            payload["reasoning"] = reasoning
        return payload

    def chat(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage],
        temperature: float = 0.2,
        reasoning: dict[str, Any] | None = None,
        operation: str = "chat",
    ) -> str:
        started = time.perf_counter()
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                reasoning=reasoning,
            ),
        )
        response.raise_for_status()
        data = response.json()
        self._record_stats(
            operation=operation,
            model=data.get("model") or model,
            usage=data.get("usage") or {},
            response_seconds=time.perf_counter() - started,
        )
        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage],
        temperature: float = 0.2,
        reasoning: dict[str, Any] | None = None,
        operation: str = "chat_stream",
    ) -> Iterator[str]:
        started = time.perf_counter()
        usage: dict[str, Any] = {}
        response_model = model
        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                reasoning=reasoning,
            ),
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if data.get("usage"):
                    usage = data["usage"]
                response_model = data.get("model") or response_model
                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        self._record_stats(
            operation=operation,
            model=response_model,
            usage=usage,
            response_seconds=time.perf_counter() - started,
        )

    def _record_stats(
        self,
        *,
        operation: str,
        model: str,
        usage: dict[str, Any],
        response_seconds: float,
    ) -> None:
        completion_tokens = int(usage.get("completion_tokens") or 0)
        reasoning_tokens = int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            or 0
        )
        stat = LLMCallStats(
            operation=operation,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=int(usage.get("total_tokens") or 0),
            cost=float(usage.get("cost") or 0.0),
            response_seconds=response_seconds,
            tokens_per_second=(
                completion_tokens / response_seconds
                if completion_tokens and response_seconds
                else 0.0
            ),
        )
        self.stats.append(stat)
        if self.on_stats:
            self.on_stats(stat)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
