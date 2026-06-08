from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import httpx

from .models import ChatMessage, PipelineConfig

class OpenRouterClient:
    def __init__(self, api_key: str, config: PipelineConfig):
        self.api_key = api_key
        self.config = config
        self.base_url = config.openrouter_base_url.rstrip("/")
        self.client = httpx.Client(timeout=300)

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
    ) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
            "stream": stream,
        }

    def chat(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage],
        temperature: float = 0.2,
    ) -> str:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            ),
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage],
        temperature: float = 0.2,
    ) -> Iterator[str]:
        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
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

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()