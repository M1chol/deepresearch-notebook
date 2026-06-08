from __future__ import annotations

import httpx

from .models import SearchResponse, SearchResult


class SearxNGClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/search",
        timeout: float = 30.0,
        user_agent: str = "research-pipeline/0.1",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )

    def search(
        self,
        query: str,
        *,
        categories: str | None = None,
        language: str | None = None,
        page: int | None = None,
        time_range: str | None = None,
        safesearch: int | None = None,
        format: str = "json",
    ) -> SearchResponse:
        if not query.strip():
            raise ValueError("Search query must not be empty")

        params: dict[str, str | int] = {
            "q": query,
            "format": format,
        }

        if categories:
            params["categories"] = categories
        if language:
            params["language"] = language
        if page is not None:
            params["pageno"] = page
        if time_range:
            params["time_range"] = time_range
        if safesearch is not None:
            params["safesearch"] = safesearch
        
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()

        data = response.json()
        return self._parse_response(query=query, data=data)

    def search_many(
        self,
        queries: list[str],
        *,
        categories: str | None = None,
        language: str | None = None,
        page: int | None = None,
        time_range: str | None = None,
        safesearch: int | None = None,
        format: str = "json",
    ) -> list[SearchResponse]:
        return [
            self.search(
                query,
                categories=categories,
                language=language,
                page=page,
                time_range=time_range,
                safesearch=safesearch,
                format=format,
            )
            for query in queries
        ]

    def _parse_response(self, query: str, data: dict) -> SearchResponse:
        results = []

        for item in data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", "").strip(),
                    url=item.get("url", "").strip(),
                    content=item.get("content", "").strip(),
                    engine=item.get("engine"),
                    score=self._safe_float(item.get("score")),
                    category=item.get("category"),
                    published_date=item.get("publishedDate"),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k
                        not in {
                            "title",
                            "url",
                            "content",
                            "engine",
                            "score",
                            "category",
                            "publishedDate",
                        }
                    },
                )
            )

        return SearchResponse(
            query=query,
            results=results,
            suggestions=data.get("suggestions", []),
            infoboxes=data.get("infoboxes", []),
            answers=data.get("answers", []),
            raw=data,
        )

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "SearxNGClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()