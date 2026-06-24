from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

import httpx

from .models import SearchResponse, SearchResult


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text while excluding non-content HTML elements."""

    _ignored_tags = {"script", "style", "noscript", "template", "svg"}
    _block_tags = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and tag in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif self._ignored_depth == 0 and tag in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        lines = []
        for line in "".join(self._parts).splitlines():
            normalized = " ".join(line.split())
            if normalized:
                lines.append(normalized)
        return "\n".join(lines)


def extract_html_text(html: str) -> str:
    """Convert HTML into normalized readable text without optional dependencies."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


class HTTPPageScraper:
    """Fetch complete pages and return normalized text."""

    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str = "research-pipeline/0.1",
        max_response_bytes: int = 10_000_000,
        client: httpx.Client | None = None,
    ) -> None:
        self.max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        )

    def scrape(self, url: str) -> str:
        if not url.strip():
            raise ValueError("Page URL must not be empty")

        with self.client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > self.max_response_bytes:
                    raise ValueError(
                        f"Page exceeds {self.max_response_bytes} byte scrape limit"
                    )
                chunks.append(chunk)

            body = b"".join(chunks)
            encoding = response.encoding or "utf-8"
            text = body.decode(encoding, errors="replace")
            content_type = response.headers.get("content-type", "").lower()

        if "html" in content_type or "<html" in text[:1000].lower():
            return extract_html_text(text)
        return text.strip()

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "HTTPPageScraper":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


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

    def _parse_response(self, query: str, data: dict[str, Any]) -> SearchResponse:
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
