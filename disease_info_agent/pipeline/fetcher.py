"""Download web pages and save RawPage JSON under output/runs/{run_id}/raw_pages/."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import aiohttp
import trafilatura

from models.run_context import RunContext
from models.schemas import RawPage

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def serialize_raw_page(page: RawPage) -> str:
    payload = page.model_dump(mode="json") if hasattr(page, "model_dump") else page.dict()
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def html_to_text(html: str, url: str = "") -> str:
    """Extract main text with trafilatura; fall back to regex tag stripping."""
    text = trafilatura.extract(
        html,
        url=url or None,
        include_comments=False,
        include_tables=True,
    )
    if text and text.strip():
        return text.strip()

    cleaned = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class Fetcher:
    def __init__(
        self,
        run_context: RunContext | None = None,
        output_dir: Path | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.run_context = run_context
        if output_dir is not None:
            self.output_dir = output_dir
        elif run_context is not None:
            self.output_dir = run_context.raw_pages_dir
        else:
            self.output_dir = None
        self.timeout_seconds = timeout_seconds

    def save_page(self, page: RawPage, index: int) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir 未设置，无法保存 RawPage")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{index:03d}.json"
        path.write_text(serialize_raw_page(page), encoding="utf-8")
        return path

    async def _fetch_html_aiohttp(self, url: str) -> tuple[str, int, str]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=DEFAULT_HEADERS,
            max_line_size=65536,
            max_field_size=65536,
        ) as session:
            async with session.get(url, allow_redirects=True) as response:
                html = await response.text(errors="ignore")
                return html, response.status, "aiohttp"

    async def _fetch_html_requests(self, url: str) -> tuple[str, int, str]:
        import requests

        def fetch() -> tuple[str, int]:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text, response.status_code

        html, status_code = await asyncio.to_thread(fetch)
        return html, status_code, "requests"

    async def fetch_one(self, url: str) -> RawPage:
        fetched_at = datetime.now(timezone.utc)
        html = ""
        status_code: int | None = None
        fetch_method = "failed"
        error: str | None = None

        try:
            html, status_code, fetch_method = await self._fetch_html_aiohttp(url)
        except Exception as aiohttp_error:
            try:
                html, status_code, fetch_method = await self._fetch_html_requests(url)
            except Exception as requests_error:
                return RawPage(
                    url=url,
                    text="",
                    html=None,
                    status_code=status_code,
                    fetch_method="failed",
                    fetched_at=fetched_at,
                    error=f"aiohttp: {aiohttp_error}; requests: {requests_error}",
                )

        text = html_to_text(html, url=url)
        if not text:
            error = "页面下载成功，但未能提取正文"

        return RawPage(
            url=url,
            text=text,
            html=html[:50000] if html else None,
            status_code=status_code,
            fetch_method=fetch_method,
            fetched_at=fetched_at,
            error=error,
        )

    async def fetch_all(self, urls: list[str], concurrency: int = 5) -> list[RawPage]:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RawPage | None] = [None] * len(urls)

        async def fetch_index(index: int, url: str) -> None:
            async with semaphore:
                page = await self.fetch_one(url)
                results[index] = page
                if self.output_dir is not None:
                    self.save_page(page, index + 1)
                status = page.status_code if page.status_code is not None else "?"
                text_len = len(page.text)
                print(f"  [{index + 1}/{len(urls)}] {url} -> {page.fetch_method} {status}, {text_len} chars")

        await asyncio.gather(*(fetch_index(i, url) for i, url in enumerate(urls)))
        return [page for page in results if page is not None]
