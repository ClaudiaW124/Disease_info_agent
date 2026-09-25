"""Phase 4 — search-driven incremental knowledge base expansion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.run_helpers import AGENT_ROOT, load_manifest, resolve_run_dir
from agent.run_trace_buffer import append_event
from models.run_context import RunContext
from models.schemas import RawPage
from pipeline.extractor_agent import ExtractorAgent
from pipeline.keyword_extract import load_raw_page
from pipeline.fetcher import Fetcher
from pipeline.validator import FactValidator
from rag.ingest import FactIngester

PREFERRED_DOMAINS = ("who.int", "nhs.uk", "wikipedia.org", "cdc.gov", "nih.gov")
BLOCKED_DOMAINS = ("facebook.com", "twitter.com", "x.com", "youtube.com", "instagram.com")


def _normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        url = "https://" + url.strip()
        parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def _existing_urls(run_dir: Path) -> set[str]:
    urls: set[str] = set()
    manifest = load_manifest(run_dir)
    for url in manifest.get("urls") or []:
        urls.add(_normalize_url(url))
    raw_dir = run_dir / "raw_pages"
    if raw_dir.exists():
        for path in raw_dir.glob("*.json"):
            try:
                page = load_raw_page(path)
                urls.add(_normalize_url(page.url))
            except Exception:  # noqa: BLE001
                continue
    return urls


def _score_url(url: str) -> int:
    domain = _domain(url)
    if any(blocked in domain for blocked in BLOCKED_DOMAINS):
        return -100
    score = 0
    for idx, preferred in enumerate(PREFERRED_DOMAINS):
        if preferred in domain:
            score += 50 - idx
    if domain.endswith(".gov") or domain.endswith(".edu"):
        score += 10
    return score


def _extract_urls_from_search(result: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(result, dict):
        for item in result.get("organic") or result.get("results") or []:
            if isinstance(item, dict):
                link = item.get("link") or item.get("url")
                if link:
                    urls.append(str(link))
        if result.get("url"):
            urls.append(str(result["url"]))
    elif isinstance(result, list):
        for item in result:
            urls.extend(_extract_urls_from_search(item))
    return urls


async def _search_urls(query: str, *, max_urls: int) -> list[str]:
    from utu.config import ConfigLoader
    from utu.tools import SearchToolkit

    append_event("kb_expand", "search_start", {"query": query})
    toolkit = SearchToolkit(ConfigLoader.load_toolkit_config("search"))
    search_query = f"{query} site:who.int OR site:nhs.uk OR site:wikipedia.org"
    try:
        raw = await toolkit.search(search_query, num_results=max(max_urls * 3, 8))
    except Exception as exc:  # noqa: BLE001
        append_event("kb_expand", "search_failed", {"error": str(exc)})
        return []

    candidates = _extract_urls_from_search(raw)
    ranked = sorted({_normalize_url(u) for u in candidates if u.startswith("http")}, key=_score_url, reverse=True)
    append_event("kb_expand", "search_done", {"candidates": len(ranked), "top": ranked[:5]})
    return ranked[:max_urls]


def _next_raw_index(run_dir: Path) -> int:
    raw_dir = run_dir / "raw_pages"
    if not raw_dir.exists():
        return 1
    indices = []
    for path in raw_dir.glob("*.json"):
        match = re.match(r"(\d+)\.json$", path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


async def expand_knowledge_base(
    topic: str,
    run_id: str = "",
    *,
    max_urls: int = 3,
) -> dict[str, Any]:
    """Search → fetch → extract → validate → append ingest for new URLs on an existing run."""
    topic = topic.strip()
    if not topic:
        return {"error": "topic 不能为空", "expanded": False}

    run_dir = resolve_run_dir(run_id or None)
    run_context = RunContext(AGENT_ROOT, run_id=run_dir.name)
    run_context.ensure_dirs()

    existing = _existing_urls(run_dir)
    discovered = await _search_urls(topic, max_urls=max_urls)
    new_urls = [url for url in discovered if _normalize_url(url) not in existing][:max_urls]

    if not new_urls:
        return {
            "expanded": False,
            "run_id": run_dir.name,
            "message": "搜索未发现可追加的新 URL（可能已全部收录或搜索 API 未配置）",
            "discovered": discovered[:5],
            "existing_count": len(existing),
        }

    append_event("kb_expand", "fetch_start", {"urls": new_urls})
    fetcher = Fetcher(run_context=run_context)
    start_index = _next_raw_index(run_dir)
    pages: list[RawPage] = []
    for offset, url in enumerate(new_urls):
        page = await fetcher.fetch_one(url)
        pages.append(page)
        fetcher.save_page(page, start_index + offset)

    success_pages = [p for p in pages if len(p.text) >= 100]
    append_event("kb_expand", "fetch_done", {"success": len(success_pages), "total": len(pages)})

    if not success_pages:
        return {
            "expanded": False,
            "run_id": run_dir.name,
            "message": "新 URL 抓取失败或无正文",
            "urls_attempted": new_urls,
        }

    extractor = ExtractorAgent(run_context=run_context)
    await extractor.load_agent()
    fact_groups: list[list] = []
    for offset, page in enumerate(success_pages):
        facts = await extractor.extract_one(page)
        idx = start_index + offset
        extractor.save_facts(facts, idx)
        fact_groups.append(facts)

    fact_count = sum(len(group) for group in fact_groups)
    validator = FactValidator(run_context)
    _, validation_stats = validator.validate_run()

    ingester = FactIngester(run_context, include_aggregated=False)
    ingest_manifest = ingester.append_ingest()

    manifest = load_manifest(run_dir)
    urls_list = list(manifest.get("urls") or [])
    urls_list.extend(new_urls)
    stats = manifest.get("stats") or {}
    stats["kb_expand_count"] = int(stats.get("kb_expand_count") or 0) + len(new_urls)
    stats["knowledge_db_document_count"] = ingest_manifest.get("document_count")
    stats["validated_fact_count"] = validation_stats.get("output_count")
    manifest["urls"] = urls_list
    manifest["stats"] = stats
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    append_event("kb_expand", "ingest_done", ingest_manifest)
    return {
        "expanded": True,
        "run_id": run_dir.name,
        "topic": topic,
        "new_urls": new_urls,
        "fetch_success": len(success_pages),
        "new_facts": fact_count,
        "validated_facts": validation_stats.get("output_count"),
        "ingest": ingest_manifest,
        "message": f"已追加 {len(new_urls)} 个 URL，入库后共 {ingest_manifest.get('document_count')} 篇文档",
    }
