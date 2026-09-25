"""Extractor Agent — LLM structured extraction from RawPage.text → ExtractedFact."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import tiktoken
from models.run_context import RunContext
from models.schemas import ExtractedFact, FactField, RawPage
from pipeline.keyword_extract import TARGET_DISEASES, KeywordExtractor, guess_diseases_from_url, load_raw_page

from utu.agents import SimpleAgent

MAX_TOKENS = 8000
VALID_FIELDS = {item.value for item in FactField}


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    text = text.strip()
    if not text:
        return text
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except Exception:
        return text[: max_tokens * 3]


def extract_json_from_text(raw: str) -> Any | None:
    if not raw or not raw.strip():
        return None

    patterns = [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"```(?:json)?\s*(\[.*?\])\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None


def normalize_disease(name: str, source_url: str) -> str:
    cleaned = (name or "").strip()
    if cleaned in TARGET_DISEASES:
        return cleaned
    for disease in TARGET_DISEASES:
        if disease in cleaned:
            return disease
    return guess_diseases_from_url(source_url)[0]


def parse_facts_lenient(raw: str, source_url: str) -> list[ExtractedFact]:
    """Salvage facts from malformed LLM JSON (common with small models)."""
    default_disease = guess_diseases_from_url(source_url)[0]
    facts: list[ExtractedFact] = []
    seen: set[str] = set()

    for block_match in re.finditer(r"\{[^{}]*?\"content\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"[^{}]*?\}", raw, re.DOTALL):
        block = block_match.group(0)
        content = block_match.group(1).encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
        content = content.replace("\\n", " ").replace("\\\"", '"').strip()
        if len(content) < 25:
            continue
        if content.count('"') > 8 or content.count("D D") > 2:
            continue

        field_match = re.search(r"\"field\"\s*:\s*\"(\w+)\"", block, re.IGNORECASE)
        field = field_match.group(1).lower() if field_match else FactField.GENERAL.value
        if field not in VALID_FIELDS:
            field = FactField.GENERAL.value

        disease_match = re.search(r"\"disease\"\s*:\s*\"([^\"]*)\"", block)
        disease = normalize_disease(disease_match.group(1) if disease_match else "", source_url)

        key = f"{disease}::{field}::{content[:120]}"
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            ExtractedFact(
                disease=disease,
                field=field,
                content=content[:800],
                source_url=source_url,
                confidence=0.4,
                context="extractor_agent_lenient",
            )
        )

    if facts:
        return facts[:20]

    for content_match in re.finditer(r"\"content\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"", raw):
        content = content_match.group(1).replace("\\n", " ").strip()
        if len(content) < 40 or content.count("D") > len(content) * 0.15:
            continue
        key = content[:120]
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            ExtractedFact(
                disease=default_disease,
                field=FactField.GENERAL.value,
                content=content[:800],
                source_url=source_url,
                confidence=0.3,
                context="extractor_agent_lenient",
            )
        )
    return facts[:20]


def serialize_facts(facts: list[ExtractedFact]) -> str:
    payload = [
        fact.model_dump(mode="json") if hasattr(fact, "model_dump") else fact.dict()
        for fact in facts
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


class ExtractorAgent:
    """P2 Extractor — SimpleAgent + schema validation."""

    def __init__(
        self,
        run_context: RunContext | None = None,
        output_dir: Path | None = None,
        fallback_keyword: bool = True,
    ) -> None:
        self.run_context = run_context
        if output_dir is not None:
            self.output_dir = output_dir
        elif run_context is not None:
            self.output_dir = run_context.facts_dir
        else:
            self.output_dir = None
        self.fallback_keyword = fallback_keyword
        self._agent: SimpleAgent | None = None
        self._keyword_extractor = KeywordExtractor()

    async def load_agent(self) -> None:
        model_name = os.getenv("RAG_LLM_MODEL") or os.getenv("UTU_LLM_MODEL")
        self._agent = SimpleAgent(config="disease_info/extractor", model=model_name)

    def build_prompt(self, page: RawPage) -> str:
        text = truncate_text(page.text)
        diseases = "、".join(TARGET_DISEASES)
        return (
            f"源 URL: {page.url}\n"
            f"目标疾病: {diseases}\n\n"
            f"请从以下正文中抽取与目标疾病相关的 facts，严格按 instructions 中的 JSON 格式输出。\n\n"
            f"--- 正文开始 ---\n{text}\n--- 正文结束 ---"
        )

    def parse_facts(self, raw: str, source_url: str) -> list[ExtractedFact]:
        parsed = extract_json_from_text(raw)
        items: list[Any] = []
        if parsed is not None:
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = parsed.get("facts", parsed.get("extracted_content", []))
                if isinstance(items, dict):
                    items = [items]

        facts: list[ExtractedFact] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if len(content) < 20:
                continue
            disease = normalize_disease(str(item.get("disease", "")).strip(), source_url)
            field = str(item.get("field", FactField.GENERAL.value)).strip().lower()
            if field not in VALID_FIELDS:
                field = FactField.GENERAL.value
            confidence = item.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            if confidence is not None:
                confidence = max(0.0, min(1.0, confidence))
            facts.append(
                ExtractedFact(
                    disease=disease,
                    field=field,
                    content=content[:800],
                    source_url=source_url,
                    confidence=confidence,
                    context="extractor_agent",
                )
            )

        if facts:
            return facts[:20]
        return parse_facts_lenient(raw, source_url)

    def save_facts(self, facts: list[ExtractedFact], index: int) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir 未设置，无法保存 ExtractedFact")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{index:03d}.json"
        path.write_text(serialize_facts(facts), encoding="utf-8")
        return path

    async def extract_one(self, page: RawPage) -> list[ExtractedFact]:
        if not self._agent:
            raise RuntimeError("Extractor agent 未加载，请先调用 load_agent()")
        if len(page.text.strip()) < 50:
            return []

        recorder = await self._agent.run(self.build_prompt(page))
        facts = self.parse_facts(recorder.final_output, page.url)
        if not facts and self.fallback_keyword:
            facts = self._keyword_extractor.extract_from_page(page)
            for fact in facts:
                fact.context = "extractor_keyword_fallback"
            if facts:
                print(f"    (Agent JSON 无效，已回退 keyword，{len(facts)} 条)")
        return facts

    async def extract_from_pages(self, pages: list[RawPage]) -> list[list[ExtractedFact]]:
        all_facts: list[list[ExtractedFact]] = []
        for index, page in enumerate(pages, 1):
            facts = await self.extract_one(page)
            all_facts.append(facts)
            if self.output_dir is not None:
                self.save_facts(facts, index)
            print(f"  [{index}/{len(pages)}] {page.url} -> {len(facts)} facts (Agent)")
        return all_facts

    async def extract_from_run(self, run_context: RunContext) -> list[list[ExtractedFact]]:
        raw_files = sorted(run_context.raw_pages_dir.glob("*.json"))
        if not raw_files:
            raise FileNotFoundError(f"未找到 raw_pages: {run_context.raw_pages_dir}")
        pages = [load_raw_page(path) for path in raw_files]
        self.output_dir = run_context.facts_dir
        return await self.extract_from_pages(pages)
