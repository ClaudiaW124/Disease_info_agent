"""Temporary keyword-based extraction from RawPage.text → ExtractedFact."""

from __future__ import annotations

import json
import re
from pathlib import Path

from models.run_context import RunContext
from models.schemas import ExtractedFact, FactField, RawPage

TARGET_DISEASES = ["登革病毒", "猩红热", "裂谷热", "流感"]
DISEASE_KEYWORDS: dict[str, list[str]] = {
    "登革病毒": ["登革", "dengue"],
    "猩红热": ["猩红热", "scarlet fever", "scarlet"],
    "裂谷热": ["裂谷热", "rift valley"],
    "流感": ["流感", "influenza", "flu"],
}


def guess_diseases_from_url(url: str) -> list[str]:
    url_lower = url.lower()
    matched: list[str] = []
    if "dengue" in url_lower:
        matched.append("登革病毒")
    if "scarlet" in url_lower:
        matched.append("猩红热")
    if "rift" in url_lower or "valley" in url_lower:
        matched.append("裂谷热")
    if "influenza" in url_lower or "flu" in url_lower or "流感" in url_lower:
        matched.append("流感")
    return matched or TARGET_DISEASES.copy()


def serialize_fact(fact: ExtractedFact) -> str:
    payload = fact.model_dump(mode="json") if hasattr(fact, "model_dump") else fact.dict()
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def load_raw_page(path: Path) -> RawPage:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RawPage(**data)


class KeywordExtractor:
    """P1 temporary extractor — no LLM, keyword sentence split only."""

    def __init__(self, run_context: RunContext | None = None, output_dir: Path | None = None) -> None:
        self.run_context = run_context
        if output_dir is not None:
            self.output_dir = output_dir
        elif run_context is not None:
            self.output_dir = run_context.facts_dir
        else:
            self.output_dir = None

    def extract_from_page(self, page: RawPage) -> list[ExtractedFact]:
        text = re.sub(r"\s+", " ", page.text).strip()
        if not text:
            return []

        chunks = re.split(r"(?<=[。！？.!?])\s+|\n{2,}", text)
        facts: list[ExtractedFact] = []

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 30:
                continue
            for disease, keywords in DISEASE_KEYWORDS.items():
                if any(keyword.lower() in chunk.lower() for keyword in keywords):
                    facts.append(
                        ExtractedFact(
                            disease=disease,
                            field=FactField.GENERAL.value,
                            content=chunk[:500],
                            source_url=page.url,
                            confidence=0.5,
                            context="keyword_extract",
                        )
                    )
                    break

        if not facts and len(text) >= 30:
            primary = guess_diseases_from_url(page.url)[0]
            facts.append(
                ExtractedFact(
                    disease=primary,
                    field=FactField.GENERAL.value,
                    content=text[:800],
                    source_url=page.url,
                    confidence=0.3,
                    context="keyword_extract_page_summary",
                )
            )

        return facts[:12]

    def save_facts(self, facts: list[ExtractedFact], index: int) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir 未设置，无法保存 ExtractedFact")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{index:03d}.json"
        payload = [
            fact.model_dump(mode="json") if hasattr(fact, "model_dump") else fact.dict()
            for fact in facts
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def extract_from_pages(self, pages: list[RawPage]) -> list[list[ExtractedFact]]:
        all_facts: list[list[ExtractedFact]] = []
        for index, page in enumerate(pages, 1):
            facts = self.extract_from_page(page)
            all_facts.append(facts)
            if self.output_dir is not None:
                self.save_facts(facts, index)
            print(f"  [{index}/{len(pages)}] {page.url} -> {len(facts)} facts")
        return all_facts

    def extract_from_run(self, run_context: RunContext) -> list[list[ExtractedFact]]:
        raw_files = sorted(run_context.raw_pages_dir.glob("*.json"))
        pages = [load_raw_page(path) for path in raw_files]
        self.output_dir = run_context.facts_dir
        return self.extract_from_pages(pages)
