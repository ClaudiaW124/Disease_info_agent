"""Aggregator Agent — validated facts → DiseaseProfile → aggregated_data.json."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from models.run_context import RunContext
from models.schemas import DiseaseProfile, ExtractedFact
from pipeline.extractor_agent import extract_json_from_text
from pipeline.keyword_extract import TARGET_DISEASES
from pipeline.validator import load_facts_from_dir

from utu.agents import SimpleAgent

MIN_SNIPPET = 20


def _unique_sources(facts: list[ExtractedFact]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for fact in facts:
        if fact.source_url not in seen:
            seen.add(fact.source_url)
            sources.append(fact.source_url)
    return sources


def _fallback_key_points(facts: list[ExtractedFact], limit: int = 8) -> list[str]:
    points: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        snippet = fact.content.strip()
        if len(snippet) < MIN_SNIPPET:
            continue
        key = snippet[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        points.append(snippet[:220])
        if len(points) >= limit:
            break
    return points


def _fallback_summary(disease: str, facts: list[ExtractedFact], key_points: list[str]) -> str:
    if not facts:
        return f"暂无关于{disease}的有效事实条目。"
    lead = key_points[0] if key_points else facts[0].content
    return f"{disease}共整理 {len(facts)} 条来源事实。代表性信息：{lead[:180]}"


class AggregatorAgent:
    """P3-2 — LLM summary strictly grounded in validated facts."""

    def __init__(self, run_context: RunContext, use_llm: bool = True) -> None:
        self.run_context = run_context
        self.use_llm = use_llm
        self._agent: SimpleAgent | None = None

    async def load_agent(self) -> None:
        if not self.use_llm:
            return
        model_name = os.getenv("RAG_LLM_MODEL") or os.getenv("UTU_LLM_MODEL")
        self._agent = SimpleAgent(config="disease_info/aggregator", model=model_name)

    def load_validated_facts(self) -> list[ExtractedFact]:
        validated_dir = self.run_context.validated_facts_dir
        if validated_dir.exists() and any(validated_dir.glob("*.json")):
            return load_facts_from_dir(validated_dir)
        return load_facts_from_dir(self.run_context.facts_dir)

    def group_by_disease(self, facts: list[ExtractedFact]) -> dict[str, list[ExtractedFact]]:
        grouped = {disease: [] for disease in TARGET_DISEASES}
        for fact in facts:
            if fact.disease in grouped:
                grouped[fact.disease].append(fact)
        return grouped

    def build_prompt(self, disease: str, facts: list[ExtractedFact]) -> str:
        lines = [
            f"目标疾病: {disease}",
            f"事实条数: {len(facts)}",
            "",
            "以下是从网页中抽取并已校验的事实，请仅基于这些内容生成摘要，禁止添加事实列表中没有的信息。",
            "",
        ]
        for index, fact in enumerate(facts[:40], 1):
            lines.append(f"{index}. [{fact.field}] {fact.content}")
        lines.extend(
            [
                "",
                "请严格输出 JSON 对象，不要 markdown 代码块，不要额外解释：",
                '{',
                '  "summary": "100-200字中文摘要，仅基于上述事实",',
                '  "key_points": ["要点1", "要点2", "要点3"]',
                "}",
            ]
        )
        return "\n".join(lines)

    def parse_llm_profile(self, raw: str, disease: str, facts: list[ExtractedFact]) -> tuple[str, list[str]]:
        parsed = extract_json_from_text(raw)
        summary = ""
        key_points: list[str] = []
        if isinstance(parsed, dict):
            summary = str(parsed.get("summary", "")).strip()
            raw_points = parsed.get("key_points", [])
            if isinstance(raw_points, list):
                key_points = [str(item).strip() for item in raw_points if str(item).strip()]

        if not key_points:
            key_points = _fallback_key_points(facts)
        if not summary:
            summary = _fallback_summary(disease, facts, key_points)
        return summary, key_points[:8]

    async def summarize_disease(self, disease: str, facts: list[ExtractedFact]) -> DiseaseProfile:
        if not facts:
            return DiseaseProfile(disease=disease, summary=f"暂无关于{disease}的有效事实。", sources=[])

        if self.use_llm and self._agent is not None:
            try:
                recorder = await self._agent.run(self.build_prompt(disease, facts))
                summary, key_points = self.parse_llm_profile(recorder.final_output, disease, facts)
            except Exception as exc:
                print(f"    Aggregator LLM 失败 ({disease}): {exc}")
                key_points = _fallback_key_points(facts)
                summary = _fallback_summary(disease, facts, key_points)
        else:
            key_points = _fallback_key_points(facts)
            summary = _fallback_summary(disease, facts, key_points)

        return DiseaseProfile(
            disease=disease,
            summary=summary,
            key_points=key_points,
            sources=_unique_sources(facts),
            count=len(facts),
            items=facts,
        )

    def profiles_to_payload(self, profiles: list[DiseaseProfile], all_facts: list[ExtractedFact]) -> dict[str, Any]:
        sources = _unique_sources(all_facts)
        diseases: dict[str, Any] = {}
        for profile in profiles:
            diseases[profile.disease] = {
                "summary": profile.summary,
                "key_points": profile.key_points,
                "sources": profile.sources,
                "count": profile.count,
                "items": [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item.dict()
                    for item in profile.items
                ],
            }

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "target_diseases": TARGET_DISEASES,
            "total_sources": len(sources),
            "sources": sources,
            "profiles": [
                profile.model_dump(mode="json") if hasattr(profile, "model_dump") else profile.dict()
                for profile in profiles
            ],
            "diseases": diseases,
        }

    async def aggregate_run(self) -> dict[str, Any]:
        facts = self.load_validated_facts()
        grouped = self.group_by_disease(facts)
        profiles: list[DiseaseProfile] = []

        for disease in TARGET_DISEASES:
            disease_facts = grouped[disease]
            print(f"  整合 {disease}: {len(disease_facts)} 条事实")
            profiles.append(await self.summarize_disease(disease, disease_facts))

        payload = self.profiles_to_payload(profiles, facts)
        self.run_context.aggregated_data_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return payload
