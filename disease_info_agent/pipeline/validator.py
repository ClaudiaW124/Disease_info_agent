"""Validate and clean ExtractedFact lists before aggregation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from models.run_context import RunContext
from models.schemas import ExtractedFact
from pipeline.extractor_agent import serialize_facts
from pipeline.keyword_extract import DISEASE_KEYWORDS, TARGET_DISEASES, guess_diseases_from_url

MIN_CONTENT_LEN = 30
JACCARD_DUP_THRESHOLD = 0.9

NAVIGATION_PREFIXES = (
    "jump to content",
    "main menu",
    "navigation",
    "skip to main",
    "search search",
    "appearance donate",
    "create account",
    "toolbox",
)

JSON_FRAGMENT_MARKERS = (
    '"url"',
    '"content"',
    '"disease"',
    '"field"',
    '"confidence"',
    '"source_url"',
    '{"facts"',
    '": "http',
)


def token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower()) if len(token) > 1}


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def looks_like_json_fragment(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if stripped.startswith(("{", "[", '"')) and any(marker in stripped for marker in JSON_FRAGMENT_MARKERS):
        return True
    if re.match(r'^["\{\[\]:,\s]+', stripped) and ('"' in stripped or ":" in stripped):
        return True
    if stripped.count('"') >= 6 and stripped.count(":") >= 2:
        return True
    return False


def looks_like_navigation(content: str) -> bool:
    lowered = content.strip().lower()
    if not lowered:
        return True
    return any(lowered.startswith(prefix) for prefix in NAVIGATION_PREFIXES)


def disease_matches_url(disease: str, source_url: str) -> bool:
    expected = guess_diseases_from_url(source_url)
    if disease in expected:
        return True

    url_lower = source_url.lower()
    keywords = DISEASE_KEYWORDS.get(disease, [])
    if any(keyword.lower() in url_lower for keyword in keywords):
        return True

    if len(expected) == 1 and disease not in expected:
        return False
    return True


def load_facts_from_dir(facts_dir: Path) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    for path in sorted(facts_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            facts.append(ExtractedFact(**item))
    return facts


def validate_facts(facts: list[ExtractedFact]) -> tuple[list[ExtractedFact], dict[str, int]]:
    stats = {
        "input_count": len(facts),
        "removed_short": 0,
        "removed_json_fragment": 0,
        "removed_navigation": 0,
        "removed_disease_mismatch": 0,
        "removed_duplicate": 0,
        "output_count": 0,
    }
    kept: list[ExtractedFact] = []

    for fact in facts:
        content = fact.content.strip()
        if len(content) < MIN_CONTENT_LEN:
            stats["removed_short"] += 1
            continue
        if looks_like_json_fragment(content):
            stats["removed_json_fragment"] += 1
            continue
        if looks_like_navigation(content):
            stats["removed_navigation"] += 1
            continue
        if fact.disease not in TARGET_DISEASES:
            stats["removed_disease_mismatch"] += 1
            continue
        if not disease_matches_url(fact.disease, fact.source_url):
            stats["removed_disease_mismatch"] += 1
            continue

        duplicate = False
        for existing in kept:
            if existing.disease != fact.disease:
                continue
            if jaccard_similarity(existing.content, content) >= JACCARD_DUP_THRESHOLD:
                duplicate = True
                break
        if duplicate:
            stats["removed_duplicate"] += 1
            continue

        kept.append(fact)

    stats["output_count"] = len(kept)
    return kept, stats


def save_validated_facts(facts: list[ExtractedFact], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[ExtractedFact]] = {}
    for fact in facts:
        by_source.setdefault(fact.source_url, []).append(fact)

    for index, (_, group) in enumerate(sorted(by_source.items()), 1):
        path = output_dir / f"{index:03d}.json"
        path.write_text(serialize_facts(group), encoding="utf-8")


class FactValidator:
    """P3-1 — filter low-quality facts and write facts_validated/."""

    def __init__(self, run_context: RunContext) -> None:
        self.run_context = run_context

    def validate_run(self) -> tuple[list[ExtractedFact], dict[str, int]]:
        facts = load_facts_from_dir(self.run_context.facts_dir)
        validated, stats = validate_facts(facts)
        save_validated_facts(validated, self.run_context.validated_facts_dir)
        return validated, stats
