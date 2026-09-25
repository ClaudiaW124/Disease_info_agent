"""P4-2 — ingest validated facts into Chroma knowledge_db."""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 `python rag/ingest.py` 运行（否则找不到 rag / models 包）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from models.run_context import RunContext
from models.schemas import ExtractedFact
from pipeline.validator import load_facts_from_dir
from rag.settings import COLLECTION_NAME, load_env, require_embedding_env


def fact_to_document(fact: ExtractedFact) -> Document:
    return Document(
        page_content=f"[{fact.disease}/{fact.field}] {fact.content}",
        metadata={
            "disease": fact.disease,
            "field": fact.field,
            "source_url": fact.source_url,
            "content": fact.content,
        },
    )


def profile_to_documents(profile: dict) -> list[Document]:
    disease = profile.get("disease", "")
    summary = profile.get("summary", "").strip()
    sources = profile.get("sources") or []
    documents: list[Document] = []
    if summary:
        documents.append(
            Document(
                page_content=f"[{disease}/summary] {summary}",
                metadata={
                    "disease": disease,
                    "field": "summary",
                    "source_url": sources[0] if sources else "",
                    "content": summary,
                },
            )
        )
    for point in profile.get("key_points") or []:
        text = str(point).strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=f"[{disease}/key_point] {text}",
                metadata={
                    "disease": disease,
                    "field": "key_point",
                    "source_url": sources[0] if sources else "",
                    "content": text,
                },
            )
        )
    return documents


def load_documents(run_context: RunContext, *, include_aggregated: bool = True) -> list[Document]:
    validated_dir = run_context.validated_facts_dir
    facts_dir = run_context.facts_dir
    if validated_dir.exists() and any(validated_dir.glob("*.json")):
        facts = load_facts_from_dir(validated_dir)
    else:
        facts = load_facts_from_dir(facts_dir)

    documents = [fact_to_document(fact) for fact in facts]

    if include_aggregated and run_context.aggregated_data_path.exists():
        payload = json.loads(run_context.aggregated_data_path.read_text(encoding="utf-8"))
        for profile in payload.get("profiles") or []:
            documents.extend(profile_to_documents(profile))

    return documents


def build_embeddings() -> OpenAIEmbeddings:
    base_url, api_key, model = require_embedding_env()
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
    )


class FactIngester:
    """Write pipeline facts into a persistent Chroma store."""

    def __init__(self, run_context: RunContext, *, include_aggregated: bool = True) -> None:
        self.run_context = run_context
        self.include_aggregated = include_aggregated

    def ingest(self) -> dict:
        documents = load_documents(self.run_context, include_aggregated=self.include_aggregated)
        if not documents:
            raise ValueError(f"没有可入库的 facts: {self.run_context.run_dir}")

        db_dir = self.run_context.knowledge_db_dir
        db_dir.mkdir(parents=True, exist_ok=True)

        embeddings = build_embeddings()
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(db_dir),
        )

        manifest = {
            "run_id": self.run_context.run_id,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "collection_name": COLLECTION_NAME,
            "document_count": len(documents),
            "persist_directory": str(db_dir),
            "include_aggregated": self.include_aggregated,
        }
        manifest_path = db_dir / "ingest_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def append_ingest(self) -> dict:
        """Append new validated facts to an existing Chroma store (incremental KB expand)."""
        documents = load_documents(self.run_context, include_aggregated=self.include_aggregated)
        if not documents:
            raise ValueError(f"没有可入库的 facts: {self.run_context.run_dir}")

        db_dir = self.run_context.knowledge_db_dir
        db_dir.mkdir(parents=True, exist_ok=True)
        embeddings = build_embeddings()

        manifest_path = db_dir / "ingest_manifest.json"
        previous_count = 0
        if manifest_path.exists():
            previous_count = int(json.loads(manifest_path.read_text(encoding="utf-8")).get("document_count") or 0)

        if db_dir.exists() and any(db_dir.iterdir()) and manifest_path.exists():
            vectorstore = Chroma(
                persist_directory=str(db_dir),
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
            existing_keys = set()
            try:
                stored = vectorstore.get(include=["metadatas"])
                for meta in stored.get("metadatas") or []:
                    if meta:
                        existing_keys.add(f"{meta.get('source_url', '')}::{meta.get('content', '')[:80]}")
            except Exception:  # noqa: BLE001
                existing_keys = set()

            fresh_docs = []
            for doc in documents:
                meta = doc.metadata or {}
                key = f"{meta.get('source_url', '')}::{meta.get('content', '')[:80]}"
                if key not in existing_keys:
                    fresh_docs.append(doc)
            if fresh_docs:
                vectorstore.add_documents(fresh_docs)
            added = len(fresh_docs)
            total = previous_count + added
        else:
            Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(db_dir),
            )
            added = len(documents)
            total = len(documents)

        manifest = {
            "run_id": self.run_context.run_id,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "collection_name": COLLECTION_NAME,
            "document_count": total,
            "added_document_count": added,
            "persist_directory": str(db_dir),
            "include_aggregated": self.include_aggregated,
            "mode": "append",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def ingest_run(project_root: Path, run_id: str, *, include_aggregated: bool = True) -> dict:
    run_context = RunContext(project_root, run_id=run_id)
    if not run_context.run_dir.exists():
        raise FileNotFoundError(f"run 不存在: {run_context.run_dir}")
    return FactIngester(run_context, include_aggregated=include_aggregated).ingest()


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="P4-2: ingest facts into Chroma knowledge_db")
    parser.add_argument("--run-id", required=True, help="pipeline run_id")
    parser.add_argument("--facts-only", action="store_true", help="仅入库 facts，不含 aggregated 摘要")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    manifest = ingest_run(project_root, args.run_id, include_aggregated=not args.facts_only)
    print(f"入库完成: {manifest['document_count']} 条文档")
    print(f"knowledge_db: {manifest['persist_directory']}")


if __name__ == "__main__":
    main()
