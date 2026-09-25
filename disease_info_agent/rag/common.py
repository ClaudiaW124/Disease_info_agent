"""Shared RAG helpers for ask.py and graph.py."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag.settings import COLLECTION_NAME, require_embedding_env, require_llm_env


def build_embeddings() -> OpenAIEmbeddings:
    base_url, api_key, model = require_embedding_env()
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
    )


def build_llm() -> ChatOpenAI:
    base_url, api_key, model = require_llm_env()
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        extra_body={"enable_thinking": False},
    )


def load_vectorstore(knowledge_db_dir: Path) -> Chroma:
    if not knowledge_db_dir.exists():
        raise FileNotFoundError(f"knowledge_db 不存在，请先运行 ingest: {knowledge_db_dir}")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
        persist_directory=str(knowledge_db_dir),
    )


def serialize_document(doc: Document, score: float) -> dict:
    meta = doc.metadata or {}
    return {
        "content": doc.page_content,
        "disease": meta.get("disease", ""),
        "field": meta.get("field", ""),
        "source_url": meta.get("source_url", ""),
        "score": score,
    }


def format_documents(documents: list[dict]) -> str:
    lines: list[str] = []
    for index, doc in enumerate(documents, 1):
        lines.append(
            f"[{index}] disease={doc.get('disease', '')} field={doc.get('field', '')}\n"
            f"url={doc.get('source_url', '')}\n"
            f"content={doc.get('content', '')}\n"
            f"relevance={doc.get('score', 0.0):.3f}"
        )
    return "\n\n".join(lines)


def extract_urls(text: str, documents: list[dict]) -> list[str]:
    urls = re.findall(r"https?://[^\s,<>]+", text)
    if urls:
        return list(dict.fromkeys(urls))
    from_docs = [doc["source_url"] for doc in documents if doc.get("source_url")]
    return list(dict.fromkeys(from_docs))


def parse_yes_no(raw: str) -> str:
    text = raw.strip().lower()
    if "yes" in text or "是" in text or text.startswith("y"):
        return "yes"
    return "no"


def distance_to_relevance(distance: float) -> float:
    """Map Chroma L2 distance to a 0~1 relevance score."""
    if distance < 0:
        return 0.0
    return 1.0 / (1.0 + distance)


def search_with_relevance(vectorstore: Chroma, query: str, *, top_k: int = 5) -> list[tuple[Document, float]]:
    """Retrieve documents with normalized relevance scores in [0, 1]."""
    chunks = vectorstore.similarity_search_with_score(query, k=top_k)
    return [(doc, distance_to_relevance(distance)) for doc, distance in chunks]

