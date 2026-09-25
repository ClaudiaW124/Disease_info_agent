"""P4-3 / B2-4 — RAG ask: LangGraph by default, linear chain via --simple."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json

from langchain_core.messages import HumanMessage, SystemMessage

from rag.common import build_llm, extract_urls, format_documents, load_vectorstore, search_with_relevance
from rag.graph import run_rag_graph
from rag.settings import MIN_RELEVANCE_SCORE, load_env

SYSTEM_PROMPT = """你是医学知识库问答助手。只能根据用户提供的检索片段回答，禁止编造或引用片段以外的知识。

规则：
1. 用中文回答，简洁准确
2. 如果片段与问题无关、信息不足，或问题与登革病毒/猩红热/裂谷热/流感无关，必须明确回复：根据现有知识库无法回答该问题。
3. 回答末尾单独一行写：来源：<URL1>, <URL2>（去重，只写检索片段里出现的 URL；拒答时不写来源）"""

DEMO_QUESTIONS = [
    "登革病毒的主要症状有哪些？",
    "裂谷热是如何传播的？",
    "流感有哪些预防方法？",
    "猩红热和流感在症状上有什么区别？",
    "比特币今天的价格是多少？",
]


def ask_question_simple(
    question: str,
    knowledge_db_dir: Path,
    *,
    top_k: int = 5,
    min_relevance: float = MIN_RELEVANCE_SCORE,
) -> dict:
    """P4 linear RAG — retrieve once, optional threshold refuse, generate once."""
    vectorstore = load_vectorstore(knowledge_db_dir)
    chunks = search_with_relevance(vectorstore, question, top_k=top_k)
    best_score = chunks[0][1] if chunks else 0.0
    documents = [
        {
            "content": doc.page_content,
            "disease": (doc.metadata or {}).get("disease", ""),
            "field": (doc.metadata or {}).get("field", ""),
            "source_url": (doc.metadata or {}).get("source_url", ""),
            "score": score,
        }
        for doc, score in chunks
    ]

    if not chunks or best_score < min_relevance:
        return {
            "question": question,
            "answer": "根据现有知识库无法回答该问题。",
            "sources": [],
            "refused": True,
            "best_relevance": best_score,
            "retrieved_count": len(chunks),
            "path": "simple",
        }

    context = format_documents(documents)
    llm = build_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"问题：{question}\n\n检索到的知识片段：\n{context}\n\n请基于以上片段回答。"
            ),
        ]
    )
    answer = response.content if isinstance(response.content, str) else str(response.content)
    sources = extract_urls(answer, documents)
    refused = "根据现有知识库无法回答该问题" in answer
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "refused": refused,
        "best_relevance": best_score,
        "retrieved_count": len(chunks),
        "path": "simple",
    }


def ask_question_graph(question: str, knowledge_db_dir: Path, *, top_k: int = 5) -> dict:
    result = run_rag_graph(question, knowledge_db_dir, top_k=top_k)
    return {
        "question": result.question,
        "original_question": result.original_question,
        "answer": result.answer,
        "sources": result.citations,
        "refused": result.refused,
        "best_relevance": result.relevance_score,
        "retrieved_count": len(result.citations),
        "path": result.path,
        "grade": result.grade,
        "rewrite_count": result.rewrite_count,
        "rewrite_history": result.rewrite_history,
        "regenerate_count": result.regenerate_count,
        "validate_passed": result.validate_passed,
        "steps": result.steps,
    }


def ask_question(
    question: str,
    knowledge_db_dir: Path,
    *,
    top_k: int = 5,
    min_relevance: float = MIN_RELEVANCE_SCORE,
    simple: bool = False,
) -> dict:
    if simple:
        return ask_question_simple(
            question,
            knowledge_db_dir,
            top_k=top_k,
            min_relevance=min_relevance,
        )
    return ask_question_graph(question, knowledge_db_dir, top_k=top_k)


def print_result(result: dict, *, verbose: bool = False) -> None:
    original = result.get("original_question")
    if original and original != result["question"]:
        print(f"\n原问题：{original}")
    print(f"问：{result['question']}")
    print(f"答：{result['answer']}")
    if result.get("rewrite_history"):
        print("改写历史：")
        for index, rewritten in enumerate(result["rewrite_history"], 1):
            print(f"  {index}. {rewritten}")
    if result.get("sources"):
        print("引用：")
        for url in result["sources"]:
            print(f"  - {url}")
    extra = f"path={result.get('path', 'graph')}"
    if result.get("rewrite_count"):
        extra += f", rewrite={result['rewrite_count']}"
    if "validate_passed" in result:
        extra += f", validate={result['validate_passed']}"
    print(f"(best_relevance={result['best_relevance']:.3f}, refused={result['refused']}, {extra})")
    if verbose and result.get("steps"):
        print("流程：", " → ".join(result["steps"]))


def run_demo(knowledge_db_dir: Path, *, simple: bool = False, verbose: bool = False) -> list[dict]:
    results = []
    for question in DEMO_QUESTIONS:
        result = ask_question(question, knowledge_db_dir, simple=simple)
        print_result(result, verbose=verbose)
        results.append(result)
    return results


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="RAG ask — LangGraph (default) or linear P4 (--simple)")
    parser.add_argument("question", nargs="?", default=None, help="用户问题")
    parser.add_argument("--run-id", required=True, help="已 ingest 的 run_id")
    parser.add_argument("--demo", action="store_true", help="运行 5 个验收问题（含 1 个无关题）")
    parser.add_argument("--top-k", type=int, default=5, help="检索条数")
    parser.add_argument("--simple", action="store_true", help="使用 P4 线性 RAG（不走 LangGraph）")
    parser.add_argument("--verbose", action="store_true", help="打印 LangGraph 节点流程")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    knowledge_db_dir = project_root / "output" / "runs" / args.run_id / "knowledge_db"

    if args.demo:
        results = run_demo(knowledge_db_dir, simple=args.simple, verbose=args.verbose)
        suffix = "simple" if args.simple else "graph"
        output_path = knowledge_db_dir / f"demo_results_{suffix}.json"
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n演示结果已保存: {output_path}")
        return

    if not args.question:
        parser.error("请提供 question，或使用 --demo")

    result = ask_question(args.question, knowledge_db_dir, top_k=args.top_k, simple=args.simple)
    print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
