"""B2 LangGraph RAG — retrieve → grade → rewrite/generate → validate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, TypedDict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rag.common import (
    build_llm,
    extract_urls,
    format_documents,
    load_vectorstore,
    parse_yes_no,
    search_with_relevance,
    serialize_document,
)
from rag.settings import MIN_RELEVANCE_SCORE, load_env

MAX_REWRITE_LOOPS = 2
MAX_REGENERATE = 1
TOP_K = 5
COMPARE_TOP_K_EACH = 4

COMPARE_MARKERS = (
    "有什么区别",
    "有何不同",
    "有什么不同",
    "的区别",
    "相比",
    "对比",
    "传播方式有何不同",
    "传播方式不同",
    "差异",
)

DISEASE_ALIASES: dict[str, list[str]] = {
    "登革病毒": ["登革病毒", "登革热", "登革"],
    "猩红热": ["猩红热", "猩红"],
    "裂谷热": ["裂谷热", "裂谷"],
    "流感": ["流感", "流行性感冒"],
}

GRADE_PROMPT = """你是检索质量评估器。根据用户问题和检索到的文档，判断文档是否足以回答问题。
只输出 yes 或 no，不要解释。

规则：
1. 若文档包含与问题相关的医学事实，输出 yes（不要求文档能完美回答每一个「为什么」）
2. 「能否用某药治疗某病」类问题：只要文档含该疾病治疗/管理信息，输出 yes
3. 只有文档完全无关或毫无可用信息时，才输出 no"""

COMPARE_GRADE_PROMPT = """你是检索质量评估器。用户问的是两种疾病的对比问题。
判断检索文档是否同时包含两种疾病的相关信息，足以进行对比回答。
只输出 yes 或 no，不要解释。"""

REWRITE_PROMPT = """你是问题改写器。知识库只包含登革病毒、猩红热、裂谷热、流感相关医学事实。

规则：
1. 若原问题与上述四类疾病完全无关，只输出：UNRELATED
2. 若原问题同时提到两种疾病并要求对比/区别，保留两种疾病名称，改写为更便于检索的对比问句，只输出改写后的问题
3. 否则将问题改写为更适合检索的简短中文问题，只输出改写后的问题"""

GENERATE_PROMPT = """你是医学知识库问答助手。只能根据提供的检索片段回答，禁止编造。
用中文回答；末尾单独一行写：来源：<实际URL1>, <实际URL2>（从片段 metadata 里取真实 URL，去重，不要写占位符）。

若用户问某药能否治疗某病，而片段中未提到该药：应明确说明「现有资料未支持该用法」，并基于片段说明该病的实际治疗/管理方式。
若用户问「为什么」，在片段信息有限时，可基于已有事实给出部分解释，不要编造片段以外的机制。"""

COMPARE_GENERATE_HINT = """用户问的是两种疾病的对比问题。请分别说明两种疾病的相关信息，并明确指出主要异同。
只能根据检索片段回答，禁止编造。"""

VALIDATE_PROMPT = """你是答案校验器。判断“答案”是否完全由“文档片段”支持，是否存在编造。
只输出 yes 或 no，不要解释。"""

REFUSAL = "根据现有知识库无法回答该问题。"


OUT_OF_KB_TERMS = (
    "埃博拉",
    "ebola",
    "猴痘",
    "mpox",
    "新冠",
    "covid",
    "艾滋病",
    "hiv",
    "疟疾",
    "malaria",
    "霍乱",
    "cholera",
    "鼠疫",
    "plague",
)


def question_mentions_in_kb_disease(question: str) -> bool:
    for aliases in DISEASE_ALIASES.values():
        if any(alias in question for alias in aliases):
            return True
    return False


def question_mentions_out_of_kb(question: str) -> bool:
    lowered = question.lower()
    return any(term.lower() in lowered for term in OUT_OF_KB_TERMS)


def detect_comparison(question: str) -> dict | None:
    """Detect two-disease comparison questions and return metadata for split retrieval."""
    if not any(marker in question for marker in COMPARE_MARKERS):
        if "和" not in question and "与" not in question:
            return None
        if not any(word in question for word in ("不同", "区别", "对比", "相比", "差异")):
            return None

    found: list[str] = []
    for canonical, aliases in DISEASE_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in question:
                found.append(canonical)
                break

    if len(found) < 2:
        return None

    aspect: str | None = None
    for keyword in ("传播", "症状", "预防", "治疗", "病因", "诊断"):
        if keyword in question:
            aspect = keyword
            break

    return {
        "diseases": found[:2],
        "aspect": aspect,
        "original": question,
    }


def build_compare_subqueries(compare_info: dict) -> list[str]:
    """Build focused sub-queries so each disease gets its own retrieval pass."""
    disease_a, disease_b = compare_info["diseases"]
    aspect = compare_info.get("aspect")
    original = compare_info["original"]

    if aspect == "传播" or "传播" in original:
        return [f"{disease_a}如何传播？", f"{disease_b}如何传播？"]
    if aspect == "症状" or "症状" in original:
        return [f"{disease_a}有哪些症状？", f"{disease_b}有哪些症状？"]
    if "预防" in original:
        return [f"{disease_a}怎么预防？", f"{disease_b}怎么预防？"]
    if "治疗" in original:
        return [f"{disease_a}如何治疗？", f"{disease_b}如何治疗？"]
    return [
        f"{disease_a}的主要症状和特点是什么？",
        f"{disease_b}的主要症状和特点是什么？",
    ]


def _merge_retrieved_chunks(chunks_by_query: list[list[tuple]], *, limit: int) -> list[tuple]:
    merged: list[tuple] = []
    seen: set[str] = set()
    for chunks in chunks_by_query:
        for doc, score in chunks:
            meta = doc.metadata or {}
            key = f"{meta.get('source_url', '')}::{doc.page_content[:120]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, score))
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged[:limit]


class RagState(TypedDict, total=False):
    question: str
    original_question: str
    rewrite_history: list[str]
    documents: list[dict]
    answer: str
    citations: list[str]
    relevance_score: float
    grade: str
    rewrite_count: int
    regenerate_count: int
    validate_passed: bool
    refused: bool
    steps: list[str]
    compare_info: dict | None
    compare_subqueries: list[str]


class RagAnswer(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
    relevance_score: float = 0.0
    grade: str = ""
    rewrite_count: int = 0
    regenerate_count: int = 0
    validate_passed: bool = False
    path: str = "graph"
    original_question: str = ""
    rewrite_history: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


def _initial_state(question: str) -> RagState:
    compare_info = detect_comparison(question)
    return {
        "question": question,
        "original_question": question,
        "rewrite_history": [],
        "documents": [],
        "answer": "",
        "citations": [],
        "relevance_score": 0.0,
        "grade": "",
        "rewrite_count": 0,
        "regenerate_count": 0,
        "validate_passed": False,
        "refused": False,
        "steps": [],
        "compare_info": compare_info,
        "compare_subqueries": build_compare_subqueries(compare_info) if compare_info else [],
    }


def build_rag_graph(knowledge_db_dir: Path, *, top_k: int = TOP_K):
    vectorstore = load_vectorstore(knowledge_db_dir)
    llm = build_llm()

    def retrieve(state: RagState) -> RagState:
        compare_info = state.get("compare_info")
        subqueries = state.get("compare_subqueries") or []
        steps = list(state.get("steps") or [])

        if compare_info and subqueries:
            chunks_by_query = [
                search_with_relevance(vectorstore, sub_q, top_k=COMPARE_TOP_K_EACH) for sub_q in subqueries
            ]
            merged = _merge_retrieved_chunks(chunks_by_query, limit=TOP_K * 2)
            documents = [serialize_document(doc, score) for doc, score in merged]
            best_score = merged[0][1] if merged else 0.0
            diseases = " vs ".join(compare_info["diseases"])
            steps.append(
                f"retrieve(compare={diseases}, subqs={len(subqueries)}, score={best_score:.3f}, hits={len(documents)})"
            )
        else:
            chunks = search_with_relevance(vectorstore, state["question"], top_k=top_k)
            documents = [serialize_document(doc, score) for doc, score in chunks]
            best_score = chunks[0][1] if chunks else 0.0
            steps.append(f"retrieve(score={best_score:.3f}, hits={len(documents)})")

        return {
            **state,
            "documents": documents,
            "relevance_score": best_score,
            "steps": steps,
        }

    def grade_documents(state: RagState) -> RagState:
        steps = list(state.get("steps") or [])
        compare_info = state.get("compare_info")
        min_score = MIN_RELEVANCE_SCORE * (0.8 if compare_info else 1.0)

        if not state.get("documents") or state.get("relevance_score", 0.0) < min_score:
            steps.append("grade=no(threshold)")
            return {**state, "grade": "no", "steps": steps}

        context = format_documents(state["documents"])
        grade_prompt = COMPARE_GRADE_PROMPT if compare_info else GRADE_PROMPT
        question_for_grade = state.get("original_question") or state["question"]
        response = llm.invoke(
            [
                SystemMessage(content=grade_prompt),
                HumanMessage(content=f"问题：{question_for_grade}\n\n文档：\n{context}\n\n这些文档是否相关且足以回答？"),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        grade = parse_yes_no(content)
        original = state.get("original_question") or state["question"]
        if grade == "no" and state.get("relevance_score", 0.0) >= 0.55 and question_mentions_in_kb_disease(original):
            grade = "yes"
            steps.append("grade=yes(relevance_fallback)")
        else:
            steps.append(f"grade={grade}(llm)")
        return {**state, "grade": grade, "steps": steps}

    def rewrite_question(state: RagState) -> RagState:
        compare_info = state.get("compare_info")
        response = llm.invoke(
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(content=f"原问题：{state.get('original_question') or state['question']}"),
            ]
        )
        rewritten = response.content if isinstance(response.content, str) else str(response.content)
        rewritten = rewritten.strip()
        steps = list(state.get("steps") or [])
        history = list(state.get("rewrite_history") or [])

        if rewritten.upper() == "UNRELATED":
            steps.append("rewrite=UNRELATED")
            return {
                **state,
                "rewrite_count": MAX_REWRITE_LOOPS,
                "steps": steps,
            }

        history.append(rewritten)
        steps.append(f"rewrite -> {rewritten}")

        next_compare = detect_comparison(rewritten) or compare_info
        next_subqueries = build_compare_subqueries(next_compare) if next_compare else []

        return {
            **state,
            "question": rewritten,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "rewrite_history": history,
            "compare_info": next_compare,
            "compare_subqueries": next_subqueries,
            "documents": [],
            "grade": "",
            "steps": steps,
        }

    def refuse(state: RagState) -> RagState:
        steps = list(state.get("steps") or [])
        steps.append("refuse")
        return {
            **state,
            "answer": REFUSAL,
            "citations": [],
            "refused": True,
            "validate_passed": False,
            "steps": steps,
        }

    def generate(state: RagState) -> RagState:
        context = format_documents(state["documents"])
        compare_info = state.get("compare_info")
        question_for_generate = state.get("original_question") or state["question"]
        extra_hint = f"\n\n{COMPARE_GENERATE_HINT}" if compare_info else ""
        response = llm.invoke(
            [
                SystemMessage(content=GENERATE_PROMPT),
                HumanMessage(
                    content=(
                        f"问题：{question_for_generate}\n\n检索片段：\n{context}\n\n请基于片段回答。{extra_hint}"
                    )
                ),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)
        citations = extract_urls(answer, state["documents"])
        regenerate_count = state.get("regenerate_count", 0)
        if state.get("answer"):
            regenerate_count += 1
        refused = REFUSAL in answer
        steps = list(state.get("steps") or [])
        steps.append("generate")
        return {
            **state,
            "answer": answer,
            "citations": citations,
            "refused": refused,
            "regenerate_count": regenerate_count,
            "validate_passed": False,
            "steps": steps,
        }

    def validate_answer(state: RagState) -> RagState:
        if state.get("refused"):
            steps = list(state.get("steps") or [])
            steps.append("validate=skip(refused)")
            return {**state, "validate_passed": True, "steps": steps}
        response = llm.invoke(
            [
                SystemMessage(content=VALIDATE_PROMPT),
                HumanMessage(
                    content=(
                        f"文档：\n{format_documents(state['documents'])}\n\n"
                        f"答案：{state['answer']}\n\n答案是否被文档完全支持？"
                    )
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        passed = parse_yes_no(content) == "yes"
        steps = list(state.get("steps") or [])
        steps.append(f"validate={'yes' if passed else 'no'}")
        return {**state, "validate_passed": passed, "steps": steps}

    def guard_question(state: RagState) -> RagState:
        steps = list(state.get("steps") or [])
        original = state.get("original_question") or state.get("question", "")
        if question_mentions_out_of_kb(original):
            steps.append("guard=out_of_kb")
            return {**state, "rewrite_count": MAX_REWRITE_LOOPS, "grade": "no", "steps": steps}
        steps.append("guard=ok")
        return {**state, "steps": steps}

    def route_after_guard(state: RagState) -> Literal["retrieve", "refuse"]:
        if question_mentions_out_of_kb(state.get("original_question") or state.get("question", "")):
            return "refuse"
        return "retrieve"

    def route_after_grade(state: RagState) -> Literal["generate", "rewrite_question", "refuse"]:
        if state.get("grade") == "yes":
            return "generate"
        if state.get("rewrite_count", 0) >= MAX_REWRITE_LOOPS:
            return "refuse"
        return "rewrite_question"

    def route_after_validate(state: RagState) -> Literal["generate", "__end__"]:
        if state.get("validate_passed"):
            return "__end__"
        if state.get("regenerate_count", 0) >= MAX_REGENERATE:
            return "__end__"
        return "generate"

    workflow = StateGraph(RagState)
    workflow.add_node("guard_question", guard_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate", generate)
    workflow.add_node("validate_answer", validate_answer)
    workflow.add_node("refuse", refuse)

    workflow.add_edge(START, "guard_question")
    workflow.add_conditional_edges(
        "guard_question",
        route_after_guard,
        {"retrieve": "retrieve", "refuse": "refuse"},
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite_question": "rewrite_question",
            "refuse": "refuse",
        },
    )
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_edge("generate", "validate_answer")
    workflow.add_conditional_edges(
        "validate_answer",
        route_after_validate,
        {
            "generate": "generate",
            "__end__": END,
        },
    )
    workflow.add_edge("refuse", END)
    return workflow.compile()


def state_to_answer(state: RagState) -> RagAnswer:
    return RagAnswer(
        question=state.get("question", ""),
        answer=state.get("answer", REFUSAL),
        citations=state.get("citations") or [],
        refused=bool(state.get("refused")),
        relevance_score=float(state.get("relevance_score") or 0.0),
        grade=state.get("grade") or "",
        rewrite_count=int(state.get("rewrite_count") or 0),
        regenerate_count=int(state.get("regenerate_count") or 0),
        validate_passed=bool(state.get("validate_passed")),
        original_question=state.get("original_question") or state.get("question", ""),
        rewrite_history=list(state.get("rewrite_history") or []),
        steps=list(state.get("steps") or []),
    )


_GRAPH_CACHE: dict[str, object] = {}


def get_rag_graph(knowledge_db_dir: Path | str, *, top_k: int = TOP_K):
    """Compile once per knowledge_db path; rebuilds are too slow for MCP."""
    load_env()
    db_dir = Path(knowledge_db_dir)
    cache_key = f"{db_dir.resolve()}::{top_k}::v4"
    graph = _GRAPH_CACHE.get(cache_key)
    if graph is None:
        graph = build_rag_graph(db_dir, top_k=top_k)
        _GRAPH_CACHE[cache_key] = graph
    return graph


def run_rag_graph(
    question: str,
    knowledge_db_dir: Path | str,
    *,
    top_k: int = TOP_K,
) -> RagAnswer:
    answer, _ = run_rag_graph_eval(question, knowledge_db_dir, top_k=top_k)
    return answer


def run_rag_graph_eval(
    question: str,
    knowledge_db_dir: Path | str,
    *,
    top_k: int = TOP_K,
) -> tuple[RagAnswer, list[str]]:
    """Run graph and return answer plus retrieved context texts (for eval / Ragas)."""
    graph = get_rag_graph(knowledge_db_dir, top_k=top_k)
    final_state = graph.invoke(_initial_state(question))
    contexts = [str(doc.get("content", "")) for doc in (final_state.get("documents") or []) if doc.get("content")]
    return state_to_answer(final_state), contexts
