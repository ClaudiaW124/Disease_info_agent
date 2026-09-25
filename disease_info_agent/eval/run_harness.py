"""B4 eval harness — golden RAG questions + pipeline pytest + HTML/JSON report.

Usage (repo root):
    uv run python -m disease_info_agent.eval.run_harness
    uv run python -m disease_info_agent.eval.run_harness --run-id 20260815-234448 --no-ragas
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

_AGENT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _AGENT_ROOT.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_REFUSAL = "根据现有知识库无法回答该问题。"


def _runs_root() -> Path:
    return _AGENT_ROOT / "output" / "runs"


def _resolve_run_dir(run_id: str | None) -> Path:
    if run_id:
        path = _runs_root() / run_id
        if not path.exists():
            raise FileNotFoundError(f"run 不存在: {path}")
        return path
    run_dirs = sorted([p for p in _runs_root().iterdir() if p.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        raise FileNotFoundError("还没有任何 pipeline run，请先跑 pipeline + ingest。")
    return run_dirs[-1]


def load_golden_questions(path: Path | None = None, *, tier: str | None = None) -> list[dict[str, Any]]:
    golden_path = path or (_GOLDEN_DIR / "rag_questions.jsonl")
    rows: list[dict[str, Any]] = []
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        item_tier = item.get("tier", "core")
        if tier is None or item_tier == tier:
            rows.append(item)
    return rows


def _keyword_pass(answer_text: str, keywords: list[str]) -> bool | None:
    if not keywords:
        return None
    lowered = answer_text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def load_pipeline_thresholds() -> dict[str, Any]:
    return json.loads((_GOLDEN_DIR / "pipeline_urls.json").read_text(encoding="utf-8"))


def _text_contains_any(text: str, fragments: list[str]) -> bool:
    lowered = text.lower()
    return any(frag.lower() in lowered for frag in fragments)


def _citation_pass(answer: dict[str, Any], expected_source_contains: list[str]) -> bool:
    if not expected_source_contains:
        return True
    citations = answer.get("citations") or []
    joined = " ".join(citations) + " " + str(answer.get("answer", ""))
    return _text_contains_any(joined, expected_source_contains)


def _behavior_pass(item: dict[str, Any], answer: dict[str, Any]) -> bool:
    should_answer = bool(item.get("should_answer", True))
    refused = bool(answer.get("refused"))
    answer_text = str(answer.get("answer", ""))
    if not should_answer:
        return refused or _REFUSAL in answer_text
    return not refused and _REFUSAL not in answer_text


def _heuristic_faithfulness(item: dict[str, Any], answer: dict[str, Any]) -> float:
    if not _behavior_pass(item, answer):
        return 0.0
    if not bool(item.get("should_answer", True)):
        return 1.0
    score = 0.5
    if answer.get("validate_passed"):
        score += 0.3
    expected = item.get("expected_source_contains") or []
    if _citation_pass(answer, expected):
        score += 0.2
    return min(score, 1.0)


def _heuristic_answer_relevancy(item: dict[str, Any], answer: dict[str, Any]) -> float:
    if not bool(item.get("should_answer", True)):
        return 1.0 if _behavior_pass(item, answer) else 0.0
    if not _behavior_pass(item, answer):
        return 0.0
    text = str(answer.get("answer", ""))
    diseases = item.get("expected_diseases") or []
    if diseases and any(d in text for d in diseases):
        return 0.9
    relevance = float(answer.get("relevance_score") or 0.0)
    return max(0.3, min(0.85, relevance + 0.2))


def _run_ragas_batch(rows: list[dict[str, Any]], *, use_ragas: bool) -> tuple[dict[str, dict[str, float | None]], str]:
    if not use_ragas:
        return {}, "disabled"
    try:
        import math

        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        print(f"[warn] Ragas not installed: {exc}", file=sys.stderr)
        return {}, "missing_deps"

    from rag.common import build_embeddings, build_llm
    from rag.settings import load_env

    load_env()
    llm = LangchainLLMWrapper(build_llm())
    embeddings = LangchainEmbeddingsWrapper(build_embeddings())

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["answer_text"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
        }
    )
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Ragas evaluate failed, using heuristics only: {exc}", file=sys.stderr)
        return {}, f"failed:{exc}"

    df = result.to_pandas()
    scores: dict[str, dict[str, float | None]] = {}
    for index, row in enumerate(rows):
        faith_val = df.iloc[index]["faithfulness"] if "faithfulness" in df.columns else None
        rel_val = df.iloc[index]["answer_relevancy"] if "answer_relevancy" in df.columns else None
        scores[row["id"]] = {
            "faithfulness": None if faith_val is None or (isinstance(faith_val, float) and math.isnan(faith_val)) else float(faith_val),
            "answer_relevancy": None if rel_val is None or (isinstance(rel_val, float) and math.isnan(rel_val)) else float(rel_val),
        }
    return scores, "ok"


def evaluate_rag_questions(
    run_dir: Path,
    *,
    limit: int | None = None,
    tier: str | None = None,
    use_ragas: bool = True,
) -> dict[str, Any]:
    from rag.graph import run_rag_graph_eval
    from rag.settings import load_env

    load_env()
    knowledge_db = run_dir / "knowledge_db"
    if not knowledge_db.exists():
        raise FileNotFoundError(f"run {run_dir.name} 缺少 knowledge_db，请先 ingest。")

    golden = load_golden_questions(tier=tier)
    if limit is not None:
        golden = golden[:limit]

    raw_rows: list[dict[str, Any]] = []
    for item in golden:
        answer_obj, contexts = run_rag_graph_eval(item["question"], knowledge_db)
        answer = answer_obj.model_dump()
        heuristic_f = _heuristic_faithfulness(item, answer)
        heuristic_r = _heuristic_answer_relevancy(item, answer)
        keywords = item.get("expected_keywords") or []
        raw_rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "tier": item.get("tier", "core"),
                "category": item.get("category"),
                "should_answer": item.get("should_answer", True),
                "answer_text": answer.get("answer", ""),
                "contexts": contexts if contexts else [""],
                "answer": answer,
                "behavior_pass": _behavior_pass(item, answer),
                "citation_pass": _citation_pass(answer, item.get("expected_source_contains") or []),
                "keyword_pass": _keyword_pass(answer.get("answer", ""), keywords),
                "heuristic_faithfulness": heuristic_f,
                "heuristic_answer_relevancy": heuristic_r,
            }
        )
        kw_flag = raw_rows[-1]["keyword_pass"]
        print(
            f"[rag] {item['id']} behavior={raw_rows[-1]['behavior_pass']} "
            f"cite={raw_rows[-1]['citation_pass']} kw={kw_flag}"
        )

    ragas_scores, ragas_status = _run_ragas_batch(raw_rows, use_ragas=use_ragas)

    results: list[dict[str, Any]] = []
    for row in raw_rows:
        ragas = ragas_scores.get(row["id"], {})
        faithfulness = ragas.get("faithfulness")
        answer_relevancy = ragas.get("answer_relevancy")
        if faithfulness is None:
            faithfulness = row["heuristic_faithfulness"]
            faithfulness_source = "heuristic"
        else:
            faithfulness_source = "ragas"
        if answer_relevancy is None:
            answer_relevancy = row["heuristic_answer_relevancy"]
            relevancy_source = "heuristic"
        else:
            relevancy_source = "ragas"

        ragas_disagree = (
            faithfulness_source == "ragas"
            and float(row["heuristic_faithfulness"]) >= 0.8
            and float(faithfulness) < 0.5
            and row["behavior_pass"]
            and row["citation_pass"]
            and (
                row.get("keyword_pass") is True
                or row["answer"].get("validate_passed") is True
            )
        )
        passed = row["behavior_pass"] and (
            not row["should_answer"]
            or (float(faithfulness) >= 0.5 and float(answer_relevancy) >= 0.5)
            or (ragas_disagree and float(answer_relevancy) >= 0.5)
        )
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "tier": row["tier"],
                "category": row.get("category"),
                "should_answer": row["should_answer"],
                "behavior_pass": row["behavior_pass"],
                "citation_pass": row["citation_pass"],
                "keyword_pass": row["keyword_pass"],
                "faithfulness": round(float(faithfulness), 4),
                "answer_relevancy": round(float(answer_relevancy), 4),
                "heuristic_faithfulness": round(float(row["heuristic_faithfulness"]), 4),
                "heuristic_answer_relevancy": round(float(row["heuristic_answer_relevancy"]), 4),
                "faithfulness_source": faithfulness_source,
                "answer_relevancy_source": relevancy_source,
                "ragas_disagree_override": ragas_disagree,
                "refused": row["answer"].get("refused"),
                "validate_passed": row["answer"].get("validate_passed"),
                "citations": row["answer"].get("citations"),
                "answer_preview": (row["answer_text"][:160] + "…") if len(row["answer_text"]) > 160 else row["answer_text"],
                "passed": passed,
            }
        )

    answered = [r for r in results if r["should_answer"]]
    refused_items = [r for r in results if not r["should_answer"]]
    hard_items = [r for r in results if r.get("tier") == "hard"]
    ragas_rows = [r for r in results if r["faithfulness_source"] == "ragas"]

    def _avg(field: str, items: list[dict[str, Any]]) -> float | None:
        if not items:
            return None
        return round(sum(float(r[field]) for r in items) / len(items), 4)

    summary = {
        "question_count": len(results),
        "should_answer_count": len(answered),
        "irrelevant_count": len(refused_items),
        "hard_case_count": len(hard_items),
        "pass_count": sum(1 for r in results if r["passed"]),
        "pass_rate": round(sum(1 for r in results if r["passed"]) / len(results), 4) if results else 0.0,
        "hard_case_pass_rate": round(sum(1 for r in hard_items if r["passed"]) / len(hard_items), 4) if hard_items else None,
        "avg_faithfulness": _avg("faithfulness", answered),
        "avg_answer_relevancy": _avg("answer_relevancy", answered),
        "avg_ragas_faithfulness": _avg("faithfulness", [r for r in answered if r["faithfulness_source"] == "ragas"]),
        "avg_ragas_answer_relevancy": _avg("answer_relevancy", [r for r in answered if r["answer_relevancy_source"] == "ragas"]),
        "ragas_status": ragas_status,
        "ragas_scored_count": len(ragas_rows),
        "refusal_accuracy": round(sum(1 for r in refused_items if r["behavior_pass"]) / len(refused_items), 4)
        if refused_items
        else None,
    }
    return {"summary": summary, "results": results}


def run_pipeline_pytest() -> dict[str, Any]:
    import pytest

    test_file = Path(__file__).resolve().parent / "test_pipeline.py"
    exit_code = pytest.main(["-q", str(test_file), "--tb=line"])
    return {"exit_code": int(exit_code), "passed": exit_code == 0}


def render_html_report(report: dict[str, Any]) -> str:
    rag = report.get("rag") or {}
    summary = rag.get("summary") or {}
    rows = rag.get("results") or []
    pipeline = report.get("pipeline_pytest") or {}

    table_rows = []
    for row in rows:
        status = "PASS" if row.get("passed") else "FAIL"
        tier = row.get("tier") or "core"
        cat = row.get("category") or "—"
        f_src = row.get("faithfulness_source", "")
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('id', '')))}</td>"
            f"<td>{escape(tier)}</td>"
            f"<td>{escape(str(cat))}</td>"
            f"<td>{escape(str(row.get('question', '')))}</td>"
            f"<td>{row.get('faithfulness')} <small>({f_src})</small></td>"
            f"<td>{row.get('answer_relevancy')}</td>"
            f"<td>{'是' if row.get('behavior_pass') else '否'}</td>"
            f"<td>{'是' if row.get('keyword_pass') else ('—' if row.get('keyword_pass') is None else '否')}</td>"
            f"<td class=\"{status.lower()}\">{status}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Disease Info Eval Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    .meta {{ color: #6b7280; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 0.5rem 0.75rem; text-align: left; vertical-align: top; }}
    th {{ background: #f9fafb; }}
    .pass {{ color: #047857; font-weight: 600; }}
    .fail {{ color: #b91c1c; font-weight: 600; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
    .card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; }}
    .card strong {{ display: block; font-size: 1.4rem; margin-top: 0.25rem; }}
  </style>
</head>
<body>
  <h1>Disease Info Eval Harness</h1>
  <p class="meta">run_id={escape(str(report.get('run_id', '')))} · generated_at={escape(str(report.get('generated_at', '')))}</p>

  <div class="cards">
    <div class="card">Pass rate<strong>{summary.get('pass_rate')}</strong></div>
    <div class="card">Avg faithfulness<strong>{summary.get('avg_faithfulness')}</strong></div>
    <div class="card">Avg answer relevancy<strong>{summary.get('avg_answer_relevancy')}</strong></div>
    <div class="card">Refusal accuracy<strong>{summary.get('refusal_accuracy')}</strong></div>
    <div class="card">Hard pass rate<strong>{summary.get('hard_case_pass_rate')}</strong></div>
    <div class="card">Ragas status<strong>{summary.get('ragas_status')}</strong></div>
    <div class="card">Ragas faithfulness<strong>{summary.get('avg_ragas_faithfulness')}</strong></div>
    <div class="card">Pipeline pytest<strong>{'PASS' if pipeline.get('passed') else 'FAIL'}</strong></div>
  </div>
  <p class="meta">方法论：disease_info_agent/docs/EVAL.md</p>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Tier</th><th>Category</th><th>Question</th>
        <th>Faithfulness</th><th>Answer relevancy</th>
        <th>Behavior</th><th>Keywords</th><th>Result</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def run_harness(
    *,
    run_id: str | None = None,
    limit: int | None = None,
    tier: str | None = None,
    use_ragas: bool = True,
    skip_pytest: bool = False,
) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    print(f"[eval] run_dir={run_dir}")

    rag_report = evaluate_rag_questions(run_dir, limit=limit, tier=tier, use_ragas=use_ragas)
    pipeline_report = {"skipped": True} if skip_pytest else run_pipeline_pytest()

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_dir.name,
        "eval_methodology": "disease_info_agent/docs/EVAL.md",
        "rag": rag_report,
        "pipeline_pytest": pipeline_report,
        "pipeline_thresholds": load_pipeline_thresholds(),
        "overall_passed": bool(rag_report["summary"]["pass_rate"] >= 0.7) and (
            skip_pytest or pipeline_report.get("passed", False)
        ),
    }

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _OUTPUT_DIR / "report.json"
    html_path = _OUTPUT_DIR / "report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html_report(report), encoding="utf-8")
    print(f"[eval] report.json -> {json_path}")
    print(f"[eval] report.html -> {html_path}")
    print(f"[eval] overall_passed={report['overall_passed']}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run disease-info eval harness (B4)")
    parser.add_argument("--run-id", default=None, help="指定 run 目录名，默认取最新")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条 golden 问题（调试）")
    parser.add_argument("--tier", choices=["core", "hard"], default=None, help="只跑 core 或 hard 子集")
    parser.add_argument("--no-ragas", action="store_true", help="跳过 Ragas，仅用启发式分数")
    parser.add_argument("--skip-pytest", action="store_true", help="跳过 pipeline pytest")
    args = parser.parse_args(argv)

    try:
        report = run_harness(
            run_id=args.run_id,
            limit=args.limit,
            tier=args.tier,
            use_ragas=not args.no_ragas,
            skip_pytest=args.skip_pytest,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] failed: {exc}", file=sys.stderr)
        return 1
    return 0 if report.get("overall_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
