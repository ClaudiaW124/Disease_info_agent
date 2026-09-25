import argparse
import asyncio
import json
from pathlib import Path

from legacy.legacy_pipeline import DiseaseInfoOrchestrator
from models.run_context import RunContext
from pipeline.aggregator_agent import AggregatorAgent
from pipeline.extractor_agent import ExtractorAgent
from pipeline.fetcher import Fetcher
from pipeline.keyword_extract import TARGET_DISEASES, KeywordExtractor
from pipeline.reporter import PipelineReporter
from pipeline.validator import FactValidator


def read_urls(urls_file_path: str, project_root: Path) -> list[str]:

    path = Path(urls_file_path)

    if not path.is_absolute():

        candidates = [

            project_root / path,

            Path.cwd() / path,

            project_root / path.name,

        ]

        for candidate in candidates:

            if candidate.exists():

                path = candidate

                break

        else:

            path = project_root / path



    if not path.exists():

        raise FileNotFoundError(f"URL 列表文件不存在: {path}")



    urls = [

        line.strip()

        for line in path.read_text(encoding="utf-8").splitlines()

        if line.strip() and not line.strip().startswith("#")

    ]

    if not urls:

        raise ValueError(f"URL 列表为空: {path}")

    return urls





def add_common_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--urls", default="urls.txt", help="URL 列表文件")

    parser.add_argument("--run-id", default=None, help="指定 run_id")

    parser.add_argument("--verbose", action="store_true", help="打印完整 JSON 结果")





def get_run_context(args: argparse.Namespace, project_root: Path, *, require_existing: bool = False) -> RunContext:

    if require_existing and not args.run_id:

        raise ValueError("该命令必须指定 --run-id")

    run_context = RunContext(project_root, run_id=args.run_id)

    if require_existing and not run_context.run_dir.exists():

        raise FileNotFoundError(f"run 不存在: {run_context.run_dir}")

    return run_context





async def run_legacy(args: argparse.Namespace, project_root: Path) -> dict:

    orchestrator = DiseaseInfoOrchestrator(

        verbose=args.verbose,

        use_llm_helpers=args.use_llm,

        run_id=args.run_id,

    )

    return await orchestrator.run_full_pipeline(args.urls)





async def run_fetch(args: argparse.Namespace, project_root: Path) -> dict:

    urls = read_urls(args.urls, project_root)

    run_context = RunContext(project_root, run_id=args.run_id)

    run_context.start_manifest(

        urls_file=args.urls,

        urls=urls,

        target_diseases=TARGET_DISEASES,

        pipeline_version="v3-fetch",

    )



    print("\n" + "=" * 50)

    print("  Python Fetcher（P1）")

    print(f"  run_id: {run_context.run_id}")

    print(f"  输出目录: {run_context.run_dir}")

    print("=" * 50 + "\n")



    fetcher = Fetcher(run_context=run_context)

    pages = await fetcher.fetch_all(urls, concurrency=args.concurrency)



    success_count = sum(1 for page in pages if len(page.text) >= 100)

    run_context.finish_manifest(

        stats={

            "url_count": len(urls),

            "fetched_count": len(pages),

            "success_count": success_count,

            "raw_pages_dir": str(run_context.raw_pages_dir),

        }

    )



    print(f"\n完成: {success_count}/{len(urls)} 个 URL 提取到 ≥100 字符正文")

    print(f"raw_pages: {run_context.raw_pages_dir}")

    print(f"run_manifest: {run_context.manifest_path}")



    return {

        "run_id": run_context.run_id,

        "output_dir": str(run_context.run_dir),

        "raw_pages_dir": str(run_context.raw_pages_dir),

        "success_count": success_count,

        "url_count": len(urls),

    }





async def run_extract(args: argparse.Namespace, project_root: Path) -> dict:

    run_context = get_run_context(args, project_root, require_existing=True)

    if not run_context.raw_pages_dir.exists():

        raise FileNotFoundError(f"run 不存在或缺少 raw_pages: {run_context.run_dir}")



    raw_count = len(list(run_context.raw_pages_dir.glob("*.json")))

    print("\n" + "=" * 50)

    print("  Extractor Agent（P2）")

    print(f"  run_id: {run_context.run_id}")

    print(f"  raw_pages: {raw_count} 个文件")

    print("=" * 50 + "\n")



    extractor = ExtractorAgent(run_context=run_context)

    await extractor.load_agent()

    fact_groups = await extractor.extract_from_run(run_context)

    fact_count = sum(len(group) for group in fact_groups)

    _update_manifest_stats(

        run_context,

        {

            "extractor_fact_count": fact_count,

            "facts_dir": str(run_context.facts_dir),

            "extractor": "agent",

        },

        pipeline_suffix="+extract",

    )



    print(f"\n完成: Agent 抽取 {fact_count} 条 facts")

    print(f"facts: {run_context.facts_dir}")



    return {

        "run_id": run_context.run_id,

        "fact_count": fact_count,

        "facts_dir": str(run_context.facts_dir),

    }





async def run_validate(args: argparse.Namespace, project_root: Path) -> dict:

    run_context = get_run_context(args, project_root, require_existing=True)

    if not run_context.facts_dir.exists():

        raise FileNotFoundError(f"run 不存在或缺少 facts: {run_context.facts_dir}")



    print("\n" + "=" * 50)

    print("  Fact Validator（P3-1）")

    print(f"  run_id: {run_context.run_id}")

    print("=" * 50 + "\n")



    validator = FactValidator(run_context)

    _, stats = validator.validate_run()

    _update_manifest_stats(

        run_context,

        {

            "validated_facts_dir": str(run_context.validated_facts_dir),

            "validation_stats": stats,

        },

        pipeline_suffix="+validate",

    )



    print(

        f"\n完成: {stats['input_count']} -> {stats['output_count']} 条 "

        f"(短文本 {stats['removed_short']}, JSON碎片 {stats['removed_json_fragment']}, "

        f"导航 {stats['removed_navigation']}, 疾病不匹配 {stats['removed_disease_mismatch']}, "

        f"重复 {stats['removed_duplicate']})"

    )

    print(f"facts_validated: {run_context.validated_facts_dir}")



    return {"run_id": run_context.run_id, "validation_stats": stats}





async def run_aggregate(args: argparse.Namespace, project_root: Path) -> dict:

    run_context = get_run_context(args, project_root, require_existing=True)



    print("\n" + "=" * 50)

    print("  Aggregator Agent（P3-2）")

    print(f"  run_id: {run_context.run_id}")

    print("=" * 50 + "\n")



    aggregator = AggregatorAgent(run_context, use_llm=not args.no_llm)

    await aggregator.load_agent()

    payload = await aggregator.aggregate_run()

    disease_count = len([name for name, info in payload.get("diseases", {}).items() if info.get("count", 0) > 0])

    _update_manifest_stats(

        run_context,

        {

            "aggregated_data_path": str(run_context.aggregated_data_path),

            "aggregated_disease_count": disease_count,

        },

        pipeline_suffix="+aggregate",

    )



    print(f"\n完成: 整合 {disease_count} 种疾病")

    print(f"aggregated_data: {run_context.aggregated_data_path}")



    return {

        "run_id": run_context.run_id,

        "aggregated_disease_count": disease_count,

        "aggregated_data_path": str(run_context.aggregated_data_path),

    }





async def run_report(args: argparse.Namespace, project_root: Path) -> dict:

    run_context = get_run_context(args, project_root, require_existing=True)



    print("\n" + "=" * 50)

    print("  Reporter（P3-3）")

    print(f"  run_id: {run_context.run_id}")

    print("=" * 50 + "\n")



    reporter = PipelineReporter(run_context)

    result = reporter.generate_all()

    _update_manifest_stats(

        run_context,

        {

            "visualizations_dir": str(run_context.visualizations_dir),

            "visualization_files": result["generated_files"],

        },

        pipeline_suffix="+report",

    )



    print(f"\n完成: 生成 {len(result['generated_files'])} 个文件")

    print(f"visualizations: {result['output_dir']}")

    for filename in result["generated_files"]:

        print(f"  - {filename}")



    return {"run_id": run_context.run_id, **result}





async def run_ingest(args: argparse.Namespace, project_root: Path) -> dict:

    from rag.ingest import FactIngester

    run_context = get_run_context(args, project_root, require_existing=True)



    print("\n" + "=" * 50)

    print("  RAG Ingest（P4-2）")

    print(f"  run_id: {run_context.run_id}")

    print("=" * 50 + "\n")



    ingester = FactIngester(run_context, include_aggregated=not args.facts_only)

    manifest = ingester.ingest()

    _update_manifest_stats(

        run_context,

        {

            "knowledge_db_dir": str(run_context.knowledge_db_dir),

            "knowledge_db_document_count": manifest["document_count"],

        },

        pipeline_suffix="+ingest",

    )



    print(f"\n完成: 入库 {manifest['document_count']} 条文档")

    print(f"knowledge_db: {manifest['persist_directory']}")



    return manifest





async def run_ask(args: argparse.Namespace, project_root: Path) -> dict:

    from rag.ask import ask_question, print_result, run_demo

    run_context = get_run_context(args, project_root, require_existing=True)



    mode = "P4 linear" if getattr(args, "simple", False) else "LangGraph"

    print("\n" + "=" * 50)

    print(f"  RAG Ask（{mode}）")

    print(f"  run_id: {run_context.run_id}")

    print("=" * 50)



    if args.demo:

        results = run_demo(
            run_context.knowledge_db_dir,
            simple=getattr(args, "simple", False),
            verbose=getattr(args, "verbose", False),
        )

        return {"run_id": run_context.run_id, "results": results}



    if not args.question:

        raise ValueError("ask 命令需要提供 question，或使用 --demo")



    result = ask_question(
        args.question,
        run_context.knowledge_db_dir,
        top_k=args.top_k,
        simple=getattr(args, "simple", False),
    )

    print_result(result, verbose=getattr(args, "verbose", False))

    return {"run_id": run_context.run_id, **result}





async def run_keyword_pipeline(args: argparse.Namespace, project_root: Path) -> dict:

    urls = read_urls(args.urls, project_root)

    run_context = RunContext(project_root, run_id=args.run_id)

    run_context.start_manifest(

        urls_file=args.urls,

        urls=urls,

        target_diseases=TARGET_DISEASES,

        pipeline_version="v2-keyword",

    )



    print("\n" + "=" * 50)

    print("  Keyword Pipeline：fetch + keyword_extract")

    print(f"  run_id: {run_context.run_id}")

    print("=" * 50)



    fetcher = Fetcher(run_context=run_context)

    pages = await fetcher.fetch_all(urls, concurrency=args.concurrency)

    fetch_success = sum(1 for page in pages if len(page.text) >= 100)



    extractor = KeywordExtractor(run_context=run_context)

    fact_groups = extractor.extract_from_pages(pages)

    fact_count = sum(len(group) for group in fact_groups)



    run_context.finish_manifest(

        stats={

            "url_count": len(urls),

            "fetch_success_count": fetch_success,

            "fact_count": fact_count,

            "raw_pages_dir": str(run_context.raw_pages_dir),

            "facts_dir": str(run_context.facts_dir),

        }

    )



    return {

        "run_id": run_context.run_id,

        "fetch_success_count": fetch_success,

        "fact_count": fact_count,

    }





async def run_pipeline(args: argparse.Namespace, project_root: Path) -> dict:

    urls = read_urls(args.urls, project_root)

    run_context = RunContext(project_root, run_id=args.run_id)

    run_context.start_manifest(

        urls_file=args.urls,

        urls=urls,

        target_diseases=TARGET_DISEASES,

        pipeline_version="v4-pipeline",

    )



    print("\n" + "=" * 50)

    print("  一键 Pipeline：fetch → extract → validate → aggregate → report → ingest")

    print(f"  run_id: {run_context.run_id}")

    print(f"  输出目录: {run_context.run_dir}")

    print("=" * 50)



    print("\n[1/6] 抓取网页...")

    fetcher = Fetcher(run_context=run_context)

    pages = await fetcher.fetch_all(urls, concurrency=args.concurrency)

    fetch_success = sum(1 for page in pages if len(page.text) >= 100)

    print(f"  抓取成功: {fetch_success}/{len(urls)}")



    print("\n[2/6] Extractor Agent 抽取 facts...")

    extractor = ExtractorAgent(run_context=run_context)

    await extractor.load_agent()

    fact_groups = await extractor.extract_from_run(run_context)

    fact_count = sum(len(group) for group in fact_groups)

    print(f"  抽取 facts: {fact_count}")



    print("\n[3/6] Validator 清洗 facts...")

    validator = FactValidator(run_context)

    _, validation_stats = validator.validate_run()

    print(f"  校验后 facts: {validation_stats['output_count']}")



    print("\n[4/6] Aggregator Agent 整合数据...")

    aggregator = AggregatorAgent(run_context, use_llm=not args.no_llm)

    await aggregator.load_agent()

    aggregated = await aggregator.aggregate_run()

    disease_count = len([name for name, info in aggregated.get("diseases", {}).items() if info.get("count", 0) > 0])

    print(f"  整合疾病: {disease_count}")



    print("\n[5/6] Reporter 生成图表和报告...")

    reporter = PipelineReporter(run_context)

    report_result = reporter.generate_all(aggregated)



    ingest_manifest: dict | None = None

    if not args.skip_rag:

        from rag.ingest import FactIngester

        print("\n[6/6] RAG Ingest 入库 knowledge_db...")

        ingester = FactIngester(run_context, include_aggregated=not args.facts_only)

        ingest_manifest = ingester.ingest()

        print(f"  入库文档: {ingest_manifest['document_count']}")

    else:

        print("\n[6/6] RAG Ingest 已跳过（--skip-rag）")



    manifest_stats = {

        "url_count": len(urls),

        "fetch_success_count": fetch_success,

        "extractor_fact_count": fact_count,

        "validation_stats": validation_stats,

        "validated_fact_count": validation_stats["output_count"],

        "aggregated_disease_count": disease_count,

        "raw_pages_dir": str(run_context.raw_pages_dir),

        "facts_dir": str(run_context.facts_dir),

        "validated_facts_dir": str(run_context.validated_facts_dir),

        "aggregated_data_path": str(run_context.aggregated_data_path),

        "visualizations_dir": str(run_context.visualizations_dir),

        "visualization_files": report_result["generated_files"],

    }

    if ingest_manifest:

        manifest_stats["knowledge_db_dir"] = str(run_context.knowledge_db_dir)

        manifest_stats["knowledge_db_document_count"] = ingest_manifest["document_count"]



    run_context.finish_manifest(stats=manifest_stats)



    print(f"\n全部完成 run_id={run_context.run_id}")

    print(f"  facts_validated: {run_context.validated_facts_dir}")

    print(f"  aggregated_data: {run_context.aggregated_data_path}")

    print(f"  visualizations: {run_context.visualizations_dir}")

    if ingest_manifest:

        print(f"  knowledge_db: {run_context.knowledge_db_dir}")



    return {

        "run_id": run_context.run_id,

        "output_dir": str(run_context.run_dir),

        "fetch_success_count": fetch_success,

        "extractor_fact_count": fact_count,

        "validated_fact_count": validation_stats["output_count"],

        "aggregated_disease_count": disease_count,

        "visualization_files": report_result["generated_files"],

        "knowledge_db_dir": str(run_context.knowledge_db_dir) if ingest_manifest else None,

        "knowledge_db_document_count": ingest_manifest["document_count"] if ingest_manifest else None,

    }





def _update_manifest_stats(run_context: RunContext, stats: dict, pipeline_suffix: str = "") -> None:

    if not run_context.manifest_path.exists():

        return

    manifest = json.loads(run_context.manifest_path.read_text(encoding="utf-8"))

    manifest_stats = manifest.get("stats", {})

    manifest_stats.update(stats)

    manifest["stats"] = manifest_stats

    if pipeline_suffix:

        manifest["pipeline_version"] = manifest.get("pipeline_version", "v3") + pipeline_suffix

    run_context.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )





async def run_chat(args: argparse.Namespace, project_root: Path) -> dict:
    from agent.coordinator import DiseaseInfoCoordinator

    no_plan = getattr(args, "no_plan", False)
    if args.message:
        async with DiseaseInfoCoordinator(project_root, enable_planner=not no_plan) as coordinator:
            result = await coordinator.chat(args.message, no_plan=no_plan)
            trace = result.trace or {}
            if trace.get("steps"):
                print(f"[执行轨迹 trace_id={result.trace_id}]")
                for step in trace["steps"]:
                    mark = "OK" if step.get("status") == "ok" else "ERR" if step.get("status") == "error" else "…"
                    print(f"  {step['id']}. [{mark}] {step['tool']} — {step.get('summary') or step.get('description')}")
                    for rag in step.get("rag_steps") or []:
                        print(f"       -> RAG: {rag.get('label')} | {rag.get('raw')}")
                print()
            elif trace.get("mode") == "direct":
                print(f"[执行轨迹] Planner → Orchestrator → RAG → Validate (trace_id={result.trace_id})\n")
            print(result.reply)
            return result.model_dump()
    from agent.coordinator import run_chat_loop

    await run_chat_loop(project_root, no_plan=no_plan)
    return {"status": "chat_ended"}


async def main() -> None:

    project_root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="疾病信息多智能体系统")

    subparsers = parser.add_subparsers(dest="command")



    pipeline_parser = subparsers.add_parser(

        "pipeline",

        help="一键执行 fetch → extract → validate → aggregate → report → ingest",

    )

    add_common_args(pipeline_parser)

    pipeline_parser.add_argument("--concurrency", type=int, default=5, help="抓取并发数")

    pipeline_parser.add_argument("--no-llm", action="store_true", help="聚合阶段不使用 LLM")

    pipeline_parser.add_argument("--skip-rag", action="store_true", help="跳过 RAG ingest 入库")

    pipeline_parser.add_argument(
        "--facts-only",
        action="store_true",
        help="RAG ingest 仅入库 facts，不含 aggregated 摘要",
    )



    fetch_parser = subparsers.add_parser("fetch", help="仅 Python 采集")

    add_common_args(fetch_parser)

    fetch_parser.add_argument("--concurrency", type=int, default=5, help="抓取并发数")



    extract_parser = subparsers.add_parser("extract", help="Extractor Agent：从 raw_pages 抽取 facts")

    extract_parser.add_argument("--run-id", required=True, help="已有 raw_pages 的 run_id")

    extract_parser.add_argument("--verbose", action="store_true", help="打印完整 JSON 结果")



    validate_parser = subparsers.add_parser("validate", help="Validator：清洗 facts")

    validate_parser.add_argument("--run-id", required=True, help="已有 facts 的 run_id")



    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregator Agent：生成 aggregated_data.json")

    aggregate_parser.add_argument("--run-id", required=True, help="已有 facts_validated 的 run_id")

    aggregate_parser.add_argument("--no-llm", action="store_true", help="不使用 LLM，仅 Python 汇总")



    report_parser = subparsers.add_parser("report", help="Reporter：生成图表和 Markdown 报告")

    report_parser.add_argument("--run-id", required=True, help="已有 aggregated_data.json 的 run_id")



    ingest_parser = subparsers.add_parser("ingest", help="RAG Ingest：facts → knowledge_db")

    ingest_parser.add_argument("--run-id", required=True, help="已有 facts_validated 的 run_id")

    ingest_parser.add_argument("--facts-only", action="store_true", help="仅入库 facts，不含 aggregated 摘要")



    ask_parser = subparsers.add_parser("ask", help="RAG Ask：检索 + 生成 + citation")

    ask_parser.add_argument("--run-id", required=True, help="已 ingest 的 run_id")

    ask_parser.add_argument("question", nargs="?", default=None, help="用户问题")

    ask_parser.add_argument("--demo", action="store_true", help="运行 5 个验收问题")

    ask_parser.add_argument("--top-k", type=int, default=5, help="检索条数")

    ask_parser.add_argument("--simple", action="store_true", help="使用 P4 线性 RAG，不走 LangGraph")

    ask_parser.add_argument("--verbose", action="store_true", help="打印 LangGraph 节点流程")

    chat_parser = subparsers.add_parser("chat", help="Coordinator 对话（Planner + Orchestrator，Phase 2）")
    chat_parser.add_argument("message", nargs="?", default=None, help="单轮问题；省略则进入交互模式")
    chat_parser.add_argument("--no-plan", action="store_true", help="跳过 Planner，仅用 Orchestrator 直连")
    chat_parser.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)

    keyword_parser = subparsers.add_parser("keyword", help="旧路径：fetch + keyword_extract")

    add_common_args(keyword_parser)

    keyword_parser.add_argument("--concurrency", type=int, default=5, help="抓取并发数")



    legacy_parser = subparsers.add_parser("legacy", help="旧 4 Agent 流水线（仅供对比）")

    add_common_args(legacy_parser)

    legacy_parser.add_argument("--use-llm", action="store_true", help="整合/可视化也调用 LLM")



    parser.add_argument("--urls", default="urls.txt", help=argparse.SUPPRESS)

    parser.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--use-llm", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--concurrency", type=int, default=5, help=argparse.SUPPRESS)

    parser.add_argument("--no-llm", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--skip-rag", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--facts-only", action="store_true", help=argparse.SUPPRESS)



    args = parser.parse_args()



    if args.command == "fetch":

        result = await run_fetch(args, project_root)

    elif args.command == "extract":

        result = await run_extract(args, project_root)

    elif args.command == "validate":

        result = await run_validate(args, project_root)

    elif args.command == "aggregate":

        result = await run_aggregate(args, project_root)

    elif args.command == "report":

        result = await run_report(args, project_root)

    elif args.command == "ingest":

        result = await run_ingest(args, project_root)

    elif args.command == "ask":

        result = await run_ask(args, project_root)

    elif args.command == "chat":

        result = await run_chat(args, project_root)

    elif args.command == "keyword":

        result = await run_keyword_pipeline(args, project_root)

    elif args.command == "legacy":

        result = await run_legacy(args, project_root)

    elif args.command == "pipeline":

        result = await run_pipeline(args, project_root)

    else:

        result = await run_pipeline(args, project_root)



    if args.verbose:

        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))





if __name__ == "__main__":

    asyncio.run(main())


