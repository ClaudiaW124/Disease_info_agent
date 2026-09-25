import asyncio
import json
import re
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from models.run_context import RunContext
from wordcloud import WordCloud

from utu.agents import SimpleAgent

matplotlib.use("Agg")

TARGET_DISEASES = ["登革病毒", "猩红热", "裂谷热", "流感"]
DISEASE_KEYWORDS: dict[str, list[str]] = {
    "登革病毒": ["登革", "dengue"],
    "猩红热": ["猩红热", "scarlet fever", "scarlet"],
    "裂谷热": ["裂谷热", "rift valley"],
    "流感": ["流感", "influenza", "flu"],
}
VISUALIZATION_FILES = [
    "disease_wordcloud.png",
    "source_distribution.png",
    "disease_distribution.png",
    "comprehensive_report.md",
]


class DiseaseInfoOrchestrator:
    def __init__(
        self,
        verbose: bool = False,
        use_llm_helpers: bool = False,
        run_id: str | None = None,
    ):
        self.verbose = verbose
        # 仅控制「整合/可视化」是否额外调用 LLM；抓取始终先用 Scraper 智能体（与技术报告一致）
        self.use_llm_helpers = use_llm_helpers
        self.project_root = Path(__file__).parent
        self.run_context = RunContext(self.project_root, run_id=run_id)
        self.run_id = self.run_context.run_id
        self.output_dir = self.run_context.run_dir
        self.scraper_results_dir = self.run_context.scraper_results_dir
        self.visualizations_dir = self.run_context.visualizations_dir

        self.planner_agent: SimpleAgent | None = None
        self.scraper_agent: SimpleAgent | None = None
        self.aggregator_agent: SimpleAgent | None = None
        self.visualizer_agent: SimpleAgent | None = None
        self.chinese_font_path = self._find_chinese_font()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _find_chinese_font(self) -> str | None:
        candidates = [
            r"C:\\Windows\\Fonts\\msyh.ttc",
            r"C:\\Windows\\Fonts\\msyhbd.ttc",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\simsun.ttc",
        ]
        for path in candidates:
            if Path(path).exists():
                return path
        return None

    def _font_kwargs(self) -> dict[str, Any]:
        if not self.chinese_font_path:
            return {}
        font_prop = fm.FontProperties(fname=self.chinese_font_path)
        return {"fontproperties": font_prop}

    async def load_agents(self) -> None:
        # youtu-agent 的 ConfigLoader 只认 configs/agents/ 下的相对路径，不能传绝对路径
        self.planner_agent = SimpleAgent(config="disease_info/planner")
        self.scraper_agent = SimpleAgent(config="disease_info/scraper")
        self.aggregator_agent = SimpleAgent(config="disease_info/aggregator")
        self.visualizer_agent = SimpleAgent(config="disease_info/visualizer")

    def _resolve_input_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if path.is_absolute():
            return path

        candidates = [
            self.project_root / path,
            Path.cwd() / path,
            self.project_root / path.name,
        ]
        if path.parts and path.parts[0] == self.project_root.name:
            candidates.append(self.project_root.parent / path)
            candidates.append(self.project_root / Path(*path.parts[1:]))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return self.project_root / path

    def _read_urls(self, urls_file_path: str) -> list[str]:
        urls_path = self._resolve_input_path(urls_file_path)
        if not urls_path.exists():
            raise FileNotFoundError(f"URL 列表文件不存在: {urls_path}")

        urls = [
            line.strip()
            for line in urls_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not urls:
            raise ValueError(f"URL 列表为空: {urls_path}")
        return urls

    def _build_tasks_from_urls(self, urls: list[str]) -> list[str]:
        return [
            f"任务：抓取URL [{url}]，并分析其中与[{', '.join(TARGET_DISEASES)}]相关的信息。"
            for url in urls
        ]

    def _parse_planner_tasks(self, planner_output: str, urls: list[str]) -> list[str]:
        tasks: list[str] = []
        for line in planner_output.splitlines():
            line = line.strip()
            if not line:
                continue
            if "任务" in line or "http" in line.lower():
                tasks.append(line)

        if len(tasks) == len(urls):
            # 用原始 URL 覆盖 Planner 可能改坏的链接
            fixed_tasks = []
            for _index, (task, url) in enumerate(zip(tasks, urls, strict=False), 1):
                if url not in task:
                    fixed_tasks.append(self._build_tasks_from_urls([url])[0])
                else:
                    fixed_tasks.append(task)
            return fixed_tasks

        return self._build_tasks_from_urls(urls)

    async def run_planner_agent(self, urls_file_path: str) -> tuple[list[str], list[str]]:
        if not self.planner_agent:
            raise RuntimeError("Planner agent 未加载，请先调用 load_agents()")

        urls = self._read_urls(urls_file_path)
        # 始终保留原始 URL 列表，供 Scraper 使用
        tasks = self._build_tasks_from_urls(urls)

        if self.use_llm_helpers:
            prompt = "以下是 URL 列表，请为每个 URL 生成一条抓取任务指令：\n" + "\n".join(urls)
            recorder = await self.planner_agent.run(prompt)
            tasks = self._parse_planner_tasks(recorder.final_output, urls)
            self._log(f"Planner 原始输出:\n{recorder.final_output}")

        return tasks, urls

    @staticmethod
    def _guess_diseases_from_url(url: str) -> list[str]:
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

    def _html_to_text(self, html: str) -> str:
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _default_headers(self) -> dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    async def _scrape_url_with_requests(self, url: str) -> str:
        import requests

        def fetch() -> str:
            response = requests.get(
                url,
                headers=self._default_headers(),
                timeout=30,
                allow_redirects=True,
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text

        html = await asyncio.to_thread(fetch)
        return self._html_to_text(html)

    async def _scrape_url_with_http(self, url: str) -> str:
        headers = self._default_headers()
        timeout = aiohttp.ClientTimeout(total=30)
        # WHO 等站点响应头很大，默认 8190 字节会报错
        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            max_line_size=65536,
            max_field_size=65536,
        ) as session:
            async with session.get(url, allow_redirects=True) as response:
                response.raise_for_status()
                html = await response.text(errors="ignore")
                return self._html_to_text(html)

    async def _scrape_url_with_jina(self, url: str) -> dict[str, Any]:
        from utu.tools.search.jina_crawl import JinaCrawl

        crawler = JinaCrawl()
        page_text = await crawler.crawl(url)
        return self._extract_disease_info_from_text(url, page_text, source="jina")

    async def _scrape_url_smart(self, url: str) -> dict[str, Any]:
        errors: list[str] = []

        print("  尝试直连网页抓取 (requests)...")
        try:
            page_text = await self._scrape_url_with_requests(url)
            data = self._extract_disease_info_from_text(url, page_text, source="requests")
            if data.get("extracted_content"):
                return data
            errors.append("requests 未提取到有效内容")
        except Exception as exc:
            errors.append(f"requests: {exc}")
            print(f"  requests 失败: {exc}")

        print("  尝试直连网页抓取 (aiohttp)...")
        try:
            page_text = await self._scrape_url_with_http(url)
            data = self._extract_disease_info_from_text(url, page_text, source="aiohttp")
            if data.get("extracted_content"):
                return data
            errors.append("aiohttp 未提取到有效内容")
        except Exception as exc:
            errors.append(f"aiohttp: {exc}")
            print(f"  aiohttp 失败: {exc}")

        print("  尝试 Jina 抓取...")
        try:
            data = await self._scrape_url_with_jina(url)
            if data.get("extracted_content"):
                return data
            errors.append("Jina 返回内容为空")
        except Exception as exc:
            errors.append(f"Jina: {exc}")
            print(f"  Jina 失败: {exc}")

        print("  三种方式均失败，跳过该 URL")
        return {
            "url": url,
            "timestamp": datetime.now(UTC).isoformat(),
            "diseases_found": [],
            "extracted_content": [],
            "errors": errors,
            "source": "failed",
        }

    def _extract_disease_info_from_text(
        self, url: str, text: str, source: str = "text"
    ) -> dict[str, Any]:
        text = re.sub(r"\s+", " ", text).strip()
        chunks = re.split(r"(?<=[。！？.!?])\s+|\n{2,}", text)
        extracted: list[dict[str, str]] = []
        diseases_found: set[str] = set()

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue
            for disease, keywords in DISEASE_KEYWORDS.items():
                if any(keyword.lower() in chunk.lower() for keyword in keywords):
                    diseases_found.add(disease)
                    extracted.append(
                        {
                            "disease": disease,
                            "content": chunk[:500],
                            "context": f"来源: {url}",
                        }
                    )
                    break

        if not extracted and text:
            primary = self._guess_diseases_from_url(url)[0]
            diseases_found.add(primary)
            extracted.append(
                {
                    "disease": primary,
                    "content": text[:800],
                    "context": f"来源: {url}（页面摘要）",
                }
            )

        return {
            "url": url,
            "timestamp": datetime.now(UTC).isoformat(),
            "diseases_found": sorted(diseases_found),
            "extracted_content": extracted[:8],
            "source": source,
        }

    def _extract_from_raw_output(self, raw: str, url: str) -> list[dict[str, str]]:
        extracted: list[dict[str, str]] = []
        for disease, keywords in DISEASE_KEYWORDS.items():
            for keyword in keywords:
                pattern = rf"([^\n。]{{0,80}}{re.escape(keyword)}[^\n。]{{0,120}})"
                for match in re.finditer(pattern, raw, re.IGNORECASE):
                    sentence = match.group(1).strip()
                    if len(sentence) >= 10:
                        extracted.append(
                            {
                                "disease": disease,
                                "content": sentence,
                                "context": "从 LLM 原始输出中提取",
                            }
                        )
                        break
        return extracted[:5]

    @staticmethod
    def _extract_url_from_task(task: str) -> str | None:
        match = re.search(r"https?://[^\s\]\)>\"']+", task)
        return match.group(0) if match else None

    @staticmethod
    def _extract_json_from_text(raw: str) -> dict[str, Any] | None:
        if not raw or not raw.strip():
            return None

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _normalize_scraper_result(self, data: dict[str, Any], task: str, index: int) -> dict[str, Any]:
        url = data.get("url") or self._extract_url_from_task(task) or f"unknown_{index:03d}"
        data["url"] = url
        data.setdefault("timestamp", datetime.now(UTC).isoformat())
        data.setdefault("diseases_found", [])
        data.setdefault("extracted_content", [])

        if not isinstance(data["diseases_found"], list):
            data["diseases_found"] = [str(data["diseases_found"])]
        if not isinstance(data["extracted_content"], list):
            data["extracted_content"] = []

        normalized_items = []
        for item in data["extracted_content"]:
            if not isinstance(item, dict):
                continue
            normalized_items.append(
                {
                    "disease": item.get("disease", "未知"),
                    "content": item.get("content", ""),
                    "context": item.get("context", ""),
                }
            )
        data["extracted_content"] = normalized_items
        return data

    async def run_scraper_agent(
        self, tasks: list[str], urls: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if not self.scraper_agent:
            raise RuntimeError("Scraper agent 未加载，请先调用 load_agents()")

        results: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, 1):
            source_url = urls[index - 1] if urls and index <= len(urls) else self._extract_url_from_task(task)
            self._log(f"Scraper 任务 {index}/{len(tasks)}: {source_url or task}")

            data: dict[str, Any]
            raw = ""

            # 与技术报告一致：始终先用 Scraper 智能体（search + web_qa）
            print("  使用 Scraper 智能体抓取...")
            try:
                #把 task 交给 Scraper 这个 SimpleAgent 执行，Agent 内部会调用搜索 / 网页工具，返回执行记录对象`recorder`
                recorder = await self.scraper_agent.run(task)
                #拿到 Agent 大模型输出原始字符串（通常期望输出 JSON）
                raw = recorder.final_output
                #工具函数，**从大模型输出文本里面抠 JSON 字符串**
                parsed = self._extract_json_from_text(raw)
                if parsed is None:
                    data = {
                        "url": source_url,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "diseases_found": [],
                        "extracted_content": [],
                        "raw_output": raw,
                        "task": task,
                    }
                else:
                    data = parsed
            except Exception as exc:
                print(f"  Scraper 智能体失败: {exc}")
                data = {
                    "url": source_url,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "diseases_found": [],
                    "extracted_content": [],
                    "scraper_error": str(exc),
                    "task": task,
                }

            data = self._normalize_scraper_result(data, task, index)
            if source_url:
                data["url"] = source_url

            if not data.get("extracted_content") and raw:
                fallback_items = self._extract_from_raw_output(raw, source_url or data["url"])
                if fallback_items:
                    data["extracted_content"] = fallback_items
                    data["diseases_found"] = sorted({item["disease"] for item in fallback_items})
                    data["fallback"] = "raw_output_regex"

            if not data.get("extracted_content") and source_url:
                print("  智能体未提取到有效内容，尝试网页直连备用抓取...")
                try:
                    fallback_data = await self._scrape_url_smart(source_url)
                    if fallback_data.get("extracted_content"):
                        data = fallback_data
                        data["url"] = source_url
                except Exception as exc:
                    data["fallback_error"] = str(exc)
                    self._log(f"备用抓取失败: {exc}")
            elif not data.get("extracted_content"):
                print(f"  警告: 未能从 {source_url} 提取内容")

            out_path = self.scraper_results_dir / f"result_{index:03d}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(data)

        return results

    def _process_scraper_results_directly(self) -> dict[str, Any]:
        by_disease: dict[str, list[dict[str, Any]]] = {}
        sources: list[str] = []
        seen_content: set[str] = set()

        for file_path in sorted(self.scraper_results_dir.glob("*.json")):
            data = json.loads(file_path.read_text(encoding="utf-8"))
            url = data.get("url") or file_path.stem
            sources.append(url)

            for item in data.get("extracted_content", []):
                if not isinstance(item, dict):
                    continue
                disease = str(item.get("disease", "未知")).strip() or "未知"
                content = str(item.get("content", "")).strip()
                context = str(item.get("context", "")).strip()
                dedupe_key = f"{disease}::{content}"
                if not content or dedupe_key in seen_content:
                    continue
                seen_content.add(dedupe_key)
                by_disease.setdefault(disease, []).append(
                    {
                        "disease": disease,
                        "content": content,
                        "context": context,
                        "source": url,
                    }
                )

        summary: dict[str, Any] = {}
        for disease, items in by_disease.items():
            summary[disease] = {
                "count": len(items),
                "key_points": [item["content"][:200] for item in items[:5]],
                "items": items,
            }

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "target_diseases": TARGET_DISEASES,
            "total_sources": len(sources),
            "sources": sources,
            "diseases": summary,
        }

    async def run_aggregator_agent(self) -> dict[str, Any]:
        if not self.aggregator_agent:
            raise RuntimeError("Aggregator agent 未加载，请先调用 load_agents()")

        agg_path = self.run_context.aggregated_data_path

        if self.use_llm_helpers:
            prompt = (
                f"请读取目录 {self.scraper_results_dir} 下的所有 JSON 抓取结果，"
                f"按疾病分类整合、去重并生成摘要，最后保存到 {agg_path}"
            )
            try:
                await self.aggregator_agent.run(prompt)
            except Exception as exc:
                self._log(f"Aggregator LLM 失败: {exc}")

        data = self._process_scraper_results_directly()
        if not data.get("diseases"):
            if agg_path.exists() and agg_path.stat().st_size >= 10:
                try:
                    cached = json.loads(agg_path.read_text(encoding="utf-8"))
                    if cached.get("diseases"):
                        return cached
                except json.JSONDecodeError:
                    pass
            print("  警告: 未提取到疾病内容，请检查 JINA_API_KEY 或网络连接")
        else:
            print("  使用 Python 直连整合数据")

        agg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    def _shorten_label(self, text: str, max_len: int = 28) -> str:
        text = text.strip()
        if len(text) <= max_len:
            return text
        parsed = urlparse(text)
        if parsed.netloc:
            return parsed.netloc
        return text[: max_len - 3] + "..."

    def _generate_disease_distribution_chart(self, data: dict[str, Any]) -> None:
        diseases = data.get("diseases", {})
        counts = {name: info.get("count", 0) for name, info in diseases.items() if info.get("count", 0) > 0}
        if not counts:
            return

        font_kwargs = self._font_kwargs()
        plt.figure(figsize=(9, 5))
        plt.bar(list(counts.keys()), list(counts.values()), color="#2563eb")
        plt.title("疾病信息分布统计", **font_kwargs)
        plt.xlabel("疾病类型", **font_kwargs)
        plt.ylabel("条目数", **font_kwargs)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "disease_distribution.png", dpi=150)
        plt.close()

    def _generate_source_pie_chart(self, data: dict[str, Any]) -> None:
        sources = data.get("sources", [])
        if not sources:
            return

        font_kwargs = self._font_kwargs()
        labels = [self._shorten_label(source) for source in sources]

        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            [1] * len(sources),
            labels=labels,
            autopct="%1.0f%%",
            startangle=90,
        )
        plt.title("数据来源分布", **font_kwargs)

        if font_kwargs:
            font_prop = font_kwargs["fontproperties"]
            for text in texts + autotexts:
                text.set_fontproperties(font_prop)

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "source_distribution.png", dpi=150)
        plt.close()

    def _generate_wordcloud_chart(self, data: dict[str, Any]) -> None:
        diseases = data.get("diseases", {})
        text = " ".join(
            item.get("content", "")
            for disease_info in diseases.values()
            for item in disease_info.get("items", [])
            if isinstance(item, dict)
        ).strip()

        if not text:
            return

        wc = WordCloud(
            width=900,
            height=450,
            background_color="white",
            max_words=120,
            collocations=False,
            font_path=self.chinese_font_path,
        )
        wc.generate(text)
        wc.to_file(str(self.visualizations_dir / "disease_wordcloud.png"))

    def _generate_markdown_report(self, data: dict[str, Any]) -> None:
        diseases = data.get("diseases", {})
        lines = [
            "# 疾病信息综合报告",
            "",
            f"- 生成时间: {data.get('generated_at', '')}",
            f"- 数据源数量: {data.get('total_sources', 0)}",
            f"- 目标疾病: {', '.join(data.get('target_diseases', TARGET_DISEASES))}",
            "",
            "## 数据概览",
            "",
        ]

        if not diseases:
            lines.append("暂无可用疾病数据。")
        else:
            for disease, info in diseases.items():
                lines.extend(
                    [
                        f"## {disease}",
                        "",
                        f"- 条目数: {info.get('count', 0)}",
                        "",
                        "### 关键点",
                        "",
                    ]
                )
                key_points = info.get("key_points", [])
                if key_points:
                    lines.extend(f"- {point}" for point in key_points)
                else:
                    lines.append("- 暂无关键点")

                lines.extend(["", "### 详细条目", ""])
                for item in info.get("items", [])[:10]:
                    if not isinstance(item, dict):
                        continue
                    source = item.get("source", "未知来源")
                    content = item.get("content", "")
                    lines.append(f"- [{source}] {content[:300]}")
                lines.append("")

        lines.extend(["## 数据来源", ""])
        for source in data.get("sources", []):
            lines.append(f"- {source}")

        report_path = self.visualizations_dir / "comprehensive_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _generate_visualizations_directly(self, data: dict[str, Any]) -> None:
        self._generate_disease_distribution_chart(data)
        self._generate_source_pie_chart(data)
        self._generate_wordcloud_chart(data)
        self._generate_markdown_report(data)

    async def run_visualizer_agent(self, aggregated_data: dict[str, Any]) -> dict[str, Any]:
        if not self.visualizer_agent:
            raise RuntimeError("Visualizer agent 未加载，请先调用 load_agents()")

        if self.use_llm_helpers:
            prompt = (
                f"请读取 {self.output_dir / 'aggregated_data.json'}，"
                f"生成词云图、来源分布饼图、疾病信息统计图和 Markdown 综合报告，"
                f"保存到 {self.visualizations_dir}"
            )
            try:
                await self.visualizer_agent.run(prompt)
            except Exception as exc:
                self._log(f"Visualizer LLM 失败: {exc}")

        missing_files = [
            filename
            for filename in VISUALIZATION_FILES
            if not (self.visualizations_dir / filename).exists()
        ]
        if missing_files or not self.use_llm_helpers:
            if missing_files:
                self._log(f"Visualizer 缺少文件 {missing_files}，启用 Python 直连可视化")
            print("  使用 Python 直连生成图表和报告")
            self._generate_visualizations_directly(aggregated_data)

        return {
            "output_dir": str(self.visualizations_dir),
            "files": VISUALIZATION_FILES,
            "generated_files": [
                filename
                for filename in VISUALIZATION_FILES
                if (self.visualizations_dir / filename).exists()
            ],
        }

    async def run_full_pipeline(self, urls_file_path: str) -> dict[str, Any]:
        print("\n" + "=" * 50)
        print("  疾病信息多智能体系统")
        print("  目标疾病: 登革病毒、猩红热、裂谷热、流感")
        print(f"  run_id: {self.run_id}")
        print("=" * 50)

        urls = self._read_urls(urls_file_path)
        self.run_context.start_manifest(
            urls_file=urls_file_path,
            urls=urls,
            target_diseases=TARGET_DISEASES,
            pipeline_version="v1-legacy",
        )
        print(f"  输出目录: {self.output_dir}")

        await self.load_agents()

        print("\n[1/4] 任务规划...")
        tasks, urls = await self.run_planner_agent(urls_file_path)
        print(f"  生成 {len(tasks)} 个任务")

        print("\n[2/4] 信息抓取...")
        scraper_results = await self.run_scraper_agent(tasks, urls=urls)
        extracted_count = sum(len(r.get("extracted_content", [])) for r in scraper_results)
        print(f"  完成 {len(scraper_results)} 个抓取，提取 {extracted_count} 条内容")

        print("\n[3/4] 数据整合...")
        aggregated_data = await self.run_aggregator_agent()
        print(f"  整合 {len(aggregated_data.get('diseases', {}))} 种疾病")

        print("\n[4/4] 数据可视化...")
        visualization_results = await self.run_visualizer_agent(aggregated_data)
        print(f"  输出目录: {visualization_results['output_dir']}")
        print(f"  已生成文件: {', '.join(visualization_results['generated_files'])}")

        extracted_count = sum(len(r.get("extracted_content", [])) for r in scraper_results)
        self.run_context.finish_manifest(
            stats={
                "task_count": len(tasks),
                "scraper_result_count": len(scraper_results),
                "extracted_item_count": extracted_count,
                "disease_count": len(aggregated_data.get("diseases", {})),
                "visualization_files": visualization_results.get("generated_files", []),
            }
        )
        print(f"\n  run_manifest: {self.run_context.manifest_path}")

        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.run_context.manifest_path),
            "tasks": tasks,
            "scraper_results": scraper_results,
            "aggregated_data": aggregated_data,
            "visualization_results": visualization_results,
        }
