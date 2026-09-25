"""Generate charts and Markdown report from aggregated_data.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from models.run_context import RunContext
from pipeline.keyword_extract import TARGET_DISEASES
from wordcloud import WordCloud

matplotlib.use("Agg")

VISUALIZATION_FILES = [
    "disease_wordcloud.png",
    "source_distribution.png",
    "disease_distribution.png",
    "comprehensive_report.md",
]


class PipelineReporter:
    """P3-3 — matplotlib/wordcloud report generation."""

    def __init__(self, run_context: RunContext) -> None:
        self.run_context = run_context
        self.output_dir = run_context.visualizations_dir
        self.chinese_font_path = self._find_chinese_font()

    def _find_chinese_font(self) -> str | None:
        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
        for path in candidates:
            if Path(path).exists():
                return path
        return None

    def _font_kwargs(self) -> dict[str, Any]:
        if not self.chinese_font_path:
            return {}
        return {"fontproperties": fm.FontProperties(fname=self.chinese_font_path)}

    def load_aggregated_data(self) -> dict[str, Any]:
        path = self.run_context.aggregated_data_path
        if not path.exists():
            raise FileNotFoundError(f"未找到 aggregated_data.json: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _shorten_label(self, text: str, max_len: int = 28) -> str:
        text = text.strip()
        if len(text) <= max_len:
            return text
        parsed = urlparse(text)
        if parsed.netloc:
            return parsed.netloc
        return text[: max_len - 3] + "..."

    def generate_disease_distribution_chart(self, data: dict[str, Any]) -> None:
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
        if font_kwargs:
            plt.xticks(rotation=30, ha="right", fontproperties=font_kwargs["fontproperties"])
        else:
            plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "disease_distribution.png", dpi=150)
        plt.close()

    def generate_source_pie_chart(self, data: dict[str, Any]) -> None:
        diseases = data.get("diseases", {})
        source_counts: dict[str, int] = {}
        for info in diseases.values():
            for source in info.get("sources", []):
                label = self._shorten_label(str(source))
                source_counts[label] = source_counts.get(label, 0) + 1

        if not source_counts:
            sources = data.get("sources", [])
            source_counts = {self._shorten_label(source): 1 for source in sources}
        if not source_counts:
            return

        font_kwargs = self._font_kwargs()
        labels = list(source_counts.keys())
        values = list(source_counts.values())

        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
        plt.title("数据来源分布", **font_kwargs)
        if font_kwargs:
            font_prop = font_kwargs["fontproperties"]
            for text in texts + autotexts:
                text.set_fontproperties(font_prop)
        plt.tight_layout()
        plt.savefig(self.output_dir / "source_distribution.png", dpi=150)
        plt.close()

    def generate_wordcloud_chart(self, data: dict[str, Any]) -> None:
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
        wc.to_file(str(self.output_dir / "disease_wordcloud.png"))

    def generate_markdown_report(self, data: dict[str, Any]) -> None:
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
            for disease in TARGET_DISEASES:
                info = diseases.get(disease)
                if not info:
                    continue
                lines.extend(
                    [
                        f"## {disease}",
                        "",
                        f"- 条目数: {info.get('count', 0)}",
                        "",
                        "### 摘要",
                        "",
                        info.get("summary", "暂无摘要"),
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
                    source = item.get("source_url") or item.get("source", "未知来源")
                    field = item.get("field", "general")
                    content = item.get("content", "")
                    lines.append(f"- [{field}] [{source}] {content[:300]}")
                lines.append("")

        lines.extend(["## 数据来源", ""])
        for source in data.get("sources", []):
            lines.append(f"- {source}")

        report_path = self.output_dir / "comprehensive_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def generate_all(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = data or self.load_aggregated_data()
        self.generate_disease_distribution_chart(payload)
        self.generate_source_pie_chart(payload)
        self.generate_wordcloud_chart(payload)
        self.generate_markdown_report(payload)
        generated = [name for name in VISUALIZATION_FILES if (self.output_dir / name).exists()]
        return {"output_dir": str(self.output_dir), "generated_files": generated}
