# Disease Info Agent

**疾病信息多智能体系统** — 基于 [Youtu-Agent](https://github.com/TencentCloudADP/Youtu-Agent)（`utu/`）构建的课程/个人项目仓库。

从 WHO / NHS / Wikipedia 等来源采集 **登革病毒、猩红热、裂谷热、流感** 四类疾病信息，经 Pipeline 清洗入库；**Plan-and-Execute Coordinator** + **LangGraph RAG** 提供带引用的问答；含 Web 演示、MCP、Ragas 评测与执行轨迹可视化。

> **完整文档（安装、命令、架构、Eval）** → [`disease_info_agent/README.md`](disease_info_agent/README.md)

---

## 快速开始

```powershell
git clone https://github.com/ClaudiaW124/Disease_info_agent.git
cd Disease_info_agent
uv sync --group dev
copy .env.example .env   # 填写 UTU_LLM_* 与 RAG_EMBEDDING_MODEL
```

仓库内已包含示例 `output/runs/` 知识库，配置好 `.env` 后可直接：

```powershell
uv run python disease_info_agent/main.py chat "登革热怎么预防？"
uv run python -m disease_info_agent.api   # http://127.0.0.1:8000/
```

从零重建知识库：

```powershell
cd disease_info_agent
uv run python main.py pipeline --urls urls.txt
```

---

## 本仓库结构

| 路径 | 说明 |
|------|------|
| **`disease_info_agent/`** | 本项目全部代码（Agent、RAG、API、Eval） |
| **`configs/agents/disease_info/`** | Orchestrator / Planner / Synthesizer 等 YAML |
| **`utu/`** | Youtu-Agent 框架（SimpleAgent、配置加载） |
| **`README_ZH.md`** | 上游 Youtu-Agent 中文说明（fork 保留） |

---

## 上游框架

本仓库在 Tencent **Youtu-Agent** 之上扩展，未修改其作为通用 Agent 框架的定位。上游文档与示例见 [README_ZH.md](README_ZH.md) 与 [官方文档](https://tencentcloudadp.github.io/youtu-agent/)。

---

## License

与上游一致，见 [LICENSE](LICENSE)。
