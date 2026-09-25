# Disease Info Agent — 疾病信息多智能体系统

从 WHO / NHS / Wikipedia 等来源采集 **登革病毒、猩红热、裂谷热、流感** 四类疾病信息，经多阶段 Pipeline 清洗入库，并通过 **Orchestrator Agent（Phase 1）** 统一调度 RAG / Pipeline / Eval 工具，底层 **LangGraph RAG** 提供带 citation 的问答。支持 CLI、MCP（Cursor）、FastAPI + Web 演示页、Eval Harness 评测。

---

## 架构概览

```
urls.txt
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  Pipeline（main.py pipeline）                                │
│  fetch → extract → validate → aggregate → report → ingest   │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    facts_validated   aggregated_data   knowledge_db (Chroma)
                           │
                           ▼
              ┌────────────────────────┐
              │  LangGraph RAG (B2)    │
              │  retrieve → grade      │
              │  → rewrite/generate    │
              │  → validate → refuse   │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Coordinator (Phase 2) │  ← 统一入口
              └───────────┬────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   ┌─────────────────┐       ┌─────────────────────┐
   │ Task Planner    │       │ Orchestrator (P1)   │
   │ 复杂任务拆步骤   │       │ 单步意图 + 5 tools  │
   └────────┬────────┘       └──────────┬──────────┘
            ▼                           │
   ┌─────────────────┐                  │
   │ Plan Executor   │──────────────────┘
   │ 按序调用 tools  │
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ Synthesizer     │  汇总多步结果
   └─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   CLI chat          MCP query_disease    Web /chat
   main.py chat      Cursor 工具          http://127.0.0.1:8000
                     (+ 直接 /ask RAG)
```

| 模块 | 路径 | 说明 |
|------|------|------|
| **Coordinator** | `agent/coordinator.py` | Phase 2：Planner → Executor → Synthesizer |
| **Task Planner** | `agent/planner.py` | 复杂任务拆解为 JSON 计划 |
| **Orchestrator** | `agent/orchestrator.py` | Phase 1：单步意图路由 + 5 个 function tools |
| Pipeline | `pipeline/` | 抓取、抽取、校验、聚合、报告 |
| RAG | `rag/` | ingest、LangGraph 问答、线性 P4 |
| MCP | `mcp_server/server.py` | Cursor 集成：`query_disease`、`run_pipeline`、`get_eval_report` |
| Eval | `eval/run_harness.py` | Golden set + Ragas/启发式 + pytest |
| API + Web | `api/app.py` | FastAPI + 浏览器演示页（`/chat` 走 Orchestrator） |

---

## 安装

### 1. 环境

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)（推荐）

```powershell
cd "D:\My Documents\Desktop\Youtu-Agent"
uv sync --group dev
```

### 2. 配置 API Key

复制仓库根目录 `.env.example` → `.env`，至少配置：

```env
UTU_LLM_BASE_URL=https://api.siliconflow.cn/v1
UTU_LLM_API_KEY=你的密钥
UTU_LLM_MODEL=Qwen/Qwen3-8B

RAG_EMBEDDING_MODEL=BAAI/bge-m3
```

> RAG 默认复用 `UTU_LLM_*`；可用 `RAG_LLM_MODEL` 单独指定模型名。

---

## 六条常用命令

### 1. 一键 Pipeline

```powershell
cd disease_info_agent
uv run python main.py pipeline --urls urls.txt
```

输出目录：`output/runs/<run_id>/`（含 `run_manifest.json`、`knowledge_db/`）

### 2. RAG Ingest（单独入库）

```powershell
uv run python main.py ingest --run-id <run_id>
```

### 3. Coordinator 对话（Phase 2 推荐入口）

```powershell
# 仓库根目录
uv run python disease_info_agent/main.py chat "登革热怎么预防？"
uv run python disease_info_agent/main.py chat "先告诉我知识库有多少文档，再说明登革热怎么预防"
uv run python disease_info_agent/main.py chat --no-plan "登革热怎么预防？"   # 跳过 Planner
uv run python disease_info_agent/main.py chat                    # 交互多轮（/noplan 切换）
uv run python -m disease_info_agent.agent
```

**简单问题** → Planner 返回 `direct` → **Orchestrator** 单步调工具  
**复杂问题**（先…再…、对比+eval 等）→ Planner 拆步骤 → **Executor** 执行 → **Synthesizer** 汇总

| 用户意图 | 路径 |
|----------|------|
| 单个疾病问答 | direct → `query_disease` |
| 多步组合任务 | plan → 多 tool + Synthesizer |
| 仅要 Phase 1 行为 | `--no-plan` 或 Web `no_plan: true` |

配置：`configs/agents/disease_info/task_planner.yaml` · `orchestrator.yaml` · `synthesizer.yaml`

### 4. 直接 RAG 问答（绕过 Orchestrator）

```powershell
uv run python main.py ask --run-id <run_id> "登革热怎么预防？"
uv run python main.py ask --run-id <run_id> --verbose   # 看 LangGraph 步骤
```

### 5. Eval Harness

```powershell
uv run python -m disease_info_agent.eval.run_harness --run-id <run_id>
# 报告：eval/output/report.html · 方法论：docs/EVAL.md
```

### 6. 启动 API + Web 演示

```powershell
# 仓库根目录
uv run python -m disease_info_agent.api
# 或
uv run uvicorn disease_info_agent.api.app:app --reload --port 8000
```

浏览器打开：**http://127.0.0.1:8000/** — Web UI 走 **POST /chat**，多步任务会展示 **Planner 执行轨迹**（时间线 + 每步工具/状态/摘要）。  
Swagger 文档：**http://127.0.0.1:8000/docs**

```powershell
# curl 验收
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"登革热如何传播?\"}"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"登革热如何传播?\"}"
```

---

## Web 完整体验

启动 API 后访问 **http://127.0.0.1:8000/** 即可：

- 查看当前 run 统计（facts 数、知识库文档数）
- 点击示例问题或自由输入，由 **Orchestrator Agent** 理解意图并调用 RAG / 状态查询等工具
- 疾病问答会走 LangGraph RAG，回答末尾带来源链接
- 可问「知识库有多少文档」「评测 pass rate 多少」等非医学问题
- 无关医学问题（如「今天天气」）会正确拒答
- 可选「运行 Pipeline」在后台重建知识库（需数分钟 + LLM 调用）
- 右上角 **API 文档** 跳转 Swagger

---

## MCP（Cursor 集成）

`.cursor/mcp.json` 已配置 stdio 服务。在 Cursor Settings → MCP 启用 `disease-info` 后可使用：

| 工具 | 功能 |
|------|------|
| `get_latest_run` | 最新 run 统计 |
| `query_disease` | RAG 问答 |
| `run_pipeline` | 跑完整 pipeline |
| `get_eval_report` | 最新评测摘要 |

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 框架 | Youtu-Agent（`utu/`）、Hydra YAML 配置 |
| **Orchestrator** | UTU **SimpleAgent** + OpenAI Agents SDK `function_tool` |
| LLM | OpenAI 兼容 API（SiliconFlow / DeepSeek） |
| 抽取 Agent | UTU SimpleAgent + trafilatura 抓取 |
| 向量库 | Chroma + BGE 类 embedding |
| RAG 编排 | LangChain + **LangGraph**（grade / rewrite / validate） |
| 评测 | Ragas + pytest golden set |
| 服务化 | FastAPI + uvicorn |
| MCP | mcp FastMCP stdio |

---

## 优化清单

- **Phase 3 协作可视化**：执行轨迹写入 `output/traces/*.json`，Web 展示 Agent 流程图 + LangGraph RAG 内部链路
- **Phase 4 KB 自主扩展**：仅当问题涉及四类疾病且 KB 答不好时才自动扩展；无关问题（如天气）直接拒答，不搜索。手动：`expand_knowledge_base`；关闭自动：`AUTO_KB_EXPAND=false`
- **Phase 2 Planner**：Task Planner 输出 JSON 计划，Executor 按序调用工具，Synthesizer 汇总
- **Phase 1 Orchestrator**：单步 SimpleAgent 调度 5 工具（Planner 判定 direct 时走此路径）
- **LangGraph 拒答链**：无关问题 rewrite 为 `UNRELATED` → 直接 refuse，避免幻觉
- **Citation 校验**：答案 validate 节点 + 来源 URL 从检索 metadata 提取
- **MCP 冷启动**：后台 RAG graph warmup + 进度 heartbeat，缓解 120s 超时
- **Chroma 相关度**：distance → `1/(1+d)` 修正负分警告
- **Qwen3 超时**：`enable_thinking: false` + 300s timeout
- **Eval Harness**：16 题 golden set（含 3 无关题）+ pipeline pytest 阈值
- **Web Demo**：单页聊天 UI，零额外前端构建

---

## 目录结构

```
disease_info_agent/
├── main.py              # CLI 入口（pipeline / ask / chat）
├── agent/               # Coordinator + Planner + Orchestrator + Executor
├── pipeline/            # fetch / extract / validate / aggregate / report
├── rag/                 # ingest / ask / graph
├── mcp_server/server.py   # Cursor MCP
├── eval/                # harness + golden set
├── api/app.py           # FastAPI + Web UI（/chat + /ask）
├── output/runs/         # 每次运行产物
└── urls.txt             # 默认 URL 列表

configs/agents/disease_info/
├── task_planner.yaml    # Phase 2 任务规划
├── orchestrator.yaml    # Phase 1 单步路由
└── synthesizer.yaml     # 多步结果汇总
```

---

## 验收清单（B5-4）

| 验收项 | 命令 / 位置 | 标准 |
|--------|-------------|------|
| 采集 | `main.py pipeline` | ≥9/10 URL 有数据，存在 `run_manifest.json` |
| RAG ingest | `main.py ingest` 或 pipeline 含 ingest | 文档数 N > 0 |
| RAG ask | `main.py ask` 或 Web `/ask` | 四类疾病问题有 citation |
| Orchestrator | `main.py chat` 或 Web `/chat` | 医学问答 + run 状态 + eval 查询均可路由 |
| 拒答 | ask 无关问题 | 返回「根据现有知识库无法回答该问题。」 |
| Eval | `eval/run_harness` | `overall_passed=true`，`report.html` 可打开 |
| API | `GET /health` | 返回 `{"status":"ok"}` |
| Web | http://127.0.0.1:8000 | 可问答（Orchestrator）、可看来源 |

---

## Demo 截图占位

> 答辩/作业时替换为实际截图路径

1. `docs/screenshots/01-pipeline-done.png` — Pipeline 终端完成输出  
2. `docs/screenshots/02-ask-citation.png` — ask 问答 + 来源  
3. `docs/screenshots/03-langgraph-flow.png` — LangGraph 流程图  
4. `docs/screenshots/04-mcp-tools.png` — Cursor MCP 工具列表  
5. `docs/screenshots/05-eval-report.png` — eval/report.html  
6. `docs/screenshots/06-fastapi-docs.png` — FastAPI /docs  

---

## 常见问题

**MCP 超时 / disconnected**  
在 Cursor Settings → MCP 重新启用 `disease-info`；改依赖前先关 MCP，避免 `.venv` 文件锁。

**uv sync 报 jiter 文件被占用**  
结束残留的 `.venv\Scripts\python.exe`（旧 MCP 进程）后再 sync。

**ask 返回拒答**  
对比类或多疾病问题可能触发 grade=no；可换更具体单疾病问题，或调 `rag/settings.py` 中 `MIN_RELEVANCE_SCORE`。

---

## License

与上级仓库 Youtu-Agent 保持一致。
