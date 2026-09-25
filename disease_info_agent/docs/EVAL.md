# Eval Harness — 评测方法论

本文档说明 `disease_info_agent` 如何**可重复、可量化**地评估 RAG 质量。面试时可作为「你怎么保证系统质量」的回答依据。

---

## 1. 评测目标

| 维度 | 含义 | 主要指标 |
|------|------|----------|
| **行为正确** | 该答的答、不该答的拒 | `behavior_pass`, `refusal_accuracy` |
| **Faithfulness** | 答案是否被检索上下文支持 | Ragas `faithfulness`（主）/ heuristic（备） |
| **Answer Relevancy** | 答案是否切题 | Ragas `answer_relevancy`（主） |
| **Citation** | 是否引用期望域名的来源 | `citation_pass` |
| **Pipeline 健康** | 采集与入库是否达标 | pytest `test_pipeline.py` |

---

## 2. Golden Set 结构

文件：`eval/golden/rag_questions.jsonl`（JSONL，每行一题）

### 2.1 Core 集（q01–q16）

- 覆盖 4 类疾病：登革病毒、猩红热、裂谷热、流感
- 含 **3 条无关题**（`should_answer: false`）测拒答
- 含 **2 条对比题**（q07、q11）测双路检索

### 2.2 Hard 集（h01–h05）

| ID | 类别 | 考察点 |
|----|------|--------|
| h01 | numeric | 能否引用 KB 中的数值事实（裂谷热病死率 ~50%） |
| h02 | hallucination_trap | 能否拒绝把流感药（奥司他韦）套用到登革热 |
| h03 | out_of_kb_compare | 知识库外疾病（埃博拉）+ 对比 → 必须拒答 |
| h04 | entity | 病原体实体是否正确（A 组链球菌） |
| h05 | reasoning | 多句推理（流感疫苗为何每年打） |

可选字段：

- `expected_keywords`：答案中应出现的关键词（hard case 诊断用）
- `expected_source_contains`：citation 域名片段
- `notes`：人工标注说明（便于答辩）

---

## 3. 运行方式

```powershell
# 完整评测（默认 Ragas + pytest）
uv run python -m disease_info_agent.eval.run_harness --run-id <run_id>

# 快速调试（跳过 Ragas API 调用）
uv run python -m disease_info_agent.eval.run_harness --run-id <run_id> --no-ragas

# 只看 hard cases
uv run python -m disease_info_agent.eval.run_harness --run-id <run_id> --tier hard
```

输出：

- `eval/output/report.json` — 机器可读
- `eval/output/report.html` — 答辩截图用

---

## 4. Ragas vs Heuristic

| 来源 | 何时使用 | 局限 |
|------|----------|------|
| **Ragas** | 默认；LLM-as-judge + embedding | 依赖 API、有成本、判分有波动 |
| **Heuristic** | Ragas 不可用 / `--no-ragas` | 基于 validate_pass、citation、拒答规则，**不能替代**语义 faithfulness |

报告里每题标注 `faithfulness_source` / `answer_relevancy_source`（`ragas` 或 `heuristic`）。

**Pass 规则（单题）：**

- `should_answer=false` → `behavior_pass` 即可
- `should_answer=true` → `behavior_pass` 且 faithfulness ≥ 0.5 且 answer_relevancy ≥ 0.5
- **Ragas 误判兜底**：Ragas faithfulness < 0.5，但 heuristic ≥ 0.8 且 `keyword_pass=true` 且 relevancy ≥ 0.5 → 仍记 pass（并在 report 标注 `ragas_disagree_override`）

**Overall pass：** `pass_rate ≥ 0.7` 且 pipeline pytest 通过。

---

## 5. LangGraph 被测路径

默认走 `rag/graph.py`（非 `--simple` 线性链）：

```
guard(知识库外疾病) → retrieve → grade → rewrite/refuse → generate → validate
```

对比题额外：`detect_comparison` → 双路 retrieve → 合并上下文。

---

## 6. 答辩话术模板

> 我们维护 **21 题 golden set**（16 core + 5 hard），其中 hard 集专门测数值幻觉、库外拒答和实体混淆。  
> 主指标用 **Ragas faithfulness / answer_relevancy**，辅以 behavior 与 citation 规则。  
> 最近一次 run `<run_id>` 上 pass_rate 为 **X%**，refusal_accuracy **Y%**。  
> 对比题从 87.5% 提升到 100% 的改动是双路检索，有 harness 前后 report 可复现。

---

## 7. 已知局限（诚实说明）

1. Golden set 规模仍小（21 题），未覆盖多轮对话
2. Ragas 判分与评测 LLM 绑定，存在 judge 偏差
3. `expected_keywords` 为弱监督，不做 hard fail（仅报告展示）
4. Pipeline 数据源变更后需重跑 ingest + harness 才可比

---

## 8. 后续扩展（含金量 roadmap）

- [ ] 消融实验：线性 RAG vs LangGraph（见 `docs/ablation.md` 待写）
- [ ] 扩大 hard set + 人工标注 gold answer
- [ ] Hybrid 检索（BM25 + 向量）对比实验
- [ ] 调用 Phoenix/LangSmith 留存 trace 截图
