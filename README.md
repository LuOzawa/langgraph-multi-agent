# 多 Agent 协作框架

[![CI](https://github.com/LuOzawa/langgraph-multi-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LuOzawa/langgraph-multi-agent/actions/workflows/ci.yml)

可扩展的 LangGraph 多 Agent 骨架：Supervisor 调度专家，专家可带工具，支持多轮会话、断点续跑、高风险动作人工审批与上下文预算控制。默认带 researcher / writer / reviewer，并附一套跨境电商示例团队。

## 结构

```
src/multi_agent/
  config.py          # 环境变量与 Settings
  state.py           # 共享状态（messages / task / turn / stop_reason）
  registry.py        # Agent 注册表与模型工厂
  graph.py           # 图编排、检查点、路由、审批、并发
  audit.py           # 审批审计（JSONL）
  agents/defaults.py # 示例专家
  tools/workspace.py # 示例工具（本地笔记）
  tools/ecommerce.py # 示例工具（到岸成本与定价）
  cli.py             # 命令行入口（多轮、进度、审批）
examples/
  ecommerce_team.py  # 跨境电商团队（含审批门）
tests/
  test_registry.py   # 注册表行为
  test_graph.py      # 路由/失败隔离/持久化/审批/裁剪
  test_tools.py      # 定价工具
.github/workflows/ci.yml
```

流程：`START → supervisor → 某个专家 → supervisor → … → END`

## 当前状态

| 能力 | 状态 |
| --- | --- |
| Supervisor 路由（动态 `Literal` + JSON 兜底） | ✅ 完成 |
| SQLite 检查点 / thread 多轮 | ✅ 完成 |
| 专家失败隔离、max_turns 保护 | ✅ 完成 |
| 人工审批（专家级 + 工具级） | ✅ 完成 |
| 审批跨进程恢复（`--resume` / `resume_task`） | ✅ 完成 |
| 审批审计 JSONL | ✅ 完成 |
| 上下文 token 裁剪 | ✅ 完成 |
| token 用量统计 | ✅ 完成 |
| 独立任务并行（`run_parallel`） | ✅ 完成 |
| 跨境电商示例团队 | ✅ 完成 |
| CI（ruff + mypy + pytest，3.10–3.12） | ✅ 完成 |
| HTTP API 服务层 | ❌ 未开始 |
| 图内 `Send` 扇出（专家并行） | ❌ 未开始 |
| token 级流式输出 | ❌ 未开始 |
| 汇率 / 多币种 | ❌ 未开始 |
| LangSmith / OTel 追踪 | ❌ 未开始 |
| 评测集（golden tasks） | ❌ 未开始 |

## 快速开始

需要 Python 3.10+（推荐 3.11）。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# 编辑 .env：填入 OPENAI_API_KEY；用兼容中转时改 OPENAI_BASE_URL
# 注意：密钥留空会直接报错，不要留 `***` 之类的占位符

multi-agent "用三句话说明这个框架怎么扩展"
```

交互模式（保持同一 thread，多轮上下文）：

```bash
python -m multi_agent
# 任务> 先调研一下 LangGraph 的中断机制
# 任务> 把上一步整理成给同事的一页说明
# /new 新建会话   空行退出
```

## 会话与持久化

- 默认检查点写入 `.multi_agent/checkpoints.sqlite`，按 `thread_id` 隔离。
- 指定会话：`multi-agent --thread-id my-task "继续上次的任务"`
- 想关掉持久化：`MULTI_AGENT_CHECKPOINT_DB=memory`
- 检查点是审批与续跑的前提，默认开启。

## 人工审批（高风险动作）

给专家加 `requires_approval=True`，supervisor 选中它时不会直接执行，而是先进入 `approve_<name>` 节点调用 `interrupt()` 挂起：

```python
AgentSpec(name="publisher", ..., requires_approval=True)
```

- CLI 会提示 `批准执行？[y/N]`，拒绝时把拒绝原因交回 supervisor，由它决定换人或收尾
- 调试时可 `multi-agent --auto-approve "..."` 跳过审批
- 程序化调用：`run_task_full(..., approver=lambda p: {"approved": True, "note": "ok"})`
- 不传 `approver` 也不自动批准时，返回 `stop_reason="awaiting_approval"` 与 `pending` 载荷

### 工具级审批

只有真正高风险的**工具**需要停，查询类工具照常跑：

    AgentSpec(
        name="pricing",
        ...,
        tools_factory=lambda: [lookup_rate, update_price],
        approval_tools=("update_price",),   # 只有改价需要人工确认
    )

机制：未批准时工具被包裹成“只回标记、不执行”；supervisor 收到标记后转审批节点，批准后把该工具加入 `approved_tools` 并重跑该专家，工具才真正执行。审批前会附带 `preview`（即将提交的内容）供人核对。

### 恢复挂起的审批

进程退出也不怕，状态在检查点里：

```bash
multi-agent --resume <thread_id>        # 交互式确认后继续
multi-agent --resume <thread_id> --auto-approve   # 调试用
```

交互模式里还有 `/pending`（看当前挂起）和 `/resume y|n [备注]`。

代码里：

```python
from multi_agent.graph import pending_approval, resume_task

payload = pending_approval(app, thread_id)
result = resume_task(thread_id, {"approved": True, "note": "复核通过"}, app=app)
```

### 审计日志

每次审批决策追加一行 JSONL（`MULTI_AGENT_AUDIT_LOG`，默认 `.multi_agent/approvals.jsonl`）：时间、thread、专家、是否批准、备注、任务摘要。写盘失败只告警，不中断业务。

## 并行任务

supervisor 内部是串行调度的；但多个**独立任务**（多 SKU / 多站点）可以并行：

```python
from multi_agent.graph import run_parallel

results = run_parallel(["上架 SKU-1", "上架 SKU-2"], max_workers=2)
```

每个任务独立 thread 与 SQLite 连接。这不是图内 `Send` 扇出，专家之间不会并行。

## 上下文预算

长会话会把历史消息越堆越多。`MULTI_AGENT_CONTEXT_TOKENS`（默认 12000）按 token 预算裁剪历史，保留首条任务消息；设为 `0` 关闭。裁剪只影响发给模型的上下文，不删除已保存的检查点消息。

## 新增专家

```python
from multi_agent.registry import AgentSpec, default_registry
from multi_agent.graph import run_task
from multi_agent.tools.workspace import workspace_tools

registry = default_registry()
registry.register(
    AgentSpec(
        name="coder",
        description="根据已确认方案写代码片段。",
        system_prompt="你是程序员。只写实现，不改需求。",
        tools_factory=workspace_tools,   # 或不给工具: lambda: ()
        model="gpt-4o",                  # 可选：专家单独指定模型
        temperature=0.2,                 # 可选
        max_steps=8,                     # 可选：该专家内部工具循环步数上限
    )
)

print(run_task("给笔记工具加一个删除接口", registry=registry))
```

重名会抛 `ValueError`，`supervisor` 是保留名。

## 新增工具

在 `src/multi_agent/tools/` 里用 `@tool` 定义函数，放进对应 `AgentSpec.tools_factory` 的返回列表。工厂每次调用都应返回新列表，避免专家之间共享可变状态。

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | 必填，留空启动即报错 |
| `OPENAI_BASE_URL` | 默认官方接口；中转需支持 function calling，否则自动退化为 JSON 路由 |
| `OPENAI_MODEL` | 专家默认模型，默认 `gpt-4o-mini` |
| `OPENAI_TIMEOUT` | 单次请求超时秒数，默认 120 |
| `OPENAI_MAX_RETRIES` | 请求重试次数，默认 2 |
| `MULTI_AGENT_SUPERVISOR_MODEL` | 路由专用模型；留空跟随 `OPENAI_MODEL` |
| `MULTI_AGENT_MAX_TURNS` | Supervisor 最大调度次数，默认 12；触顶会返回部分结果并附提示 |
| `MULTI_AGENT_WORKER_STEPS` | 单个专家内部最大步数，默认 25（可被 `AgentSpec.max_steps` 覆盖） |
| `MULTI_AGENT_CONTEXT_TOKENS` | 发给模型的上下文 token 预算，默认 12000；`0` 关闭裁剪 |
| `MULTI_AGENT_CHECKPOINT_DB` | SQLite 检查点路径；`memory` 表示不落盘 |
| `MULTI_AGENT_NOTES_DIR` | 笔记目录，默认 `notes` |
| `MULTI_AGENT_AUDIT_LOG` | 审批审计 JSONL 路径；`off` 关闭 |
| `MULTI_AGENT_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING`，默认 WARNING |

## 设计约定

- **状态**：`messages` 用 `add_messages`；`turn` / `stop_reason` 是覆盖语义，专家不要回写。
- **专家可见性**：专家只收到业务消息，`name="supervisor"` 的路由记录会被过滤，避免上下文污染。
- **失败隔离**：专家抛异常会转成一条 `[专家] 执行失败：...` 消息交回 supervisor，可换人重试。
- **路由**：候选值由注册表动态生成 `Literal`，非法值进不了 Pydantic 校验；structured output 不可用时退化为 JSON 解析。
- **产出**：保留原始 `AIMessage`（含 content blocks），不会把结构化内容压成字符串。
- **用量**：`RunResult.usage` 汇总 `input_tokens / output_tokens / total_tokens`（模型返回 usage 时才有值）。
- **审批粒度**：支持专家级（`requires_approval`）与工具级（`approval_tools`）；工具级会在首次尝试时暂停并按需放行。

## 文档

- [架构说明](docs/architecture.md)：组件图、状态契约、路由/审批/裁剪/并发
- [接入指南](docs/integration-guide.md)：把默认团队换成自己业务的五步
- [中转兼容性](docs/relay-compatibility.md)：真实探针结果
- [CHANGELOG](CHANGELOG.md)

## 模型/中转兼容性

已对 deepseek-v4-flash 做真实探针与端到端验证：普通对话与 tool calling 可用，`response_format: json_schema` 返回 400，路由自动走 JSON 兜底并正常完成。详见 [docs/relay-compatibility.md](docs/relay-compatibility.md)。

## 跨境电商示例

`examples/ecommerce_team.py` 组了一个五人团队：

| 专家 | 职责 |
| --- | --- |
| sourcing | 选品与供应商比价 |
| listing | 英文标题 / 五点描述 / 搜索词 |
| pricing | 调用 `calc_landed_cost` 算到岸成本、佣金与建议售价（支持 VAT、退货率、广告费） |
| compliance | 禁售词、认证与目的国合规 |
| publisher | 上架发布，`requires_approval=True` |

```bash
python examples/ecommerce_team.py "帮我把这款收纳盒上架到亚马逊美国站"
```

## 测试与检查

```bash
pytest -q      # 22 个离线测试，不消耗 API
ruff check .   # 静态检查
mypy src       # 类型检查
```

CI 在 Python 3.10 / 3.11 / 3.12 上跑同一套命令。测试覆盖：完整四跳路由、路由消息不外泄、专家失败隔离、max_turns 保护、非法路由降级、按 thread 保留历史、审批通过/拒绝/待审批、工具级审批拦截与放行、审计落盘、并行任务、上下文裁剪、定价工具边界。

## 未完成的板块（欢迎接手）

按优先级排列，每条都写了入口和建议做法，方便 clone 之后直接开工。

### P0 · HTTP API 服务层

**现状**：只有 CLI 和 Python 库两种入口，ERP / 飞书 / 前端无法通过网络调用。

**建议**：
- 新建 `src/multi_agent/api.py`，用 FastAPI 包现有函数，不改核心
- 端点：`POST /tasks`（提交，返回 thread_id）、`GET /tasks/{id}`（结果/状态）、`GET /approvals`（待审批列表）、`POST /approvals/{id}`（批准/拒绝）
- 鉴权：一个 `X-API-Key` 环境变量即可，不要上完整 OAuth
- 验收：`curl` 能走完“提交 → 待审批 → 批准 → 拿结果”闭环

### P1 · 图内 `Send` 扇出

**现状**：supervisor 一次只能选一个专家；多个独立子任务只能排队或走 `run_parallel`（任务级，不是图内）。

**建议**：
- 让 supervisor 返回 `[Send("agent", {...}), ...]`，或在图形层加一个 fan-out 节点
- 注意状态合并：`messages` 有 `add_messages` 没问题，`turn` / `stop_reason` 是覆盖语义，并行分支会互相覆盖，需要改成 `Annotated[..., operator.add]` 或只让 supervisor 写
- 验收：两个专家并发执行，总耗时接近较慢的那个

### P1 · token 级流式

**现状**：`on_event` 只到节点级（“谁开始/谁完成”），看不到正在写的内容。

**建议**：
- 用 `astream_events(version="v2")` 过滤 `on_chat_model_stream`
- 在 `run_task_full` 旁加 `stream_task()` 生成器，CLI 与未来的 API 共用
- 验收：CLI 能逐字输出最终答案

### P2 · 汇率与多币种

**现状**：`calc_landed_cost` 所有金额按美元输入，人民币采购价要人工换算。

**建议**：
- 加 `fx_rate`（CNY/USD）与 `fob_cny` 参数，或接一个汇率工具（注意缓存与失败降级）
- 验收：人民币 FOB 能直接算出美元售价，汇率缺失时报错而不是静默用 1.0

### P2 · 可观测性

**现状**：有结构化日志和 token 统计，但没有链路追踪。

**建议**：接 LangSmith（设 `LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY` 即可，无需改代码），或在 `run_config` 里加 `metadata` 供 OTel 使用。

### P2 · 评测集

**现状**：测试全是假模型，验证的是“流程对不对”，不是“答得好不好”。

**建议**：建 `evals/`，放 10–20 条真实任务 + 期望要点，用真实模型跑，记录通过率与 token 成本。改 prompt 前后各跑一次。

## Clone 之后的接手步骤

```bash
git clone https://github.com/LuOzawa/langgraph-multi-agent.git
cd langgraph-multi-agent

# 1. 装依赖（不需要 API key）
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. 先跑离线测试，确认环境没问题（22 个，全部不联网）
pytest -q

# 3. 配密钥
cp .env.example .env   # 填 OPENAI_API_KEY

# 4. 验证你的中转是否兼容（假设用 deepseek）
python tools/compat_probe.py --provider deepseek

# 5. 跑一次跨境示例，确认整条链路
python examples/ecommerce_team.py "帮我把这款收纳盒上架到亚马逊美国站"
```

读代码的顺序：`state.py`（状态契约）→ `graph.py`（路由/审批/裁剪）→ `registry.py`（怎么加专家）→ `cli.py`（入口）。架构细节见 [docs/architecture.md](docs/architecture.md)。

## 已知取舍

这是 **supervisor 路由**（每跳回中枢），适合需要统一审计与集中改规则的场景；若追求并行与低成本，官方更推荐「专家即工具」的 Subagents 模式，可在此基础上改造。

## 许可

MIT，见 [LICENSE](LICENSE)。
