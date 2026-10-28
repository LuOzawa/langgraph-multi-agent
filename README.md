# LangGraph 多 Agent 协作框架

[![CI](https://github.com/LuOzawa/langgraph-multi-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LuOzawa/langgraph-multi-agent/actions/workflows/ci.yml)

可扩展的 **LangGraph** 多 Agent 骨架：一个 supervisor 调度若干专家，专家各自带工具，
支持多轮会话、断点续跑、高风险动作人工审批、上下文预算控制。
默认带 researcher / writer / reviewer 三个专家，另附一套跨境电商示例团队。

安装后是一条命令：

```bash
multi-agent "用三句话说明这个框架怎么扩展"
```

## 安装

本项目**尚未发布到 PyPI**（PyPI 上的 `multi-agent` 是另一个项目），请从源码安装：

```bash
git clone https://github.com/LuOzawa/langgraph-multi-agent.git
cd langgraph-multi-agent

python3.11 -m venv .venv && source .venv/bin/activate
# Windows: py -3.11 -m venv .venv && .venv\Scripts\activate

pip install -e "."            # 运行时最小依赖
pip install -e ".[dev]"       # 加测试与静态检查（pytest / ruff / mypy）
pip install -e ".[api]"       # 需要 HTTP 服务时（多一个 uvicorn）
```

## 它在解决什么

写多 Agent 系统的难点不在「调通一次对话」，而在**它跑起来之后的事**：

- 跑到第 8 轮突然超预算，前面的上下文全堆着；
- 专家调了个危险工具，没人拦得住；
- 进程崩了，检查点还在，但没人知道怎么接回去；
- 某个专家抛异常，整个任务就废了；
- 换了 prompt 之后，输出变好了还是变坏了，说不清。

所以这个框架把上面这些都做进去了：**审批门、检查点、失败隔离、上下文裁剪、用量统计、审计流水**，
都不是示例代码，是默认就有的行为。

## 快速开始

需要 Python 3.10+（推荐 3.11）。先按上一节 clone 并 `pip install -e ".[dev]"`，然后：

```bash
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

脚本 / CI 里看退出码：`stop_reason="error"` 的任务返回 `1`，其余（含 `awaiting_approval`）返回 `0`；
挂起的审批用 `--resume <thread_id>` 接着处理。

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
  api/
    __main__.py      # multi-agent-api 启动器
    app.py           # FastAPI 应用与路由
    schemas.py       # 请求 / 响应模型
    service.py       # 任务表、后台执行、审批恢复（不依赖 FastAPI）
examples/
  ecommerce_team.py  # 跨境电商团队（含审批门）
tests/
  test_config.py     # Settings 校验（日志级别等）
  test_cli.py        # CLI 退出码与审批提示
  test_registry.py   # 注册表行为
  test_graph.py      # 路由/失败隔离/持久化/审批/裁剪
  test_tools.py      # 定价工具
  test_api.py        # HTTP API（提交 / 轮询 / 审批 / 取消与回收 / 错误映射）
```

流程：`START → supervisor → 某个专家 → supervisor → … → END`

## 核心机制

### 会话与持久化

- 默认检查点写入 `.multi_agent/checkpoints.sqlite`，按 `thread_id` 隔离；
- 指定会话：`multi-agent --thread-id my-task "继续上次的任务"`；
- 想关掉持久化：`MULTI_AGENT_CHECKPOINT_DB=memory`；
- 检查点是审批与续跑的前提，默认开启。

### 人工审批（高风险动作）

给专家加 `requires_approval=True`，supervisor 选中它时不会直接执行，而是先进入 `approve_<name>` 节点调用 `interrupt()` 挂起：

```python
AgentSpec(name="publisher", ..., requires_approval=True)
```

- CLI 会提示 `批准执行？[y/N]`，拒绝时把拒绝原因交回 supervisor，由它决定换人或收尾；
- 调试时可 `multi-agent --auto-approve "..."` 跳过审批；
- 程序化调用：`run_task_full(..., approver=lambda p: {"approved": True, "note": "ok"})`；
- 不传 `approver` 也不自动批准时，返回 `stop_reason="awaiting_approval"` 与 `pending` 载荷。

#### 工具级审批

只有真正高风险的**工具**需要停，查询类工具照常跑：

```python
AgentSpec(
    name="pricing",
    ...,
    tools_factory=lambda: [lookup_rate, update_price],
    approval_tools=("update_price",),   # 只有改价需要人工确认
)
```

机制：未批准时工具被包裹成「只回标记、不执行」；supervisor 收到标记后转审批节点，
批准后把该工具加入 `approved_tools` 并重跑该专家，工具才真正执行。
审批前会附带 `preview`（即将提交的内容）供人核对。

#### 恢复挂起的审批

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

#### 审计日志

每次审批决策追加一行 JSONL（`MULTI_AGENT_AUDIT_LOG`，默认 `.multi_agent/approvals.jsonl`）：
时间、thread、专家、是否批准、备注、任务摘要。写盘失败只告警，不中断业务。

### 并行任务

supervisor 内部是串行调度的；但多个**独立任务**（多 SKU / 多站点）可以并行：

```python
from multi_agent.graph import run_parallel

results = run_parallel(["上架 SKU-1", "上架 SKU-2"], max_workers=2)
```

每个任务独立 thread 与 SQLite 连接。这不是图内 `Send` 扇出，专家之间不会并行。

### 上下文预算

长会话会把历史消息越堆越多。`MULTI_AGENT_CONTEXT_TOKENS`（默认 12000）按 token 预算裁剪历史，
保留首条任务消息；设为 `0` 关闭。裁剪只影响发给模型的上下文，不删除已保存的检查点消息。

### 失败隔离与轮次保护

专家抛异常会被转成一条 `[专家] 执行失败：...` 的消息交回 supervisor，由它决定换人重试，
而不是整个任务崩掉。`MULTI_AGENT_MAX_TURNS`（默认 12）是调度轮次上限，触顶会返回部分结果并附提示，
不会无限绕圈。

## HTTP API 服务

ERP、飞书、前端不想装 Python 时，用 `multi-agent-api` 把同一套图暴露成 HTTP：

```bash
pip install -e ".[api]"
OPENAI_API_KEY=sk-... multi-agent-api            # 默认 127.0.0.1:8000
multi-agent-api --host 0.0.0.0 --port 8765 --max-workers 8 --task-timeout 900  # 0.0.0.0 仅在有鉴权代理后使用
multi-agent-api --log-level DEBUG                # 默认跟随 MULTI_AGENT_LOG_LEVEL
```

`--host` / `--port` / `--max-workers` / `--task-timeout` / `--retention` / `--max-records` 也可用
`MULTI_AGENT_API_HOST` / `MULTI_AGENT_API_PORT` / `MULTI_AGENT_API_MAX_WORKERS` /
`MULTI_AGENT_API_TASK_TIMEOUT` / `MULTI_AGENT_API_RETENTION` / `MULTI_AGENT_API_MAX_RECORDS`
给默认值。OpenAPI 文档在 `/docs`。

| 方法与路径 | 说明 |
| --- | --- |
| `POST /tasks` | 提交任务（`{"task": "...", "thread_id": "可选"}`），立即返回 `202` + `queued` |
| `GET /tasks` | 列出任务，可按 `?status=` 过滤，按创建时间升序 |
| `GET /tasks/{thread_id}` | 单个任务状态 |
| `DELETE /tasks/{thread_id}` | 取消任务（软取消），立即返回 `202` + `canceled` |
| `GET /approvals` | 当前挂起的人工审批 |
| `GET /approvals/{thread_id}` | 单个待审批载荷 |
| `POST /approvals/{thread_id}` | 批准或拒绝（`{"approved": true, "note": "同意"}`），立即返回 `202` |

状态机：`queued` → `running` → `completed` / `failed`，遇到高风险动作停在 `awaiting_approval`，
批准或拒绝后回到 `running`；`DELETE`、执行超时（`--task-timeout`）或待审批过期会把任务收敛到
`canceled`。错误码：`404` 任务不存在、`409` 任务已存在或当前不可审批或已结束、
`503` 服务正在关闭或任务表已满、`422` 参数非法（空任务、非法 `thread_id`）。

几点约定：

- **任务表在进程内**：重启后 `GET /tasks` 不认识旧任务，但检查点在；默认团队可用 `multi-agent --resume <thread_id>` 接着跑，
  自定义 `registry` 的 thread 要在自己的脚本里用同一 registry 重新 `build_graph(registry=registry)` 再 `resume_task(...)`（CLI 只加载默认团队）；
- **任务表会回收**：`--retention` 之后已结束任务从 `GET /tasks` 消失（再查 `404`）；表超过 `--max-records`
  时只淘汰已结束记录，`queued` / `running` / `awaiting_approval` 不会被挤掉；活跃记录占满上限后，
  新提交返回 `503`（`detail` 含 `task table is full`），等任务结束或调大 `--max-records`；待审批超过保留期转
  `canceled`（`error="approval expired"`），记录里的图实例同时释放（SQLite 连接按线程复用，不随记录关闭）；
- **取消是软取消**：`DELETE` 立即返回 `canceled`，但 Python 线程无法强杀，底层模型调用会继续到
  `OPENAI_TIMEOUT`（默认 120s），结果被丢弃——这期间该 worker 仍被占用；
- **执行超时**：`--task-timeout` 作用于 `queued` / `running`，批准审批后重新计时，超时 `error="task timed out after Ns"`；
- **单进程**：任务并发由 `--max-workers` 控制，不要给 uvicorn 开多 worker（每个进程一份任务表）；
- **审批串到同一条审计链路**：每次决策（含 `--resume` / `POST /approvals`）追加 JSONL（`MULTI_AGENT_AUDIT_LOG`）；
- **没有内置认证**：默认只监听回环地址，对外暴露请让反向代理做认证。

```bash
curl -s -X POST localhost:8000/tasks -H 'content-type: application/json' \
     -d '{"task": "上架收纳盒到亚马逊美国站", "thread_id": "api-1a2b3c4d"}'
curl -s localhost:8000/tasks/api-1a2b3c4d
curl -s localhost:8000/approvals
curl -s -X POST localhost:8000/approvals/api-1a2b3c4d \
     -H 'content-type: application/json' -d '{"approved": true, "note": "复核通过"}'
curl -s -X DELETE localhost:8000/tasks/api-1a2b3c4d   # 运行中 202 + canceled；已结束 409
```

## 扩展

### 新增专家

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

### 新增工具

在 `src/multi_agent/tools/` 里用 `@tool` 定义函数，放进对应 `AgentSpec.tools_factory` 的返回列表。
工厂每次调用都应返回新列表，避免专家之间共享可变状态。

## 配置项

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
| `MULTI_AGENT_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`，默认 WARNING |
| `MULTI_AGENT_API_HOST` | `multi-agent-api` 默认监听地址，默认 `127.0.0.1` |
| `MULTI_AGENT_API_PORT` | `multi-agent-api` 默认端口，默认 `8000` |
| `MULTI_AGENT_API_MAX_WORKERS` | `multi-agent-api` 并发任务数，默认 `4` |
| `MULTI_AGENT_API_TASK_TIMEOUT` | 单任务执行超时秒数，默认 `1800`；`0` 不限 |
| `MULTI_AGENT_API_RETENTION` | 任务记录保留秒数，默认 `86400`；`0` 关闭回收 |
| `MULTI_AGENT_API_MAX_RECORDS` | 任务表容量上限，默认 `1000`（只淘汰已结束记录；活跃记录占满后新提交 `503`） |

## 设计约定

- **状态**：`messages` 用 `add_messages`；`turn` / `stop_reason` 是覆盖语义，专家不要回写；
- **专家可见性**：专家只收到业务消息，`name="supervisor"` 的路由记录会被过滤，避免上下文污染；
- **失败隔离**：专家抛异常会转成一条 `[专家] 执行失败：...` 消息交回 supervisor，可换人重试；
- **路由**：候选值由注册表动态生成 `Literal`，非法值进不了 Pydantic 校验；structured output 不可用时退化为 JSON 解析；
- **产出**：保留原始 `AIMessage`（含 content blocks），不会把结构化内容压成字符串；
- **用量**：`RunResult.usage` 汇总 `input_tokens / output_tokens / total_tokens`（模型返回 usage 时才有值）；
- **审批粒度**：支持专家级（`requires_approval`）与工具级（`approval_tools`）；工具级会在首次尝试时暂停并按需放行。

## 示例团队：跨境电商

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
pytest -q      # 66 个离线测试，不消耗 API
ruff check .   # 静态检查
mypy src       # 类型检查
```

CI 在 Python 3.10 / 3.11 / 3.12 上跑同一套命令。测试覆盖：完整四跳路由、路由消息不外泄、
专家失败隔离、max_turns 保护、非法路由降级、按 thread 保留历史、审批通过/拒绝/待审批、
工具级审批拦截与放行、审计落盘、并行任务、上下文裁剪、定价工具边界，
HTTP API（提交与轮询、审批恢复、错误映射、参数校验、取消 / 超时 / 保留期回收 / 容量上限），
以及真实图 + SQLite 的审批往返（含跨线程 `resume`）。

API 测试用 `fastapi.testclient` + 假执行器，不联网；`.[dev]` 已包含 fastapi / httpx，
跑真实服务才需要 `.[api]`（多一个 uvicorn）。

## 本地开发

```bash
git clone https://github.com/LuOzawa/langgraph-multi-agent.git
cd langgraph-multi-agent

# 1. 装依赖（不需要 API key）
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. 先跑离线测试，确认环境没问题（66 个，全部不联网）
pytest -q

# 3. 配密钥
cp .env.example .env   # 填 OPENAI_API_KEY

# 4. 验证你的中转是否兼容（默认读 .env 的 OPENAI_*；也可指定 provider / 显式参数）
python tools/compat_probe.py
python tools/compat_probe.py --provider deepseek
python tools/compat_probe.py --api-key sk-... --base-url https://... --model gpt-4o-mini

# 5. 跑一次跨境示例，确认整条链路
python examples/ecommerce_team.py "帮我把这款收纳盒上架到亚马逊美国站"
```

读代码的顺序：`state.py`（状态契约）→ `graph.py`（路由/审批/裁剪）→ `registry.py`（怎么加专家）→
`cli.py`（命令行入口）→ `api/`（HTTP 入口）。

## 还没做的

- **跨重启的任务表**：API 的任务状态只在进程内，重启后旧任务查不到（检查点仍在，可 `--resume` 接着跑）。
  要长期查询就得把任务表落到 SQLite，并把检查点表当内部实现、不对外暴露。
- **认证与限流**：API 没有鉴权、没有 CORS、没有限流，默认只监听回环地址，对外必须靠反向代理。
- **硬中断执行**：`DELETE` 是软取消——记录立刻变 `canceled`、结果丢弃，但底层模型调用仍在跑，
  要等 `OPENAI_TIMEOUT` 才释放 worker。真正的节点级中断需要把取消标志传进 `on_event`
  并在每个节点检查，属于 `LangGraphExecutor` 的改动。
- **图内 `Send` 扇出**：supervisor 一次只能选一个专家，多个独立子任务只能排队或走 `run_parallel`。
  改成 `Send` 扇出要注意状态合并——`messages` 用 `add_messages` 没问题，`turn` / `stop_reason` 是覆盖语义，
  并行分支会互相覆盖，得改成 `Annotated[..., operator.add]` 或只让 supervisor 写。
- **token 级流式**：`on_event` 只到节点级（谁开始/谁完成），看不到正在写的内容。
  用 `astream_events(version="v2")` 过滤 `on_chat_model_stream`，加一个 `stream_task()` 生成器。
- **汇率与多币种**：`calc_landed_cost` 的金额都按美元输入，人民币采购价要人工换算。
  加 `fx_rate`（CNY/USD）与 `fob_cny` 参数，汇率缺失时报错而不是静默用 1.0。
- **可观测性**：有结构化日志和 token 统计，但没有链路追踪。接 LangSmith 只需设
  `LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY`，不用改代码。
- **评测集**：现在的测试全是假模型，验证的是「流程对不对」，不是「答得好不好」。
  想建 `evals/`，放 10–20 条真实任务 + 期望要点，用真实模型跑，记录通过率与 token 成本，
  改 prompt 前后各跑一次。

## 文档

- [架构说明](docs/architecture.md)：组件图、状态契约、路由/审批/裁剪/并发
- [接入指南](docs/integration-guide.md)：把默认团队换成自己业务的五步
- [中转兼容性](docs/relay-compatibility.md)：真实探针结果
- [CHANGELOG](CHANGELOG.md)

## 模型 / 中转兼容性

已对 deepseek-v4-flash 做真实探针与端到端验证：普通对话与 tool calling 可用，
`response_format: json_schema` 返回 400，路由自动走 JSON 兜底并正常完成。
详见 [docs/relay-compatibility.md](docs/relay-compatibility.md)。

## 已知取舍

这是 **supervisor 路由**（每跳回中枢），适合需要统一审计与集中改规则的场景；
若追求并行与低成本，官方更推荐「专家即工具」的 Subagents 模式，可以在此基础上改造。

## 许可

MIT，见 [LICENSE](LICENSE)。
