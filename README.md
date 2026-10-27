# 多 Agent 协作框架

可扩展的 LangGraph 多 Agent 骨架：Supervisor 调度专家，专家可带工具，支持多轮会话与断点续跑。默认带 researcher / writer / reviewer，方便替换成自己的业务角色。

## 结构

```
src/multi_agent/
  config.py          # 环境变量与 Settings
  state.py           # 共享状态（messages / task / turn / stop_reason）
  registry.py        # Agent 注册表与模型工厂
  graph.py           # Supervisor 图编排、检查点、路由
  agents/defaults.py # 示例专家
  tools/workspace.py # 示例工具（本地笔记）
  cli.py             # 命令行入口（支持多轮、进度输出）
tests/
  test_registry.py   # 注册表行为
  test_graph.py      # 离线假模型跑通路由/失败隔离/持久化
```

流程：`START → supervisor → 某个专家 → supervisor → … → END`

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
- 有了检查点，`interrupt()` 人工审批与 `Command(resume=...)` 续跑都可以接。

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
| `MULTI_AGENT_WORKER_STEPS` | 单个专家内部最大步数，默认 25 |
| `MULTI_AGENT_CHECKPOINT_DB` | SQLite 检查点路径；`memory` 表示不落盘 |
| `MULTI_AGENT_NOTES_DIR` | 笔记目录，默认 `notes` |
| `MULTI_AGENT_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING`，默认 WARNING |

## 设计约定

- **状态**：`messages` 用 `add_messages`；`turn` / `stop_reason` 是覆盖语义，专家不要回写。
- **专家可见性**：专家只收到业务消息，`name="supervisor"` 的路由记录会被过滤，避免上下文污染。
- **失败隔离**：专家抛异常会转成一条 `[专家] 执行失败：...` 消息交回 supervisor，可换人重试。
- **路由**：候选值由注册表动态生成 `Literal`，非法值进不了 Pydantic 校验；structured output 不可用时退化为 JSON 解析。
- **产出**：保留原始 `AIMessage`（含 content blocks），不会把结构化内容压成字符串。

## 测试

```bash
pytest -q
```

测试用离线假模型（`FakeChatModel`）覆盖：完整四跳路由、专家看不到路由消息、专家失败隔离、max_turns 保护、非法路由降级、按 thread 保留历史。不消耗 API。

## 已知取舍

这是 **supervisor 路由**（每跳回中枢），适合需要统一审计与集中改规则的场景；若追求并行与低成本，官方更推荐「专家即工具」的 Subagents 模式，可在此基础上改造。
