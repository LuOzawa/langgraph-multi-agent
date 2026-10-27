# 多 Agent 协作框架

可扩展的 LangGraph 多 Agent 骨架：Supervisor 调度专家，专家可带工具。默认带了 researcher / writer / reviewer，方便你换成自己的业务角色。

## 结构

```
src/multi_agent/
  config.py          # 环境变量
  state.py           # 共享状态
  registry.py        # Agent 注册表
  graph.py           # Supervisor 图编排
  agents/defaults.py # 示例专家
  tools/workspace.py # 示例工具（本地笔记）
  cli.py             # 命令行入口
```

流程：`START → supervisor → 某个专家 → supervisor → … → END`

## 快速开始

需要 Python 3.10+（推荐 3.11）。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（兼容 OpenAI 的中转把 OPENAI_BASE_URL 改掉即可）

python -m multi_agent "用三句话说明这个框架怎么扩展"
```

或：

```bash
multi-agent "写一份给新同事的接入说明"
```

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
        tools_factory=workspace_tools,  # 或不给工具: lambda: ()
    )
)

print(run_task("给笔记工具加一个删除接口", registry=registry))
```

## 新增工具

在 `src/multi_agent/tools/` 里用 `@tool` 定义函数，放到对应 `AgentSpec.tools_factory` 的返回列表中。

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | 必填 |
| `OPENAI_BASE_URL` | 默认官方接口 |
| `OPENAI_MODEL` | 默认 `gpt-4o-mini` |
| `MULTI_AGENT_MAX_TURNS` | Supervisor 最大调度次数，默认 12 |
