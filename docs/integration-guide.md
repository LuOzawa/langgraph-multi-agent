# 接入自己的业务

以跨境电商为例，五步就能替换掉默认的 researcher / writer / reviewer。

## 1. 定义角色

```python
from multi_agent.registry import AgentRegistry, AgentSpec

registry = AgentRegistry()
registry.register(
    AgentSpec(
        name="listing",
        description="写亚马逊英文 listing：标题、五点描述、搜索词。",   # supervisor 靠它选人
        system_prompt="你是亚马逊文案专家，输出英文，避免夸大与医疗宣称。",
        model="gpt-4o",        # 可选：专家单独指定模型
        max_steps=6,           # 可选：工具循环步数
    )
)
```

**description 是路由质量的关键**：写成“这个专家在什么情况下该被选中”，不要写“负责 XX”。

## 2. 定义工具

```python
from langchain_core.tools import tool

@tool
def calc_landed_cost(fob_usd: float, freight_usd: float, duty_rate: float) -> str:
    """计算到岸成本。参数为美元与小数比例。"""
    ...
```

- 放进 `AgentSpec.tools_factory=lambda: [calc_landed_cost]`
- 工厂每次返回新列表，避免专家之间共享状态
- 工具要返回**确定性结果**，不要在里面再调模型

## 3. 标记高风险动作

```python
AgentSpec(name="publisher", ..., requires_approval=True)
```

选中后先 `interrupt()`，批准才执行；拒绝原因会回到 supervisor。每次决策写审计日志。

## 4. 配置

```bash
cp .env.example .env
```

| 变量 | 建议 |
| --- | --- |
| `OPENAI_MODEL` | 专家用强模型 |
| `MULTI_AGENT_SUPERVISOR_MODEL` | 用小模型省钱，路由任务简单 |
| `MULTI_AGENT_MAX_TURNS` | 业务流程长就调到 20 |
| `MULTI_AGENT_CONTEXT_TOKENS` | 长会话设 12000，短任务可设 0 |
| `MULTI_AGENT_AUDIT_LOG` | 合规场景必须开 |

## 5. 调用

```python
from multi_agent.graph import build_graph, run_task_full, pending_approval, resume_task

app = build_graph(registry=registry)

result = run_task_full("把 SKU-1 上架", registry=registry, app=app)
if result.pending:
    result = resume_task(result.thread_id, {"approved": True, "note": "复核通过"}, app=app)

print(result.answer, result.usage)
```

批量任务：

```python
from multi_agent.graph import run_parallel

results = run_parallel(["上架 SKU-1", "上架 SKU-2", "上架 SKU-3"], registry=registry, max_workers=3)
```

## 上线检查清单

- [ ] 每个专家的 `description` 写的是“何时选它”
- [ ] 高风险动作都标了 `requires_approval=True`
- [ ] 审计日志路径在持久化目录，且已接入日志收集
- [ ] 检查点数据库有备份策略
- [ ] supervisor 用小模型跑过一轮完整业务，确认路由正确
- [ ] 对目标中转跑过 `python tools/compat_probe.py`
