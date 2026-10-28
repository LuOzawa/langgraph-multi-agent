"""跨境电商团队示例。

角色分工：
- sourcing    选品与供应商比价
- listing     英文标题 / 五点描述 / 关键词
- pricing     到岸成本与定价（带 calc_landed_cost 工具）
- compliance  合规与禁售词、认证要求
- publisher   上架发布（requires_approval=True，必须人工批准）

运行（需要 .env 里的 OPENAI_API_KEY）：
    python examples/ecommerce_team.py "帮我把这款收纳盒上架到亚马逊美国站"
"""

from __future__ import annotations

import sys

from multi_agent.graph import build_graph, run_task_full
from multi_agent.registry import AgentRegistry, AgentSpec
from multi_agent.tools.ecommerce import calc_landed_cost


def build_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="sourcing",
            description="选品与供应商比价：给候选、成本区间与风险。",
            system_prompt=(
                "你是跨境选品研究员。输出候选供应商类型、FOB 价格区间、MOQ、交期与风险点。"
                "不要编造具体供应商，只给可核验的筛选维度。"
            ),
            role="researcher",
        )
    )
    registry.register(
        AgentSpec(
            name="listing",
            description="写亚马逊英文 listing：标题、五点描述、搜索词。",
            system_prompt=(
                "你是亚马逊 listing 文案专家。输出英文标题（不超过 200 字符）、5 条 bullet、"
                "以及 10 个搜索词。避免夸大与医疗宣称。"
            ),
            role="writer",
        )
    )
    registry.register(
        AgentSpec(
            name="pricing",
            description="测算到岸成本、平台佣金与建议售价。",
            system_prompt=(
                "你是定价分析师。必须调用 calc_landed_cost 计算，"
                "然后给出定价区间与毛利敏感性说明。"
            ),
            tools_factory=lambda: [calc_landed_cost],
            role="researcher",
        )
    )
    registry.register(
        AgentSpec(
            name="compliance",
            description="检查禁售词、认证与目的国合规要求。",
            system_prompt=(
                "你是合规审查员。列出该品类在美国站需要的认证、标签与高风险表述，"
                "并明确指出不能上架的情形。"
            ),
            role="reviewer",
        )
    )
    registry.register(
        AgentSpec(
            name="publisher",
            description="执行上架发布（高风险，需要人工审批）。",
            system_prompt="你是发布执行员。只输出最终提交内容摘要与状态，不要改写文案。",
            requires_approval=True,
            role="executor",
        )
    )
    return registry


def main() -> int:
    task = " ".join(sys.argv[1:]).strip() or "帮我把这款收纳盒上架到亚马逊美国站"
    registry = build_registry()
    app = build_graph(registry=registry)

    def approver(payload: dict) -> dict:
        print(f"\n[需要审批] {payload.get('agent')}：{payload.get('reason')}")
        print(f"任务：{str(payload.get('task', ''))[:160]}")
        try:
            answer = input("批准执行？[y/N] ").strip().lower()
        except EOFError:
            print("（非交互环境，默认拒绝）")
            return {"approved": False, "note": "非交互环境，默认拒绝"}
        approved = answer in {"y", "yes"}
        note = ""
        if approved:
            try:
                note = input("备注（可空）: ").strip()
            except EOFError:
                note = ""
        return {"approved": approved, "note": note}

    result = run_task_full(task, registry=registry, app=app, approver=approver)
    print(f"\n{result.answer}")
    print(f"\nthread_id={result.thread_id} | turns={result.turns} | stop={result.stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
