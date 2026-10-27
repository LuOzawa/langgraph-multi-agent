"""演示：在默认团队上再注册一个专家。需要有效的 .env。"""

from multi_agent.graph import run_task
from multi_agent.registry import AgentSpec, default_registry
from multi_agent.tools.workspace import workspace_tools


def main() -> None:
    registry = default_registry()
    registry.register(
        AgentSpec(
            name="summarizer",
            description="把最终结果压成条目清单。",
            system_prompt="你是总结员。只输出条目，每条一行。",
            tools_factory=workspace_tools,
        )
    )
    print(run_task("解释什么是多 Agent，并给三条扩展建议", registry=registry))


if __name__ == "__main__":
    main()
