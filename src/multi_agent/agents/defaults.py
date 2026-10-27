from __future__ import annotations

from multi_agent.registry import AgentRegistry, AgentSpec
from multi_agent.tools.workspace import workspace_tools


def register_default_agents(registry: AgentRegistry) -> None:
    """示例专家。业务接入时用 registry.register 增加，或替换这一批。"""

    registry.register(
        AgentSpec(
            name="researcher",
            description="拆解问题、列出已知事实、缺口和待核验点，并把要点存成笔记。",
            system_prompt=(
                "你是研究员。只做调研与结构化要点，不要写最终成稿。\n"
                "输出：事实、假设、未知项、建议下一步交给谁。\n"
                "需要沉淀时用 save_note。"
            ),
            tools_factory=workspace_tools,
        )
    )
    registry.register(
        AgentSpec(
            name="writer",
            description="根据研究员笔记和对话，写出完整答复或草稿。",
            system_prompt=(
                "你是写手。基于已有调研写清楚、可直接交付的正文。\n"
                "不要编造未出现的事实。可用 list_notes / read_note 读取笔记。"
            ),
            tools_factory=workspace_tools,
        )
    )
    registry.register(
        AgentSpec(
            name="reviewer",
            description="检查草稿是否完整、是否有幻觉或遗漏，给出通过或返工意见。",
            system_prompt=(
                "你是审查员。检查遗漏、前后矛盾和空泛表述。\n"
                "若可通过，明确写「审查通过」并给最终修订稿；\n"
                "若需返工，写清问题与应交给 researcher 还是 writer。"
            ),
            tools_factory=workspace_tools,
        )
    )
