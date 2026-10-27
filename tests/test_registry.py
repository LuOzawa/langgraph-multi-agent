from multi_agent.registry import AgentRegistry, AgentSpec


def test_register_and_list():
    registry = AgentRegistry()
    registry.register(
        AgentSpec(name="alpha", description="A", system_prompt="you are alpha")
    )
    assert registry.names() == ["alpha"]
    assert "alpha" in registry.descriptions_block()


def test_reject_duplicate_and_reserved():
    registry = AgentRegistry()
    spec = AgentSpec(name="alpha", description="A", system_prompt="x")
    registry.register(spec)
    try:
        registry.register(spec)
        raise AssertionError("should reject duplicate")
    except ValueError:
        pass
    try:
        registry.register(AgentSpec(name="supervisor", description="x", system_prompt="x"))
        raise AssertionError("should reject reserved name")
    except ValueError:
        pass
