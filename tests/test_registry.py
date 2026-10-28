from multi_agent.registry import AgentRegistry, AgentSpec, _thinking_off_body


def test_thinking_disabled_only_for_deepseek():
    assert _thinking_off_body("https://api.deepseek.com/v1", "deepseek-v4-flash") == {
        "thinking": {"type": "disabled"}
    }
    assert _thinking_off_body("https://api.example.com/v1", "deepseek-v4-flash") == {
        "thinking": {"type": "disabled"}
    }
    assert _thinking_off_body("https://api.deepseek.com/v1", "some-model") == {
        "thinking": {"type": "disabled"}
    }
    assert _thinking_off_body("https://api.openai.com/v1", "gpt-4o-mini") is None


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
