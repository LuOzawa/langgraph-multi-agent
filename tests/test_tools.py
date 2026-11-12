from multi_agent.tools.ecommerce import calc_landed_cost
from multi_agent.tools.workspace import MAX_NOTE_CHARS, workspace_tools


def _workspace_tools() -> dict:
    return {item.name: item for item in workspace_tools()}


def test_workspace_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_AGENT_NOTES_DIR", str(tmp_path))
    tools = _workspace_tools()

    tools["save_note"].invoke({"title": "周报", "content": "本周完成 3 项"})

    assert "本周完成 3 项" in tools["read_note"].invoke({"title": "周报"})
    assert "周报" in tools["list_notes"].invoke({})


def test_read_note_truncates_oversized_note(tmp_path, monkeypatch):
    """笔记目录可能是用户手放的，超大文件不能整份灌进模型上下文。"""
    monkeypatch.setenv("MULTI_AGENT_NOTES_DIR", str(tmp_path))
    tools = _workspace_tools()
    (tmp_path / "big.md").write_text("啊" * (MAX_NOTE_CHARS + 100), encoding="utf-8")

    out = tools["read_note"].invoke({"title": "big"})

    assert "已截断" in out
    assert len(out) <= MAX_NOTE_CHARS + 200


def test_landed_cost_math():
    out = calc_landed_cost.invoke(
        {
            "fob_usd": 10,
            "freight_usd": 2,
            "duty_rate": 0.1,
            "platform_fee_rate": 0.15,
            "target_margin": 0.3,
        }
    )
    # landed = 10 + 2 + 1 = 13; price = 13 / (1 - 0.15 - 0.3) = 23.636
    assert "到岸成本 $13.00" in out
    assert "建议售价 $23.64" in out


def test_landed_cost_rejects_bad_input():
    assert "必须大于 0" in calc_landed_cost.invoke(
        {"fob_usd": 0, "freight_usd": 1, "duty_rate": 0.1, "platform_fee_rate": 0.15}
    )
    assert "无法定价" in calc_landed_cost.invoke(
        {
            "fob_usd": 10,
            "freight_usd": 1,
            "duty_rate": 0.1,
            "platform_fee_rate": 0.6,
            "target_margin": 0.5,
        }
    )


def test_landed_cost_with_vat_returns_and_ads():
    out = calc_landed_cost.invoke(
        {
            "fob_usd": 10,
            "freight_usd": 2,
            "duty_rate": 0.1,
            "platform_fee_rate": 0.15,
            "vat_rate": 0.2,
            "return_rate": 0.05,
            "ad_rate": 0.1,
            "target_margin": 0.3,
        }
    )
    assert "退货成本" in out
    assert "增值税" in out
    assert "广告费" in out
    assert "建议售价 $54.60" in out
