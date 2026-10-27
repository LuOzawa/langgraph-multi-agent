from multi_agent.tools.ecommerce import calc_landed_cost


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
