from __future__ import annotations

from langchain_core.tools import tool


@tool
def calc_landed_cost(
    fob_usd: float,
    freight_usd: float,
    duty_rate: float,
    platform_fee_rate: float,
    target_margin: float = 0.3,
) -> str:
    """计算跨境商品的到岸成本与建议售价（美元）。

    参数均为小数比例，例如 duty_rate=0.12 表示 12%。
    返回成本拆分与满足目标毛利率的建议售价。
    """
    if fob_usd <= 0:
        return "FOB 价必须大于 0"
    if not 0 <= duty_rate < 1 or not 0 <= platform_fee_rate < 1:
        return "税率与平台佣金率必须在 [0, 1) 之间"
    if not 0 < target_margin < 1:
        return "目标毛利率必须在 (0, 1) 之间"

    duty = fob_usd * duty_rate
    landed = fob_usd + freight_usd + duty
    # 售价 P 满足：P - P*platform_fee - landed = P*target_margin
    denominator = 1 - platform_fee_rate - target_margin
    if denominator <= 0:
        return "平台佣金与目标毛利之和不能大于等于 1，无法定价"
    price = landed / denominator
    profit = price * target_margin

    return (
        f"到岸成本 ${landed:.2f}（FOB ${fob_usd:.2f} + 运费 ${freight_usd:.2f}"
        f" + 关税 ${duty:.2f}）\n"
        f"平台佣金 ${price * platform_fee_rate:.2f}（{platform_fee_rate:.0%}）\n"
        f"建议售价 ${price:.2f}，毛利 ${profit:.2f}（{target_margin:.0%}）"
    )
