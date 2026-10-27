from __future__ import annotations

from langchain_core.tools import tool


@tool
def calc_landed_cost(
    fob_usd: float,
    freight_usd: float,
    duty_rate: float,
    platform_fee_rate: float,
    target_margin: float = 0.3,
    vat_rate: float = 0.0,
    return_rate: float = 0.0,
    ad_rate: float = 0.0,
) -> str:
    """计算跨境商品的到岸成本与建议售价（美元）。

    参数均为小数比例，例如 duty_rate=0.12 表示 12%。
    vat_rate 为售价内增值税，return_rate 为退货导致的成本上浮，
    ad_rate 为按售价计提的广告费。返回成本拆分与满足目标毛利率的建议售价。
    """
    if fob_usd <= 0:
        return "FOB 价必须大于 0"
    if not 0 <= duty_rate < 1 or not 0 <= platform_fee_rate < 1:
        return "税率与平台佣金率必须在 [0, 1) 之间"
    if not 0 < target_margin < 1:
        return "目标毛利率必须在 (0, 1) 之间"
    for label, value in (
        ("vat_rate", vat_rate),
        ("return_rate", return_rate),
        ("ad_rate", ad_rate),
    ):
        if not 0 <= value < 1:
            return f"{label} 必须在 [0, 1) 之间"

    duty = fob_usd * duty_rate
    landed = fob_usd + freight_usd + duty
    effective_cost = landed * (1 + return_rate)
    # 售价 P 满足：P - P*(佣金+增值税+广告) - effective_cost = P*毛利率
    denominator = 1 - platform_fee_rate - vat_rate - ad_rate - target_margin
    if denominator <= 0:
        return "佣金、税费、广告与目标毛利之和不能大于等于 1，无法定价"
    price = effective_cost / denominator
    profit = price * target_margin

    lines = [
        f"到岸成本 ${landed:.2f}（FOB ${fob_usd:.2f} + 运费 ${freight_usd:.2f}"
        f" + 关税 ${duty:.2f}）"
    ]
    if return_rate:
        lines.append(f"退货成本 ${landed * return_rate:.2f}（退货率 {return_rate:.0%}）")
    lines.append(f"平台佣金 ${price * platform_fee_rate:.2f}（{platform_fee_rate:.0%}）")
    if vat_rate:
        lines.append(f"增值税 ${price * vat_rate:.2f}（{vat_rate:.0%}）")
    if ad_rate:
        lines.append(f"广告费 ${price * ad_rate:.2f}（{ad_rate:.0%}）")
    lines.append(f"建议售价 ${price:.2f}，毛利 ${profit:.2f}（{target_margin:.0%}）")
    return "\n".join(lines)
