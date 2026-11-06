"""
DCF (Discounted Cash Flow) 估值计算器
完整实现，包含结果类和详细分析
"""
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    """估值结果数据类"""
    intrinsic_value: float
    current_price: Optional[float]
    upside_potential: float
    margin_of_safety: float
    recommendation: str
    details: dict


class DCFCalculator:
    """DCF估值计算器 - 完整实现"""

    def __init__(self):
        pass

    def calculate_dcf(
        self,
        free_cash_flows: List[float],
        wacc: float,
        terminal_growth_rate: float,
        current_price: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
        net_debt: float = 0
    ) -> ValuationResult:
        """
        计算DCF估值

        Args:
            free_cash_flows: 未来自由现金流预测列表（通常5-10年）
            wacc: 加权平均资本成本（WACC）
            terminal_growth_rate: 永续增长率
            current_price: 当前股价（可选）
            shares_outstanding: 流通股数（可选）
            net_debt: 净债务（默认0）

        Returns:
            ValuationResult: 估值结果对象
        """
        try:
            logger.info(f"计算DCF估值: wacc={wacc}, terminal_growth={terminal_growth_rate}")

            # 验证输入
            if not free_cash_flows or len(free_cash_flows) == 0:
                raise ValueError("自由现金流预测不能为空")

            if wacc <= 0 or wacc >= 1:
                raise ValueError("WACC必须在0-1之间")

            if terminal_growth_rate < 0 or terminal_growth_rate >= wacc:
                raise ValueError("永续增长率必须小于WACC且非负")

            # 计算自由现金流现值
            pv_fcf = []
            total_pv_fcf = 0.0

            for i, fcf in enumerate(free_cash_flows, 1):
                pv = fcf / ((1 + wacc) ** i)
                pv_fcf.append({
                    "year": i,
                    "fcf": round(fcf, 2),
                    "discount_factor": round((1 + wacc) ** i, 4),
                    "pv": round(pv, 2)
                })
                total_pv_fcf += pv

            # 计算终值（Terminal Value）
            last_fcf = free_cash_flows[-1]
            terminal_fcf = last_fcf * (1 + terminal_growth_rate)
            terminal_value = terminal_fcf / (wacc - terminal_growth_rate)

            # 终值的现值
            n_years = len(free_cash_flows)
            pv_terminal_value = terminal_value / ((1 + wacc) ** n_years)

            # 企业价值（Enterprise Value）
            enterprise_value = total_pv_fcf + pv_terminal_value

            # 股权价值（Equity Value）= 企业价值 - 净债务
            equity_value = enterprise_value - net_debt

            # 计算每股内在价值
            if shares_outstanding and shares_outstanding > 0:
                intrinsic_value = equity_value / shares_outstanding
            else:
                # 如果没有提供流通股数，返回股权价值作为内在价值
                intrinsic_value = equity_value

            # 计算估值指标
            upside_potential = 0.0
            margin_of_safety = 0.0
            recommendation = "N/A"

            if current_price and current_price > 0:
                upside_potential = ((intrinsic_value - current_price) / current_price) * 100
                margin_of_safety = ((intrinsic_value - current_price) / intrinsic_value) * 100

                # 估值建议
                if margin_of_safety >= 30:
                    recommendation = "强烈买入"
                elif margin_of_safety >= 15:
                    recommendation = "买入"
                elif margin_of_safety >= -15:
                    recommendation = "持有"
                elif margin_of_safety >= -30:
                    recommendation = "卖出"
                else:
                    recommendation = "强烈卖出"

            # 构建详细信息
            details = {
                "enterprise_value": round(enterprise_value, 2),
                "equity_value": round(equity_value, 2),
                "net_debt": round(net_debt, 2),
                "total_pv_fcf": round(total_pv_fcf, 2),
                "pv_terminal_value": round(pv_terminal_value, 2),
                "terminal_value": round(terminal_value, 2),
                "terminal_fcf": round(terminal_fcf, 2),
                "wacc": wacc,
                "terminal_growth_rate": terminal_growth_rate,
                "projection_years": n_years,
                "shares_outstanding": shares_outstanding,
                "cash_flows": pv_fcf,
                "terminal_value_percentage": round((pv_terminal_value / enterprise_value) * 100, 1) if enterprise_value > 0 else 0
            }

            result = ValuationResult(
                intrinsic_value=round(intrinsic_value, 2),
                current_price=current_price,
                upside_potential=round(upside_potential, 2),
                margin_of_safety=round(margin_of_safety, 2),
                recommendation=recommendation,
                details=details
            )

            logger.info(f"DCF估值完成: intrinsic_value={intrinsic_value:.2f}, recommendation={recommendation}")
            return result

        except Exception as e:
            logger.error(f"DCF估值计算失败: {str(e)}")
            raise