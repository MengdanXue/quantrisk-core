"""
专业回测报告生成器

生成多种格式的回测报告：
- HTML报告（带图表）
- PDF报告
- JSON数据
- Excel报告
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
import base64
from io import BytesIO

from .engine import BacktestResult, Trade
from .portfolio import PortfolioResult
from .monte_carlo import MonteCarloResult
from .walk_forward import WFOResult


@dataclass
class ReportConfig:
    """报告配置"""
    title: str = "策略回测报告"
    author: str = "TraderX"
    include_trades: bool = True
    max_trades_display: int = 100
    include_charts: bool = True
    include_monte_carlo: bool = True
    include_wfo: bool = True
    theme: str = "professional"  # professional, dark, light
    language: str = "zh"  # zh, en


class ReportGenerator:
    """
    回测报告生成器

    支持多种输出格式和自定义配置
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()

    def generate_html(
        self,
        result: Union[BacktestResult, PortfolioResult],
        monte_carlo: Optional[MonteCarloResult] = None,
        wfo: Optional[WFOResult] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成HTML报告

        Args:
            result: 回测结果
            monte_carlo: 蒙特卡洛结果（可选）
            wfo: Walk-Forward结果（可选）
            output_path: 输出路径（可选）

        Returns:
            HTML字符串
        """
        html = self._build_html_report(result, monte_carlo, wfo)

        if output_path:
            Path(output_path).write_text(html, encoding='utf-8')

        return html

    def generate_json(
        self,
        result: Union[BacktestResult, PortfolioResult],
        monte_carlo: Optional[MonteCarloResult] = None,
        wfo: Optional[WFOResult] = None
    ) -> Dict[str, Any]:
        """生成JSON报告数据"""
        report_data = {
            'metadata': {
                'title': self.config.title,
                'author': self.config.author,
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'summary': self._generate_summary(result),
            'metrics': result.metrics,
            'equity_curve': {
                'dates': [str(d) for d in result.equity_curve.index],
                'values': result.equity_curve.tolist()
            },
            'drawdown': {
                'dates': [str(d) for d in result.drawdown_curve.index],
                'values': result.drawdown_curve.tolist()
            },
            'returns': {
                'daily': self._series_to_dict(result.daily_returns),
                'monthly': result.monthly_returns.to_dict() if hasattr(result, 'monthly_returns') else {},
                'yearly': self._series_to_dict(result.yearly_returns) if hasattr(result, 'yearly_returns') else {}
            }
        }

        if self.config.include_trades and hasattr(result, 'trades'):
            report_data['trades'] = [
                self._trade_to_dict(t)
                for t in result.trades[:self.config.max_trades_display]
            ]

        if monte_carlo and self.config.include_monte_carlo:
            report_data['monte_carlo'] = self._monte_carlo_to_dict(monte_carlo)

        if wfo and self.config.include_wfo:
            report_data['walk_forward'] = self._wfo_to_dict(wfo)

        return report_data

    def generate_excel(
        self,
        result: Union[BacktestResult, PortfolioResult],
        output_path: str,
        monte_carlo: Optional[MonteCarloResult] = None,
        wfo: Optional[WFOResult] = None
    ):
        """生成Excel报告"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 概要页
            summary_df = pd.DataFrame([self._generate_summary(result)])
            summary_df.to_excel(writer, sheet_name='概要', index=False)

            # 绩效指标
            metrics_df = pd.DataFrame([result.metrics])
            metrics_df.to_excel(writer, sheet_name='绩效指标', index=False)

            # 权益曲线
            equity_df = pd.DataFrame({
                '日期': result.equity_curve.index,
                '权益': result.equity_curve.values,
                '回撤': result.drawdown_curve.values
            })
            equity_df.to_excel(writer, sheet_name='权益曲线', index=False)

            # 日收益
            returns_df = pd.DataFrame({
                '日期': result.daily_returns.index,
                '日收益率': result.daily_returns.values
            })
            returns_df.to_excel(writer, sheet_name='日收益', index=False)

            # 月度收益
            if hasattr(result, 'monthly_returns'):
                result.monthly_returns.to_excel(writer, sheet_name='月度收益')

            # 交易记录
            if self.config.include_trades and hasattr(result, 'trades'):
                trades_df = pd.DataFrame([
                    self._trade_to_dict(t)
                    for t in result.trades[:self.config.max_trades_display]
                ])
                trades_df.to_excel(writer, sheet_name='交易记录', index=False)

            # 蒙特卡洛
            if monte_carlo and self.config.include_monte_carlo:
                mc_df = pd.DataFrame({
                    '指标': ['平均收益', '中位收益', '收益标准差', '亏损概率', '破产概率'],
                    '值': [
                        monte_carlo.mean_return,
                        monte_carlo.median_return,
                        monte_carlo.std_return,
                        monte_carlo.probability_of_loss,
                        monte_carlo.probability_of_ruin
                    ]
                })
                mc_df.to_excel(writer, sheet_name='蒙特卡洛', index=False)

            # Walk-Forward
            if wfo and self.config.include_wfo:
                wfo_df = pd.DataFrame(wfo.fold_details)
                wfo_df.to_excel(writer, sheet_name='Walk-Forward', index=False)

    def _build_html_report(
        self,
        result: Union[BacktestResult, PortfolioResult],
        monte_carlo: Optional[MonteCarloResult],
        wfo: Optional[WFOResult]
    ) -> str:
        """构建HTML报告"""
        theme = self._get_theme_styles()
        summary = self._generate_summary(result)

        # 生成图表数据
        equity_chart = self._generate_equity_chart_data(result)
        drawdown_chart = self._generate_drawdown_chart_data(result)
        monthly_returns_chart = self._generate_monthly_returns_chart(result)

        html = f"""
<!DOCTYPE html>
<html lang="{self.config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {theme}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            color: var(--primary-color);
        }}
        .header .meta {{
            color: var(--text-secondary);
            margin-top: 10px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: var(--primary-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: var(--text-secondary);
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: var(--text-primary);
        }}
        .metric-card .value.positive {{
            color: var(--success-color);
        }}
        .metric-card .value.negative {{
            color: var(--danger-color);
        }}
        .chart-container {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
        }}
        .trades-table th, .trades-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        .trades-table th {{
            background: var(--primary-color);
            color: white;
        }}
        .trades-table tr:hover {{
            background: var(--hover-bg);
        }}
        .profit {{
            color: var(--success-color);
        }}
        .loss {{
            color: var(--danger-color);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.title}</h1>
        <div class="meta">
            作者: {self.config.author} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>

    <div class="section">
        <h2>📊 绩效概要</h2>
        <div class="metrics-grid">
            {self._generate_metric_cards(result.metrics)}
        </div>
    </div>

    <div class="section">
        <h2>📈 权益曲线</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <div class="section">
        <h2>📉 回撤分析</h2>
        <div class="chart-container">
            <canvas id="drawdownChart"></canvas>
        </div>
    </div>

    <div class="section">
        <h2>📅 月度收益</h2>
        <div class="chart-container">
            <canvas id="monthlyChart"></canvas>
        </div>
    </div>

    {self._generate_trades_section(result) if self.config.include_trades else ''}

    {self._generate_monte_carlo_section(monte_carlo) if monte_carlo else ''}

    {self._generate_wfo_section(wfo) if wfo else ''}

    <script>
        // 权益曲线图
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {equity_chart},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: '权益曲线' }}
                }},
                scales: {{
                    y: {{ beginAtZero: false }}
                }}
            }}
        }});

        // 回撤图
        new Chart(document.getElementById('drawdownChart'), {{
            type: 'line',
            data: {drawdown_chart},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: '回撤曲线' }}
                }},
                scales: {{
                    y: {{ max: 0 }}
                }}
            }}
        }});

        // 月度收益图
        new Chart(document.getElementById('monthlyChart'), {{
            type: 'bar',
            data: {monthly_returns_chart},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: '月度收益率' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html

    def _get_theme_styles(self) -> str:
        """获取主题样式"""
        if self.config.theme == 'dark':
            return """
                :root {
                    --primary-color: #4dabf7;
                    --success-color: #51cf66;
                    --danger-color: #ff6b6b;
                    --warning-color: #fcc419;
                    --text-primary: #f8f9fa;
                    --text-secondary: #adb5bd;
                    --card-bg: #343a40;
                    --border-color: #495057;
                    --hover-bg: #495057;
                    --body-bg: #212529;
                }
                body { background: var(--body-bg); color: var(--text-primary); }
            """
        else:
            return """
                :root {
                    --primary-color: #228be6;
                    --success-color: #40c057;
                    --danger-color: #fa5252;
                    --warning-color: #fab005;
                    --text-primary: #212529;
                    --text-secondary: #868e96;
                    --card-bg: #ffffff;
                    --border-color: #dee2e6;
                    --hover-bg: #f8f9fa;
                    --body-bg: #f8f9fa;
                }
                body { background: var(--body-bg); }
            """

    def _generate_summary(self, result: Union[BacktestResult, PortfolioResult]) -> Dict[str, Any]:
        """生成摘要信息"""
        m = result.metrics
        return {
            '初始资金': f"${m.get('initial_capital', 0):,.0f}",
            '最终资金': f"${m.get('final_capital', 0):,.0f}",
            '总收益率': f"{m.get('total_return', 0)*100:.2f}%",
            '年化收益率': f"{m.get('annual_return', 0)*100:.2f}%",
            '夏普比率': f"{m.get('sharpe_ratio', 0):.2f}",
            '最大回撤': f"{m.get('max_drawdown', 0)*100:.2f}%",
            '胜率': f"{m.get('win_rate', 0)*100:.1f}%",
            '盈亏比': f"{m.get('profit_factor', 0):.2f}",
            '总交易次数': m.get('total_trades', 0)
        }

    def _generate_metric_cards(self, metrics: Dict[str, float]) -> str:
        """生成指标卡片HTML"""
        cards = []
        key_metrics = [
            ('total_return', '总收益率', True, True),
            ('annual_return', '年化收益率', True, True),
            ('sharpe_ratio', '夏普比率', False, True),
            ('sortino_ratio', '索提诺比率', False, True),
            ('max_drawdown', '最大回撤', True, False),
            ('volatility', '年化波动率', True, False),
            ('win_rate', '胜率', True, True),
            ('profit_factor', '盈亏比', False, True),
            ('total_trades', '总交易次数', False, False),
            ('avg_trade', '平均盈亏', False, True),
        ]

        for key, label, is_pct, positive_is_good in key_metrics:
            if key in metrics:
                value = metrics[key]
                if is_pct:
                    display_value = f"{value * 100:.2f}%"
                elif isinstance(value, float):
                    display_value = f"{value:.2f}"
                else:
                    display_value = str(value)

                if positive_is_good:
                    value_class = 'positive' if value > 0 else 'negative' if value < 0 else ''
                else:
                    value_class = 'negative' if value < -0.1 else ''

                cards.append(f"""
                    <div class="metric-card">
                        <div class="label">{label}</div>
                        <div class="value {value_class}">{display_value}</div>
                    </div>
                """)

        return '\n'.join(cards)

    def _generate_equity_chart_data(self, result) -> str:
        """生成权益曲线图数据"""
        dates = [str(d)[:10] for d in result.equity_curve.index]
        values = result.equity_curve.tolist()

        # 下采样（如果数据点过多）
        if len(dates) > 500:
            step = len(dates) // 500
            dates = dates[::step]
            values = values[::step]

        return json.dumps({
            'labels': dates,
            'datasets': [{
                'label': '权益',
                'data': values,
                'borderColor': 'rgb(75, 192, 192)',
                'fill': False,
                'tension': 0.1
            }]
        })

    def _generate_drawdown_chart_data(self, result) -> str:
        """生成回撤图数据"""
        dates = [str(d)[:10] for d in result.drawdown_curve.index]
        values = [v * 100 for v in result.drawdown_curve.tolist()]  # 转为百分比

        if len(dates) > 500:
            step = len(dates) // 500
            dates = dates[::step]
            values = values[::step]

        return json.dumps({
            'labels': dates,
            'datasets': [{
                'label': '回撤 (%)',
                'data': values,
                'borderColor': 'rgb(255, 99, 132)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'fill': True,
                'tension': 0.1
            }]
        })

    def _generate_monthly_returns_chart(self, result) -> str:
        """生成月度收益图数据"""
        if not hasattr(result, 'monthly_returns') or result.monthly_returns.empty:
            return json.dumps({'labels': [], 'datasets': []})

        monthly = result.monthly_returns['return'] if 'return' in result.monthly_returns.columns else result.monthly_returns.iloc[:, 0]
        dates = [str(d)[:7] for d in monthly.index]
        values = [v * 100 for v in monthly.tolist()]

        colors = ['rgba(75, 192, 192, 0.8)' if v >= 0 else 'rgba(255, 99, 132, 0.8)' for v in values]

        return json.dumps({
            'labels': dates,
            'datasets': [{
                'label': '月度收益 (%)',
                'data': values,
                'backgroundColor': colors
            }]
        })

    def _generate_trades_section(self, result) -> str:
        """生成交易记录部分"""
        if not hasattr(result, 'trades') or not result.trades:
            return ''

        trades_html = []
        for trade in result.trades[:self.config.max_trades_display]:
            pnl_class = 'profit' if trade.pnl > 0 else 'loss'
            trades_html.append(f"""
                <tr>
                    <td>{trade.trade_id}</td>
                    <td>{trade.symbol}</td>
                    <td>{trade.side}</td>
                    <td>{trade.quantity:.2f}</td>
                    <td>${trade.entry_price:.2f}</td>
                    <td>${trade.exit_price:.2f}</td>
                    <td class="{pnl_class}">${trade.pnl:.2f}</td>
                    <td class="{pnl_class}">{trade.pnl_pct*100:.2f}%</td>
                    <td>{trade.holding_period}天</td>
                </tr>
            """)

        return f"""
        <div class="section">
            <h2>📝 交易记录</h2>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>交易ID</th>
                        <th>标的</th>
                        <th>方向</th>
                        <th>数量</th>
                        <th>入场价</th>
                        <th>出场价</th>
                        <th>盈亏</th>
                        <th>盈亏%</th>
                        <th>持仓时间</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(trades_html)}
                </tbody>
            </table>
        </div>
        """

    def _generate_monte_carlo_section(self, mc: MonteCarloResult) -> str:
        """生成蒙特卡洛分析部分"""
        return f"""
        <div class="section">
            <h2>🎲 蒙特卡洛分析</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">平均收益率</div>
                    <div class="value {'positive' if mc.mean_return > 0 else 'negative'}">{mc.mean_return*100:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">中位收益率</div>
                    <div class="value">{mc.median_return*100:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">亏损概率</div>
                    <div class="value negative">{mc.probability_of_loss*100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">破产概率 (>50%亏损)</div>
                    <div class="value negative">{mc.probability_of_ruin*100:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">95% VaR</div>
                    <div class="value negative">{mc.var_estimates.get(0.95, 0)*100:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">95% CVaR</div>
                    <div class="value negative">{mc.cvar_estimates.get(0.95, 0)*100:.2f}%</div>
                </div>
            </div>
        </div>
        """

    def _generate_wfo_section(self, wfo: WFOResult) -> str:
        """生成Walk-Forward分析部分"""
        return f"""
        <div class="section">
            <h2>🔄 Walk-Forward 优化</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">综合夏普比率</div>
                    <div class="value">{wfo.combined_metrics.get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">稳健性比率</div>
                    <div class="value {'positive' if wfo.robustness_ratio > 0.5 else 'negative'}">{wfo.robustness_ratio:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">折数</div>
                    <div class="value">{len(wfo.fold_details)}</div>
                </div>
                <div class="metric-card">
                    <div class="label">综合收益率</div>
                    <div class="value">{wfo.combined_metrics.get('total_return', 0)*100:.2f}%</div>
                </div>
            </div>
        </div>
        """

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """将Trade转换为字典"""
        return {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'side': trade.side,
            'quantity': trade.quantity,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'entry_time': str(trade.entry_time),
            'exit_time': str(trade.exit_time),
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'commission': trade.commission,
            'holding_period': trade.holding_period
        }

    def _series_to_dict(self, series: pd.Series) -> Dict[str, float]:
        """Series转字典"""
        return {str(k): float(v) for k, v in series.items()}

    def _monte_carlo_to_dict(self, mc: MonteCarloResult) -> Dict[str, Any]:
        """蒙特卡洛结果转字典"""
        return {
            'mean_return': mc.mean_return,
            'median_return': mc.median_return,
            'std_return': mc.std_return,
            'probability_of_loss': mc.probability_of_loss,
            'probability_of_ruin': mc.probability_of_ruin,
            'var_estimates': mc.var_estimates,
            'cvar_estimates': mc.cvar_estimates,
            'return_confidence_intervals': {
                str(k): list(v) for k, v in mc.return_confidence_intervals.items()
            }
        }

    def _wfo_to_dict(self, wfo: WFOResult) -> Dict[str, Any]:
        """WFO结果转字典"""
        return {
            'robustness_ratio': wfo.robustness_ratio,
            'combined_metrics': wfo.combined_metrics,
            'parameter_stability': wfo.parameter_stability,
            'fold_details': wfo.fold_details
        }
