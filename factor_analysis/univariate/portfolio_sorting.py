"""
分组回测分析器

将股票按因子值分组，计算各组的收益表现，直观展示因子的区分能力。

核心功能：
- 按因子值分位数分组
- 计算各组等权收益
- Long-Short组合构建
- 性能指标：年化收益、夏普、最大回撤、换手率
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
import matplotlib.pyplot as plt

from ..core import FactorData, AnalysisResult
from ..utils.data_prep import compute_forward_returns
from ..utils.metrics import compute_returns_metrics, compute_max_drawdown


class PortfolioSorter:
    """
    分组回测分析器

    按因子值分组，计算各组收益表现，评估因子的区分能力。

    Attributes:
        n_quantiles: 分组数量（通常为5或10）
        holding_period: 持有期（天数）
        rebalance_freq: 调仓频率
        long_short: 是否构建Long-Short组合
    """

    def __init__(self,
                 n_quantiles: int = 5,
                 holding_period: int = 1,
                 rebalance_freq: str = 'daily',
                 long_short: bool = True):
        """
        初始化分组回测器

        Args:
            n_quantiles: 分组数量（5表示五分位数，10表示十分位数）
            holding_period: 持有期（天数）
            rebalance_freq: 调仓频率（'daily', 'weekly', 'monthly'）
            long_short: 是否计算Long-Short组合

        Example:
            >>> sorter = PortfolioSorter(n_quantiles=5, holding_period=1)
            >>> result = sorter.analyze(factor_data)
        """
        if n_quantiles < 2:
            raise ValueError("n_quantiles 必须 >= 2")

        self.n_quantiles = n_quantiles
        self.holding_period = holding_period
        self.rebalance_freq = rebalance_freq
        self.long_short = long_short

    def analyze(self, factor_data: FactorData) -> AnalysisResult:
        """
        执行分组回测

        Args:
            factor_data: 标准化的因子数据

        Returns:
            AnalysisResult: 包含各组收益、统计指标、可视化

        Example:
            >>> result = sorter.analyze(factor_data)
            >>> print(result.summary)
            >>> print(f"Long-Short夏普: {result.metrics['ls_sharpe']:.2f}")
            >>> result.figures['long_short_return'].show()
        """
        # 1. 按因子值分组
        quantile_labels = self._assign_quantiles(factor_data.factor_values)

        # 2. 计算各组收益
        group_returns = self._compute_group_returns(
            quantile_labels,
            factor_data.returns,
            self.holding_period
        )

        # 3. 计算Long-Short组合
        if self.long_short:
            ls_returns = self._compute_long_short(group_returns)
        else:
            ls_returns = None

        # 4. 计算统计指标
        metrics = self._compute_metrics(group_returns, ls_returns)

        # 5. 计算换手率
        turnover = self._compute_turnover(quantile_labels)

        # 6. 生成可视化
        figures = self._create_plots(
            group_returns,
            ls_returns,
            factor_data.factor_name
        )

        # 7. 生成摘要
        summary = self._generate_summary(metrics, factor_data.factor_name)

        return AnalysisResult(
            name=f'Portfolio Sorting - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'group_returns': group_returns,
                'ls_returns': ls_returns,
                'turnover': turnover,
                'quantile_labels': quantile_labels
            },
            figures=figures,
            summary=summary
        )

    def _assign_quantiles(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        """
        将因子值分配到分位数组

        在每个交易日，按因子值的横截面分组。

        Args:
            factor_values: 因子值 DataFrame (MultiIndex: datetime, asset)

        Returns:
            分组标签 DataFrame (相同索引)
        """
        quantile_labels = pd.DataFrame(
            index=factor_values.index,
            columns=['quantile']
        )

        dates = sorted(factor_values.index.get_level_values('timestamp').unique())

        for date in dates:
            try:
                # 获取该日横截面
                cross_section = factor_values.xs(date, level='timestamp')['factor_value']

                # 分组（使用pd.qcut）
                labels = pd.qcut(
                    cross_section,
                    q=self.n_quantiles,
                    labels=False,
                    duplicates='drop'  # 处理重复值
                )

                # 存储结果
                for asset in labels.index:
                    quantile_labels.loc[(date, asset), 'quantile'] = labels[asset]

            except Exception as e:
                warnings.warn(f"日期 {date} 的分组失败: {e}")
                # 分组失败时，标记为NaN
                cross_section = factor_values.xs(date, level='timestamp')
                for asset in cross_section.index:
                    quantile_labels.loc[(date, asset), 'quantile'] = np.nan

        return quantile_labels

    def _compute_group_returns(self,
                              quantile_labels: pd.DataFrame,
                              returns: pd.DataFrame,
                              holding_period: int) -> pd.DataFrame:
        """
        计算各组的收益

        Args:
            quantile_labels: 分组标签
            returns: 收益率 DataFrame
            holding_period: 持有期

        Returns:
            各组收益 DataFrame (columns=['Q1', 'Q2', ..., 'Q5'], index=datetime)
        """
        # 计算前向收益
        forward_returns = compute_forward_returns(returns, holding_period)

        # 准备结果容器
        dates = sorted(quantile_labels.index.get_level_values('timestamp').unique())
        group_returns_list = []

        for date in dates:
            try:
                # 获取该日的分组和收益
                date_labels = quantile_labels.xs(date, level='timestamp')['quantile']
                date_returns = forward_returns.xs(date, level='timestamp')['return']

                # 对齐
                aligned = pd.DataFrame({
                    'quantile': date_labels,
                    'return': date_returns
                }).dropna()

                # 计算每组的等权收益
                group_ret = aligned.groupby('quantile')['return'].mean()

                # 构建结果
                result_row = {'datetime': date}
                for q in range(self.n_quantiles):
                    if q in group_ret.index:
                        result_row[f'Q{q+1}'] = group_ret[q]
                    else:
                        result_row[f'Q{q+1}'] = np.nan

                group_returns_list.append(result_row)

            except Exception as e:
                warnings.warn(f"日期 {date} 的组收益计算失败: {e}")

        # 转换为DataFrame
        group_returns = pd.DataFrame(group_returns_list)
        group_returns = group_returns.set_index('datetime')

        return group_returns

    def _compute_long_short(self, group_returns: pd.DataFrame) -> pd.Series:
        """
        计算Long-Short组合收益

        Args:
            group_returns: 各组收益 DataFrame

        Returns:
            Long-Short收益序列 (做多最高组，做空最低组)
        """
        # 做多最高分位数组，做空最低分位数组
        high_group = f'Q{self.n_quantiles}'
        low_group = 'Q1'

        ls_returns = group_returns[high_group] - group_returns[low_group]

        return ls_returns

    def _compute_metrics(self,
                        group_returns: pd.DataFrame,
                        ls_returns: Optional[pd.Series]) -> Dict:
        """
        计算统计指标

        Args:
            group_returns: 各组收益
            ls_returns: Long-Short收益

        Returns:
            指标字典
        """
        metrics = {}

        # 各组指标
        for col in group_returns.columns:
            returns_series = group_returns[col].dropna()

            if len(returns_series) == 0:
                continue

            # 累计收益
            metrics[f'{col}_cum_return'] = (1 + returns_series).prod() - 1

            # 年化收益
            metrics[f'{col}_annual_return'] = returns_series.mean() * 252

            # 年化波动率
            metrics[f'{col}_annual_vol'] = returns_series.std() * np.sqrt(252)

            # 夏普比率
            if metrics[f'{col}_annual_vol'] > 0:
                metrics[f'{col}_sharpe'] = (metrics[f'{col}_annual_return'] /
                                           metrics[f'{col}_annual_vol'])
            else:
                metrics[f'{col}_sharpe'] = 0

            # 最大回撤
            metrics[f'{col}_max_drawdown'] = compute_max_drawdown(returns_series)

        # Long-Short指标
        if ls_returns is not None:
            ls_returns_clean = ls_returns.dropna()

            if len(ls_returns_clean) > 0:
                metrics['ls_cum_return'] = (1 + ls_returns_clean).prod() - 1
                metrics['ls_annual_return'] = ls_returns_clean.mean() * 252
                metrics['ls_annual_vol'] = ls_returns_clean.std() * np.sqrt(252)

                if metrics['ls_annual_vol'] > 0:
                    metrics['ls_sharpe'] = (metrics['ls_annual_return'] /
                                           metrics['ls_annual_vol'])
                else:
                    metrics['ls_sharpe'] = 0

                metrics['ls_max_drawdown'] = compute_max_drawdown(ls_returns_clean)

                # t统计量
                metrics['ls_t_statistic'] = (ls_returns_clean.mean() /
                                            (ls_returns_clean.std() / np.sqrt(len(ls_returns_clean))))

                # 胜率
                metrics['ls_win_rate'] = (ls_returns_clean > 0).mean()

        return metrics

    def _compute_turnover(self, quantile_labels: pd.DataFrame) -> pd.DataFrame:
        """
        计算各组换手率

        Args:
            quantile_labels: 分组标签

        Returns:
            换手率 DataFrame
        """
        dates = sorted(quantile_labels.index.get_level_values('timestamp').unique())
        turnover_list = []

        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]

            try:
                # 获取前后两期的分组
                prev_labels = quantile_labels.xs(prev_date, level='timestamp')['quantile']
                curr_labels = quantile_labels.xs(curr_date, level='timestamp')['quantile']

                # 对齐（只考虑两期都存在的资产）
                aligned = pd.DataFrame({
                    'prev': prev_labels,
                    'curr': curr_labels
                }).dropna()

                # 计算每组的换手率
                turnover_row = {'datetime': curr_date}
                for q in range(self.n_quantiles):
                    # 前期属于该组的资产
                    prev_in_group = aligned[aligned['prev'] == q].index

                    if len(prev_in_group) > 0:
                        # 计算变化比例
                        changed = aligned.loc[prev_in_group, 'prev'] != aligned.loc[prev_in_group, 'curr']
                        turnover_row[f'Q{q+1}'] = changed.mean()
                    else:
                        turnover_row[f'Q{q+1}'] = np.nan

                turnover_list.append(turnover_row)

            except Exception as e:
                warnings.warn(f"日期 {curr_date} 的换手率计算失败: {e}")

        turnover_df = pd.DataFrame(turnover_list)
        if not turnover_df.empty:
            turnover_df = turnover_df.set_index('datetime')

        return turnover_df

    def _create_plots(self,
                     group_returns: pd.DataFrame,
                     ls_returns: Optional[pd.Series],
                     factor_name: str) -> Dict:
        """
        生成可视化图表

        Args:
            group_returns: 各组收益
            ls_returns: Long-Short收益
            factor_name: 因子名称

        Returns:
            图表字典
        """
        figures = {}

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # 图1：分组累计收益曲线
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        cum_returns = (1 + group_returns).cumprod()

        for col in cum_returns.columns:
            cum_returns[col].plot(ax=ax1, label=col, linewidth=1.5)

        ax1.set_title(f'Cumulative Returns by Quantile - {factor_name}',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Cumulative Return', fontsize=10)
        ax1.legend(loc='best', ncol=min(self.n_quantiles, 5))
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
        fig1.tight_layout()
        figures['group_cumulative_returns'] = fig1

        # 图2：Long-Short收益曲线
        if ls_returns is not None:
            ls_returns_clean = ls_returns.dropna()

            if len(ls_returns_clean) > 0:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ls_cum = (1 + ls_returns_clean).cumprod()
                ls_cum.plot(ax=ax2, label='Long-Short', linewidth=2,
                          color='darkgreen', alpha=0.8)
                ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                ax2.set_title(f'Long-Short Portfolio Cumulative Return - {factor_name}',
                            fontsize=12, fontweight='bold')
                ax2.set_xlabel('Date', fontsize=10)
                ax2.set_ylabel('Cumulative Return', fontsize=10)
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3, linestyle='--')
                fig2.tight_layout()
                figures['long_short_return'] = fig2

        # 图3：各组平均收益对比（柱状图）
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        annual_returns = group_returns.mean() * 252

        colors = ['red' if x < 0 else 'steelblue' for x in annual_returns]
        annual_returns.plot(kind='bar', ax=ax3, color=colors, edgecolor='black', alpha=0.7)
        ax3.set_title(f'Annual Return by Quantile - {factor_name}',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Quantile', fontsize=10)
        ax3.set_ylabel('Annual Return', fontsize=10)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticklabels(annual_returns.index, rotation=0)
        fig3.tight_layout()
        figures['group_annual_returns'] = fig3

        return figures

    def _generate_summary(self,
                         metrics: Dict,
                         factor_name: str) -> str:
        """
        生成文字摘要

        Args:
            metrics: 指标字典
            factor_name: 因子名称

        Returns:
            格式化的摘要字符串
        """
        summary = f"""
Portfolio Sorting Analysis Summary - {factor_name}
{'=' * 60}
Configuration:
--------------
Number of Quantiles:  {self.n_quantiles}
Holding Period:       {self.holding_period} day(s)
Rebalance Frequency:  {self.rebalance_freq}

Group Performance:
------------------
"""
        for q in range(1, self.n_quantiles + 1):
            q_key = f'Q{q}'
            if f'{q_key}_cum_return' in metrics:
                summary += f"""
{q_key}:
  Cumulative Return: {metrics[f'{q_key}_cum_return']:>8.2%}
  Annual Return:     {metrics[f'{q_key}_annual_return']:>8.2%}
  Annual Vol:        {metrics[f'{q_key}_annual_vol']:>8.2%}
  Sharpe Ratio:      {metrics[f'{q_key}_sharpe']:>8.4f}
  Max Drawdown:      {metrics[f'{q_key}_max_drawdown']:>8.2%}
"""

        if 'ls_cum_return' in metrics:
            summary += f"""
Long-Short Portfolio (Q{self.n_quantiles} - Q1):
{'=' * 40}
  Cumulative Return: {metrics['ls_cum_return']:>8.2%}
  Annual Return:     {metrics['ls_annual_return']:>8.2%}
  Annual Vol:        {metrics['ls_annual_vol']:>8.2%}
  Sharpe Ratio:      {metrics['ls_sharpe']:>8.4f}
  Max Drawdown:      {metrics['ls_max_drawdown']:>8.2%}
  t-statistic:       {metrics['ls_t_statistic']:>8.4f}
  Win Rate:          {metrics['ls_win_rate']:>8.2%}

Interpretation:
---------------
"""
            # 添加解释
            if metrics['ls_sharpe'] > 1.0:
                summary += "✓ Long-Short夏普比率 > 1.0: 因子区分能力强，风险调整后收益优秀\n"
            elif metrics['ls_sharpe'] > 0.5:
                summary += "○ Long-Short夏普比率 0.5-1.0: 因子有一定的区分能力\n"
            else:
                summary += "✗ Long-Short夏普比率 < 0.5: 因子区分能力较弱\n"

            if metrics['ls_annual_return'] > 0.05:
                summary += "✓ Long-Short年化收益 > 5%: 因子有较好的超额收益能力\n"
            elif metrics['ls_annual_return'] > 0:
                summary += "○ Long-Short年化收益 > 0: 因子有正向超额收益\n"
            else:
                summary += "✗ Long-Short年化收益 < 0: 因子无超额收益能力\n"

            if abs(metrics['ls_t_statistic']) > 2.0:
                summary += "✓ |t统计量| > 2.0: 收益显著性强（95%置信度）\n"
            elif abs(metrics['ls_t_statistic']) > 1.65:
                summary += "○ |t统计量| > 1.65: 收益在90%置信度下显著\n"
            else:
                summary += "✗ |t统计量| < 1.65: 收益显著性不足\n"

        return summary


# 便捷函数

def portfolio_sorting_test(factor_data: FactorData,
                          n_quantiles: int = 5,
                          holding_period: int = 1,
                          long_short: bool = True) -> AnalysisResult:
    """
    快速执行分组回测

    便捷函数，用于快速进行分组回测分析。

    Args:
        factor_data: 因子数据
        n_quantiles: 分组数量（5或10）
        holding_period: 持有期
        long_short: 是否计算Long-Short

    Returns:
        AnalysisResult: 分组回测结果

    Example:
        >>> from factor_analysis.univariate import portfolio_sorting_test
        >>>
        >>> result = portfolio_sorting_test(factor_data, n_quantiles=5)
        >>> print(result.summary)
        >>> print(f"Long-Short夏普: {result.metrics['ls_sharpe']:.2f}")
        >>> result.figures['long_short_return'].savefig('ls_return.png')
    """
    sorter = PortfolioSorter(
        n_quantiles=n_quantiles,
        holding_period=holding_period,
        long_short=long_short
    )
    return sorter.analyze(factor_data)
