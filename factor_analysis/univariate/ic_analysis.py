"""
IC分析器

计算因子值与未来收益的相关性（Information Coefficient）。
这是最基础也是最重要的因子检验方法。

核心功能：
- 计算每日IC（Pearson/Spearman）
- 统计指标：ICIR、正值占比、t统计量
- 可视化：IC时间序列、分布直方图、滚动IC
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..core import FactorData, AnalysisResult
from ..utils.data_prep import compute_forward_returns, align_factor_and_returns
from ..utils.metrics import compute_ic_metrics


class ICAnalyzer:
    """
    IC分析器

    计算因子与收益的相关性指标，评估因子的预测能力。

    Attributes:
        method: 相关系数计算方法（'pearson' 或 'spearman'）
        forward_periods: 预测未来N期收益
        min_periods: 计算IC所需的最小样本数
    """

    def __init__(self,
                 method: str = 'pearson',
                 forward_periods: int = 1,
                 min_periods: int = 20):
        """
        初始化IC分析器

        Args:
            method: 相关系数计算方法
                   'pearson': Pearson相关系数（线性相关）
                   'spearman': Spearman等级相关系数（单调相关，更稳健）
            forward_periods: 预测的未来期数（1表示预测下一期）
            min_periods: 计算IC所需的最小横截面样本数

        Example:
            >>> analyzer = ICAnalyzer(method='spearman', forward_periods=1)
            >>> result = analyzer.analyze(factor_data)
        """
        if method not in ['pearson', 'spearman']:
            raise ValueError("method 必须是 'pearson' 或 'spearman'")

        self.method = method
        self.forward_periods = forward_periods
        self.min_periods = min_periods

    def analyze(self, factor_data: FactorData) -> AnalysisResult:
        """
        执行IC分析

        Args:
            factor_data: 标准化的因子数据

        Returns:
            AnalysisResult: 包含IC序列、统计指标、可视化图表

        Example:
            >>> result = analyzer.analyze(factor_data)
            >>> print(result.summary)
            >>> print(f"ICIR: {result.metrics['icir']:.4f}")
            >>> result.figures['ic_timeseries'].savefig('ic.png')
        """
        # 1. 计算前向收益
        forward_returns = compute_forward_returns(
            factor_data.returns,
            self.forward_periods
        )

        # 2. 计算每日IC
        ic_series = self._compute_daily_ic(
            factor_data.factor_values,
            forward_returns
        )

        # 3. 计算统计指标
        metrics = compute_ic_metrics(ic_series)

        # 4. 生成可视化
        figures = self._create_plots(ic_series, factor_data.factor_name)

        # 5. 生成摘要
        summary = self._generate_summary(metrics, factor_data.factor_name)

        return AnalysisResult(
            name=f'IC Analysis - {factor_data.factor_name}',
            metrics=metrics,
            data={'ic_series': ic_series, 'forward_returns': forward_returns},
            figures=figures,
            summary=summary
        )

    def _compute_daily_ic(self,
                         factor_values: pd.DataFrame,
                         forward_returns: pd.DataFrame) -> pd.Series:
        """
        计算每日IC

        在每个交易日，计算横截面的因子值与未来收益的相关性。

        Args:
            factor_values: 因子值 DataFrame (MultiIndex: datetime, asset)
            forward_returns: 前向收益率 DataFrame

        Returns:
            IC时间序列 (index: datetime, values: IC)
        """
        ic_list = []
        dates = sorted(factor_values.index.get_level_values('timestamp').unique())

        for date in dates:
            try:
                # 获取该日横截面数据
                factor_cross = factor_values.xs(date, level='timestamp')['factor_value']
                returns_cross = forward_returns.xs(date, level='timestamp')['return']

                # 对齐并去除NaN
                aligned = pd.DataFrame({
                    'factor': factor_cross,
                    'return': returns_cross
                }).dropna()

                # 样本量检查
                if len(aligned) >= self.min_periods:
                    # 计算相关系数
                    if self.method == 'pearson':
                        ic = aligned['factor'].corr(aligned['return'])
                    elif self.method == 'spearman':
                        ic = aligned['factor'].corr(aligned['return'], method='spearman')
                    else:
                        raise ValueError(f"未知的相关系数方法: {self.method}")

                    ic_list.append({'date': date, 'ic': ic})
                else:
                    # 样本量不足，记录为NaN
                    ic_list.append({'date': date, 'ic': np.nan})

            except Exception as e:
                warnings.warn(f"日期 {date} 的IC计算失败: {e}")
                ic_list.append({'date': date, 'ic': np.nan})

        # 转换为Series
        ic_df = pd.DataFrame(ic_list)
        ic_series = ic_df.set_index('date')['ic']
        ic_series.index.name = 'datetime'

        return ic_series

    def _create_plots(self,
                     ic_series: pd.Series,
                     factor_name: str) -> Dict:
        """
        生成IC可视化图表

        Args:
            ic_series: IC时间序列
            factor_name: 因子名称

        Returns:
            图表字典 {'ic_timeseries': fig1, 'ic_distribution': fig2, 'ic_rolling': fig3}
        """
        figures = {}

        # 过滤有效数据
        valid_ic = ic_series.dropna()

        if len(valid_ic) == 0:
            warnings.warn("IC序列为空，无法生成图表")
            return figures

        # 设置中文字体（如果需要）
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # 图1：IC时间序列
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        valid_ic.plot(ax=ax1, label='IC', linewidth=1.5, color='steelblue')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=valid_ic.mean(), color='green', linestyle='--',
                   label=f'Mean IC: {valid_ic.mean():.4f}', linewidth=1.5, alpha=0.7)
        ax1.set_title(f'IC Time Series - {factor_name}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('IC', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        fig1.tight_layout()
        figures['ic_timeseries'] = fig1

        # 图2：IC分布直方图
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        valid_ic.hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=valid_ic.mean(), color='green', linestyle='--',
                   label=f'Mean: {valid_ic.mean():.4f}', linewidth=1.5, alpha=0.7)
        ax2.set_title(f'IC Distribution - {factor_name}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('IC', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        fig2.tight_layout()
        figures['ic_distribution'] = fig2

        # 图3：滚动IC稳定性（30日滚动均值和标准差）
        if len(valid_ic) >= 30:
            fig3, ax3 = plt.subplots(figsize=(12, 4))
            rolling_window = min(30, len(valid_ic) // 3)
            rolling_ic = valid_ic.rolling(window=rolling_window).mean()
            rolling_std = valid_ic.rolling(window=rolling_window).std()

            rolling_ic.plot(ax=ax3, label=f'Rolling IC ({rolling_window}d)',
                          linewidth=2, color='steelblue')
            ax3.fill_between(rolling_ic.index,
                            rolling_ic - rolling_std,
                            rolling_ic + rolling_std,
                            alpha=0.2, color='steelblue', label='±1 Std')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax3.set_title(f'Rolling IC Stability - {factor_name}',
                        fontsize=12, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=10)
            ax3.set_ylabel('IC', fontsize=10)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3, linestyle='--')
            fig3.tight_layout()
            figures['ic_rolling'] = fig3

        return figures

    def _generate_summary(self,
                         metrics: Dict,
                         factor_name: str) -> str:
        """
        生成文字摘要

        Args:
            metrics: IC指标字典
            factor_name: 因子名称

        Returns:
            格式化的摘要字符串
        """
        method_name = 'Pearson IC' if self.method == 'pearson' else 'Rank IC (Spearman)'

        summary = f"""
IC Analysis Summary - {factor_name}
{'=' * 60}
Method:               {method_name}
Forward Periods:      {self.forward_periods}

Key Metrics:
------------
IC Mean:              {metrics['ic_mean']:.4f}
IC Std:               {metrics['ic_std']:.4f}
ICIR:                 {metrics['icir']:.4f}
IC Positive Ratio:    {metrics['ic_positive_ratio']:.2%}
IC Abs Mean:          {metrics['ic_abs_mean']:.4f}

Statistical Significance:
-------------------------
t-statistic:          {metrics['t_statistic']:.4f}
p-value:              {metrics['p_value']:.4f}
95% Confidence:       [{metrics['confidence_lower']:.4f}, {metrics['confidence_upper']:.4f}]

Interpretation:
---------------
"""
        # 添加解释
        if metrics['icir'] > 0.5:
            summary += "✓ ICIR > 0.5: 因子稳定性较好，预测能力较强\n"
        elif metrics['icir'] > 0.3:
            summary += "○ ICIR 0.3-0.5: 因子有一定的预测能力，稳定性中等\n"
        else:
            summary += "✗ ICIR < 0.3: 因子预测能力较弱或不稳定\n"

        if metrics['ic_positive_ratio'] > 0.6:
            summary += "✓ IC正值占比 > 60%: 因子预测方向一致性强\n"
        elif metrics['ic_positive_ratio'] > 0.5:
            summary += "○ IC正值占比 > 50%: 因子预测方向基本一致\n"
        else:
            summary += "✗ IC正值占比 < 50%: 因子预测方向不稳定\n"

        if metrics['p_value'] < 0.05:
            summary += "✓ p-value < 0.05: IC显著不为0（95%置信度）\n"
        elif metrics['p_value'] < 0.10:
            summary += "○ p-value < 0.10: IC在90%置信度下显著\n"
        else:
            summary += "✗ p-value >= 0.10: IC统计显著性不足\n"

        summary += "\nConclusion:\n-----------\n"
        if metrics['icir'] > 0.5 and metrics['ic_positive_ratio'] > 0.6:
            summary += "该因子表现优秀，具有较强的预测能力和稳定性。\n"
        elif metrics['icir'] > 0.3 or metrics['ic_positive_ratio'] > 0.55:
            summary += "该因子表现尚可，可考虑与其他因子组合使用。\n"
        else:
            summary += "该因子预测能力较弱，不建议单独使用。\n"

        return summary


# 便捷函数

def compute_ic(factor_data: FactorData,
              method: str = 'pearson',
              forward_periods: int = 1,
              min_periods: int = 20) -> AnalysisResult:
    """
    快速计算IC

    便捷函数，用于快速执行IC分析。

    Args:
        factor_data: 因子数据
        method: 'pearson' 或 'spearman'
        forward_periods: 预测期数
        min_periods: 最小样本数

    Returns:
        AnalysisResult: IC分析结果

    Example:
        >>> from factor_analysis.core import FactorData
        >>> from factor_analysis.univariate import compute_ic
        >>>
        >>> # 假设已经有 factor_df 和 returns_df
        >>> factor_data = FactorData.from_engine_output(
        ...     factor_df, returns_df, '动量因子'
        ... )
        >>>
        >>> # 一行代码完成IC分析
        >>> result = compute_ic(factor_data, method='spearman')
        >>> print(result.summary)
        >>> print(f"ICIR: {result.metrics['icir']:.4f}")
        >>>
        >>> # 保存结果和图表
        >>> result.figures['ic_timeseries'].savefig('ic_timeseries.png')
        >>> result.save('ic_analysis_result')
    """
    analyzer = ICAnalyzer(
        method=method,
        forward_periods=forward_periods,
        min_periods=min_periods
    )
    return analyzer.analyze(factor_data)


def batch_ic_analysis(factors: Dict[str, FactorData],
                     method: str = 'spearman',
                     forward_periods: int = 1) -> pd.DataFrame:
    """
    批量IC分析

    对多个因子执行IC分析，返回汇总结果。

    Args:
        factors: 因子字典 {因子名: FactorData}
        method: 相关系数方法
        forward_periods: 预测期数

    Returns:
        汇总结果 DataFrame

    Example:
        >>> factors = {
        ...     '动量': momentum_factor_data,
        ...     '反转': reversal_factor_data,
        ...     '波动': volatility_factor_data
        ... }
        >>> summary = batch_ic_analysis(factors, method='spearman')
        >>> print(summary.sort_values('icir', ascending=False))
    """
    results = []

    for factor_name, factor_data in factors.items():
        try:
            result = compute_ic(factor_data, method, forward_periods)
            metrics = result.metrics.copy()
            metrics['factor_name'] = factor_name
            results.append(metrics)
        except Exception as e:
            warnings.warn(f"因子 {factor_name} 的IC分析失败: {e}")

    if not results:
        return pd.DataFrame()

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.set_index('factor_name')

    # 按ICIR排序
    summary_df = summary_df.sort_values('icir', ascending=False)

    return summary_df
