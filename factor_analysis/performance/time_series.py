"""
Time Series Performance Analysis

Analyze factor performance across different time periods.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from ..core import FactorData, AnalysisResult
from ..univariate.ic_analysis import ICAnalyzer


class TimeSeriesAnalysis:
    """
    Time Series Performance Analyzer

    Analyze factor performance in different time periods and rolling windows.
    """

    def __init__(self, method: str = 'spearman'):
        """
        Initialize time series analyzer

        Args:
            method: Correlation method for IC calculation
        """
        self.method = method

    def analyze_by_period(self,
                         factor_data: FactorData,
                         periods: List[Tuple[str, str]],
                         period_names: Optional[List[str]] = None) -> AnalysisResult:
        """
        Analyze factor performance in different time periods

        Args:
            factor_data: Factor data
            periods: List of (start_date, end_date) tuples
            period_names: Optional names for periods

        Returns:
            AnalysisResult with period-wise performance
        """
        if period_names is None:
            period_names = [f"Period_{i+1}" for i in range(len(periods))]

        if len(period_names) != len(periods):
            raise ValueError("period_names length must match periods length")

        # Calculate IC for each period
        period_results = {}
        ic_analyzer = ICAnalyzer(method=self.method)

        for period_name, (start, end) in zip(period_names, periods):
            # Filter data for this period
            period_factor_data = self._filter_by_date(factor_data, start, end)

            if len(period_factor_data.factor_values) == 0:
                warnings.warn(f"No data in period {period_name} ({start} to {end})")
                continue

            # Calculate IC for this period
            try:
                result = ic_analyzer.analyze(period_factor_data)
                period_results[period_name] = {
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir'],
                    'ic_positive_ratio': result.metrics['ic_positive_ratio'],
                    'start_date': start,
                    'end_date': end,
                    'n_observations': len(result.data['ic_series'])
                }
            except Exception as e:
                warnings.warn(f"Failed to analyze period {period_name}: {e}")
                continue

        # Create summary DataFrame
        summary_df = pd.DataFrame(period_results).T

        # Calculate metrics
        metrics = {
            'n_periods': len(period_results),
            'mean_ic': summary_df['ic_mean'].mean(),
            'mean_icir': summary_df['icir'].mean(),
            'ic_stability': summary_df['ic_mean'].std(),
            'best_period': summary_df['icir'].idxmax() if len(summary_df) > 0 else None,
            'worst_period': summary_df['icir'].idxmin() if len(summary_df) > 0 else None
        }

        # Create visualizations
        figures = self._create_period_plots(summary_df, period_names)

        # Generate summary
        summary = self._generate_period_summary(summary_df, metrics)

        return AnalysisResult(
            name=f'Time Period Analysis - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'period_summary': summary_df,
                'period_results': period_results
            },
            figures=figures,
            summary=summary
        )

    def rolling_analysis(self,
                        factor_data: FactorData,
                        window: int = 60,
                        step: int = 20) -> AnalysisResult:
        """
        Rolling window performance analysis

        Args:
            factor_data: Factor data
            window: Rolling window size (days)
            step: Step size for rolling window

        Returns:
            AnalysisResult with rolling performance
        """
        # Get all dates
        dates = sorted(factor_data.factor_values.index.get_level_values('timestamp').unique())

        if len(dates) < window:
            raise ValueError(f"Not enough data: {len(dates)} days < {window} window")

        rolling_results = []
        ic_analyzer = ICAnalyzer(method=self.method)

        # Rolling window calculation
        i = 0
        while i + window <= len(dates):
            start_date = dates[i]
            end_date = dates[min(i + window - 1, len(dates) - 1)]

            # Filter data
            window_data = self._filter_by_date(
                factor_data,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            # Calculate IC
            try:
                result = ic_analyzer.analyze(window_data)
                rolling_results.append({
                    'end_date': end_date,
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir'],
                    'ic_positive_ratio': result.metrics['ic_positive_ratio']
                })
            except:
                pass

            i += step

        # Create DataFrame
        rolling_df = pd.DataFrame(rolling_results)
        rolling_df.set_index('end_date', inplace=True)

        # Calculate metrics
        metrics = {
            'n_windows': len(rolling_df),
            'mean_rolling_ic': rolling_df['ic_mean'].mean(),
            'mean_rolling_icir': rolling_df['icir'].mean(),
            'ic_trend': self._calculate_trend(rolling_df['ic_mean']),
            'icir_stability': rolling_df['icir'].std()
        }

        # Create visualizations
        figures = self._create_rolling_plots(rolling_df)

        # Generate summary
        summary = self._generate_rolling_summary(rolling_df, metrics, window)

        return AnalysisResult(
            name=f'Rolling Window Analysis - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'rolling_results': rolling_df
            },
            figures=figures,
            summary=summary
        )

    def _filter_by_date(self, factor_data: FactorData,
                       start_date: str, end_date: str) -> FactorData:
        """Filter FactorData by date range"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter factor values
        dates = factor_data.factor_values.index.get_level_values('timestamp')
        mask = (dates >= start_dt) & (dates <= end_dt)
        filtered_factors = factor_data.factor_values[mask]

        # Filter returns
        dates_ret = factor_data.returns.index.get_level_values('timestamp')
        mask_ret = (dates_ret >= start_dt) & (dates_ret <= end_dt)
        filtered_returns = factor_data.returns[mask_ret]

        return FactorData(
            factor_values=filtered_factors,
            returns=filtered_returns,
            factor_name=factor_data.factor_name,
            start_date=start_date,
            end_date=end_date,
            frequency=factor_data.frequency
        )

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend (slope)"""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = series.values
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def _create_period_plots(self, summary_df: pd.DataFrame,
                            period_names: List[str]) -> Dict:
        """Create period comparison plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Figure 1: IC comparison
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        x = np.arange(len(summary_df))
        width = 0.35

        ax1.bar(x, summary_df['ic_mean'], width, label='IC Mean', alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Period', fontsize=12)
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title('IC Mean by Period', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_df.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(x, summary_df['icir'], width, label='ICIR', color='orange', alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Period', fontsize=12)
        ax2.set_ylabel('ICIR', fontsize=12)
        ax2.set_title('ICIR by Period', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(summary_df.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        fig1.tight_layout()
        figures['period_comparison'] = fig1

        return figures

    def _create_rolling_plots(self, rolling_df: pd.DataFrame) -> Dict:
        """Create rolling window plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Figure: Rolling IC and ICIR
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1.plot(rolling_df.index, rolling_df['ic_mean'], label='IC Mean', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.fill_between(rolling_df.index, 0, rolling_df['ic_mean'],
                        where=rolling_df['ic_mean'] > 0, alpha=0.3, color='green')
        ax1.fill_between(rolling_df.index, 0, rolling_df['ic_mean'],
                        where=rolling_df['ic_mean'] <= 0, alpha=0.3, color='red')
        ax1.set_ylabel('Rolling IC Mean', fontsize=12)
        ax1.set_title('Rolling Window Performance', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(rolling_df.index, rolling_df['icir'], label='ICIR',
                color='orange', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.fill_between(rolling_df.index, 0, rolling_df['icir'],
                        where=rolling_df['icir'] > 0, alpha=0.3, color='green')
        ax2.fill_between(rolling_df.index, 0, rolling_df['icir'],
                        where=rolling_df['icir'] <= 0, alpha=0.3, color='red')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Rolling ICIR', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        figures['rolling_performance'] = fig

        return figures

    def _generate_period_summary(self, summary_df: pd.DataFrame,
                                 metrics: Dict) -> str:
        """Generate period analysis summary"""
        summary = f"""
Time Period Analysis Summary
{'='*60}
Number of Periods:      {metrics['n_periods']}
Mean IC:                {metrics['mean_ic']:.4f}
Mean ICIR:              {metrics['mean_icir']:.4f}
IC Stability (Std):     {metrics['ic_stability']:.4f}
"""

        if metrics['best_period']:
            summary += f"Best Period:            {metrics['best_period']}\n"
        if metrics['worst_period']:
            summary += f"Worst Period:           {metrics['worst_period']}\n"

        summary += f"\nPeriod Details:\n"
        summary += "─" * 60 + "\n"

        for period_name, row in summary_df.iterrows():
            summary += f"\n{period_name}:\n"
            summary += f"  IC Mean:      {row['ic_mean']:>7.4f}\n"
            summary += f"  ICIR:         {row['icir']:>7.4f}\n"
            summary += f"  IC Pos Ratio: {row['ic_positive_ratio']:>7.2%}\n"
            summary += f"  Observations: {int(row['n_observations'])}\n"

        return summary

    def _generate_rolling_summary(self, rolling_df: pd.DataFrame,
                                  metrics: Dict, window: int) -> str:
        """Generate rolling analysis summary"""
        summary = f"""
Rolling Window Analysis Summary
{'='*60}
Window Size:            {window} days
Number of Windows:      {metrics['n_windows']}
Mean Rolling IC:        {metrics['mean_rolling_ic']:.4f}
Mean Rolling ICIR:      {metrics['mean_rolling_icir']:.4f}
ICIR Stability (Std):   {metrics['icir_stability']:.4f}
IC Trend (Slope):       {metrics['ic_trend']:.6f}

Interpretation:
--------------
"""

        if metrics['ic_trend'] > 0.0001:
            summary += "- Positive trend: Factor performance improving over time\n"
        elif metrics['ic_trend'] < -0.0001:
            summary += "- Negative trend: Factor performance declining over time\n"
        else:
            summary += "- Stable: Factor performance relatively constant\n"

        if metrics['icir_stability'] < 0.5:
            summary += "- Low volatility: Stable factor performance\n"
        elif metrics['icir_stability'] < 1.0:
            summary += "- Moderate volatility: Reasonably stable performance\n"
        else:
            summary += "- High volatility: Inconsistent factor performance\n"

        return summary


# Convenience functions
def analyze_by_period(factor_data: FactorData,
                     periods: List[Tuple[str, str]],
                     period_names: Optional[List[str]] = None,
                     method: str = 'spearman') -> AnalysisResult:
    """
    Quick time period analysis

    Args:
        factor_data: Factor data
        periods: List of (start_date, end_date) tuples
        period_names: Optional period names
        method: Correlation method

    Returns:
        AnalysisResult with period analysis

    Example:
        >>> periods = [
        ...     ('2024-01-01', '2024-03-31'),  # Q1
        ...     ('2024-04-01', '2024-06-30'),  # Q2
        ...     ('2024-07-01', '2024-09-30'),  # Q3
        ...     ('2024-10-01', '2024-12-31'),  # Q4
        ... ]
        >>> result = analyze_by_period(factor_data, periods,
        ...                            period_names=['Q1', 'Q2', 'Q3', 'Q4'])
        >>> print(result.summary)
    """
    analyzer = TimeSeriesAnalysis(method=method)
    return analyzer.analyze_by_period(factor_data, periods, period_names)


def rolling_analysis(factor_data: FactorData,
                    window: int = 60,
                    step: int = 20,
                    method: str = 'spearman') -> AnalysisResult:
    """
    Quick rolling window analysis

    Args:
        factor_data: Factor data
        window: Rolling window size (days)
        step: Step size
        method: Correlation method

    Returns:
        AnalysisResult with rolling analysis

    Example:
        >>> result = rolling_analysis(factor_data, window=90, step=30)
        >>> print(result.summary)
        >>> result.figures['rolling_performance'].savefig('rolling.png')
    """
    analyzer = TimeSeriesAnalysis(method=method)
    return analyzer.rolling_analysis(factor_data, window, step)
