"""
Cross-Sectional Performance Analysis

Analyze factor performance across different asset groups.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from ..core import FactorData, AnalysisResult
from ..univariate.ic_analysis import ICAnalyzer


class CrossSectionAnalysis:
    """
    Cross-Sectional Performance Analyzer

    Analyze factor performance across different groups of assets.
    """

    def __init__(self, method: str = 'spearman'):
        """
        Initialize cross-sectional analyzer

        Args:
            method: Correlation method for IC calculation
        """
        self.method = method

    def analyze_by_groups(self,
                         factor_data: FactorData,
                         groups: Dict[str, List[str]],
                         group_names: Optional[List[str]] = None) -> AnalysisResult:
        """
        Analyze factor performance across different asset groups

        Args:
            factor_data: Factor data
            groups: Dictionary of {group_name: [asset_list]}
            group_names: Optional list of group names (for ordering)

        Returns:
            AnalysisResult with group-wise performance

        Example:
            >>> groups = {
            >>>     'Large Cap': ['510050.SH', '510300.SH'],
            >>>     'Small Cap': ['510500.SH', '510700.SH']
            >>> }
            >>> result = analyzer.analyze_by_groups(factor_data, groups)
        """
        if group_names is None:
            group_names = list(groups.keys())

        # Calculate IC for each group
        group_results = {}
        ic_analyzer = ICAnalyzer(method=self.method)

        for group_name in group_names:
            if group_name not in groups:
                warnings.warn(f"Group {group_name} not found in groups dictionary")
                continue

            assets = groups[group_name]

            # Filter data for this group
            group_factor_data = self._filter_by_assets(factor_data, assets)

            if len(group_factor_data.factor_values) == 0:
                warnings.warn(f"No data for group {group_name}")
                continue

            # Calculate IC for this group
            try:
                result = ic_analyzer.analyze(group_factor_data)
                group_results[group_name] = {
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir'],
                    'ic_positive_ratio': result.metrics['ic_positive_ratio'],
                    'n_assets': len(assets),
                    'n_observations': len(result.data['ic_series'])
                }
            except Exception as e:
                warnings.warn(f"Failed to analyze group {group_name}: {e}")
                continue

        # Create summary DataFrame
        summary_df = pd.DataFrame(group_results).T

        # Calculate metrics
        metrics = {
            'n_groups': len(group_results),
            'mean_ic': summary_df['ic_mean'].mean(),
            'mean_icir': summary_df['icir'].mean(),
            'ic_dispersion': summary_df['ic_mean'].std(),
            'best_group': summary_df['icir'].idxmax() if len(summary_df) > 0 else None,
            'worst_group': summary_df['icir'].idxmin() if len(summary_df) > 0 else None
        }

        # Create visualizations
        figures = self._create_group_plots(summary_df)

        # Generate summary
        summary = self._generate_group_summary(summary_df, metrics)

        return AnalysisResult(
            name=f'Cross-Section Analysis - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'group_summary': summary_df,
                'group_results': group_results
            },
            figures=figures,
            summary=summary
        )

    def analyze_by_quantiles(self,
                            factor_data: FactorData,
                            quantile_col: str,
                            n_quantiles: int = 5) -> AnalysisResult:
        """
        Analyze factor performance by quantiles of another characteristic

        Args:
            factor_data: Factor data
            quantile_col: Column name to split quantiles on (e.g., 'market_cap', 'volatility')
            n_quantiles: Number of quantiles

        Returns:
            AnalysisResult with quantile-wise performance

        Example:
            >>> # Analyze factor performance across market cap quintiles
            >>> result = analyzer.analyze_by_quantiles(
            >>>     factor_data,
            >>>     quantile_col='market_cap',
            >>>     n_quantiles=5
            >>> )
        """
        # Check if quantile_col exists
        if quantile_col not in factor_data.factor_values.columns:
            raise ValueError(f"Column {quantile_col} not found in factor_values")

        # Split into quantiles
        quantile_labels = [f'Q{i+1}' for i in range(n_quantiles)]

        # Group by date and assign quantiles
        factor_values = factor_data.factor_values.copy()
        factor_values['quantile'] = factor_values.groupby('timestamp')[quantile_col].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=quantile_labels, duplicates='drop')
        )

        # Analyze each quantile
        quantile_results = {}
        ic_analyzer = ICAnalyzer(method=self.method)

        for quantile in quantile_labels:
            # Filter by quantile
            mask = factor_values['quantile'] == quantile
            quantile_factor_values = factor_values[mask].drop('quantile', axis=1)

            if len(quantile_factor_values) == 0:
                warnings.warn(f"No data for quantile {quantile}")
                continue

            # Get corresponding returns
            quantile_returns = factor_data.returns.loc[quantile_factor_values.index]

            # Create FactorData for this quantile
            quantile_factor_data = FactorData(
                factor_values=quantile_factor_values,
                returns=quantile_returns,
                factor_name=factor_data.factor_name,
                start_date=factor_data.start_date,
                end_date=factor_data.end_date,
                frequency=factor_data.frequency
            )

            # Calculate IC
            try:
                result = ic_analyzer.analyze(quantile_factor_data)
                quantile_results[quantile] = {
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir'],
                    'ic_positive_ratio': result.metrics['ic_positive_ratio'],
                    'n_observations': len(result.data['ic_series'])
                }
            except Exception as e:
                warnings.warn(f"Failed to analyze quantile {quantile}: {e}")
                continue

        # Create summary DataFrame
        summary_df = pd.DataFrame(quantile_results).T

        # Calculate metrics
        metrics = {
            'n_quantiles': len(quantile_results),
            'split_variable': quantile_col,
            'mean_ic': summary_df['ic_mean'].mean(),
            'mean_icir': summary_df['icir'].mean(),
            'ic_spread': summary_df['ic_mean'].max() - summary_df['ic_mean'].min(),
            'monotonicity': self._calculate_monotonicity(summary_df['ic_mean'])
        }

        # Create visualizations
        figures = self._create_quantile_plots(summary_df, quantile_col)

        # Generate summary
        summary = self._generate_quantile_summary(summary_df, metrics)

        return AnalysisResult(
            name=f'Quantile Analysis - {factor_data.factor_name} by {quantile_col}',
            metrics=metrics,
            data={
                'quantile_summary': summary_df,
                'quantile_results': quantile_results
            },
            figures=figures,
            summary=summary
        )

    def _filter_by_assets(self, factor_data: FactorData, assets: List[str]) -> FactorData:
        """Filter FactorData by asset list"""
        # Filter factor values
        asset_mask = factor_data.factor_values.index.get_level_values('asset').isin(assets)
        filtered_factors = factor_data.factor_values[asset_mask]

        # Filter returns
        return_mask = factor_data.returns.index.get_level_values('asset').isin(assets)
        filtered_returns = factor_data.returns[return_mask]

        return FactorData(
            factor_values=filtered_factors,
            returns=filtered_returns,
            factor_name=factor_data.factor_name,
            start_date=factor_data.start_date,
            end_date=factor_data.end_date,
            frequency=factor_data.frequency
        )

    def _calculate_monotonicity(self, ic_series: pd.Series) -> float:
        """
        Calculate monotonicity score

        Returns:
            Score between -1 and 1:
            - 1: perfectly monotonic increasing
            - -1: perfectly monotonic decreasing
            - 0: no monotonic pattern
        """
        if len(ic_series) < 2:
            return 0.0

        diffs = ic_series.diff().dropna()

        # Count increasing and decreasing steps
        increasing = (diffs > 0).sum()
        decreasing = (diffs < 0).sum()
        total = len(diffs)

        if total == 0:
            return 0.0

        # Monotonicity score
        score = (increasing - decreasing) / total
        return score

    def _create_group_plots(self, summary_df: pd.DataFrame) -> Dict:
        """Create group comparison plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Figure: Group comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # IC Mean comparison
        x = np.arange(len(summary_df))
        ax1.bar(x, summary_df['ic_mean'], alpha=0.7, color='steelblue')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Group', fontsize=12)
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title('IC Mean by Group', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_df.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # ICIR comparison
        ax2.bar(x, summary_df['icir'], alpha=0.7, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Group', fontsize=12)
        ax2.set_ylabel('ICIR', fontsize=12)
        ax2.set_title('ICIR by Group', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(summary_df.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        figures['group_comparison'] = fig

        return figures

    def _create_quantile_plots(self, summary_df: pd.DataFrame, quantile_col: str) -> Dict:
        """Create quantile analysis plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Figure: Quantile performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(summary_df))

        # IC Mean across quantiles
        ax1.plot(x, summary_df['ic_mean'], marker='o', linewidth=2, markersize=8)
        ax1.fill_between(x,
                        summary_df['ic_mean'] - summary_df['ic_std'],
                        summary_df['ic_mean'] + summary_df['ic_std'],
                        alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_xlabel(f'{quantile_col} Quantile', fontsize=12)
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title(f'IC Mean across {quantile_col} Quantiles', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_df.index)
        ax1.grid(True, alpha=0.3)

        # ICIR across quantiles
        ax2.bar(x, summary_df['icir'], alpha=0.7, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_xlabel(f'{quantile_col} Quantile', fontsize=12)
        ax2.set_ylabel('ICIR', fontsize=12)
        ax2.set_title(f'ICIR across {quantile_col} Quantiles', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(summary_df.index)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        figures['quantile_performance'] = fig

        return figures

    def _generate_group_summary(self, summary_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate group analysis summary"""
        summary = f"""
Cross-Section Analysis Summary (by Groups)
{'='*60}
Number of Groups:       {metrics['n_groups']}
Mean IC:                {metrics['mean_ic']:.4f}
Mean ICIR:              {metrics['mean_icir']:.4f}
IC Dispersion (Std):    {metrics['ic_dispersion']:.4f}
"""

        if metrics['best_group']:
            summary += f"Best Group:             {metrics['best_group']}\n"
        if metrics['worst_group']:
            summary += f"Worst Group:            {metrics['worst_group']}\n"

        summary += f"\nGroup Details:\n"
        summary += "─" * 60 + "\n"

        for group_name, row in summary_df.iterrows():
            summary += f"\n{group_name}:\n"
            summary += f"  IC Mean:      {row['ic_mean']:>7.4f}\n"
            summary += f"  ICIR:         {row['icir']:>7.4f}\n"
            summary += f"  IC Pos Ratio: {row['ic_positive_ratio']:>7.2%}\n"
            summary += f"  # Assets:     {int(row['n_assets'])}\n"
            summary += f"  # Obs:        {int(row['n_observations'])}\n"

        return summary

    def _generate_quantile_summary(self, summary_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate quantile analysis summary"""
        summary = f"""
Cross-Section Analysis Summary (by Quantiles)
{'='*60}
Split Variable:         {metrics['split_variable']}
Number of Quantiles:    {metrics['n_quantiles']}
Mean IC:                {metrics['mean_ic']:.4f}
Mean ICIR:              {metrics['mean_icir']:.4f}
IC Spread (max-min):    {metrics['ic_spread']:.4f}
Monotonicity Score:     {metrics['monotonicity']:.4f}

Interpretation:
--------------
"""

        # Interpret monotonicity
        mono = metrics['monotonicity']
        if mono > 0.5:
            summary += "- Strong monotonic increasing: Factor effectiveness increases with split variable\n"
        elif mono > 0.2:
            summary += "- Weak monotonic increasing: Factor tends to work better with higher split variable\n"
        elif mono < -0.5:
            summary += "- Strong monotonic decreasing: Factor effectiveness decreases with split variable\n"
        elif mono < -0.2:
            summary += "- Weak monotonic decreasing: Factor tends to work better with lower split variable\n"
        else:
            summary += "- No clear monotonic pattern: Factor effectiveness varies non-monotonically\n"

        # Interpret IC spread
        if metrics['ic_spread'] > 0.1:
            summary += "- Large IC spread: Factor performance varies significantly across quantiles\n"
        elif metrics['ic_spread'] > 0.05:
            summary += "- Moderate IC spread: Some variation in factor performance\n"
        else:
            summary += "- Small IC spread: Factor performance relatively consistent\n"

        summary += f"\nQuantile Details:\n"
        summary += "─" * 60 + "\n"

        for quantile, row in summary_df.iterrows():
            summary += f"\n{quantile}:\n"
            summary += f"  IC Mean:      {row['ic_mean']:>7.4f}\n"
            summary += f"  ICIR:         {row['icir']:>7.4f}\n"
            summary += f"  IC Pos Ratio: {row['ic_positive_ratio']:>7.2%}\n"

        return summary


# Convenience functions
def analyze_by_groups(factor_data: FactorData,
                     groups: Dict[str, List[str]],
                     group_names: Optional[List[str]] = None,
                     method: str = 'spearman') -> AnalysisResult:
    """
    Quick cross-sectional analysis by groups

    Args:
        factor_data: Factor data
        groups: Dictionary of {group_name: [asset_list]}
        group_names: Optional list of group names
        method: Correlation method

    Returns:
        AnalysisResult with group analysis

    Example:
        >>> groups = {
        ...     'Large Cap': ['510050.SH', '510300.SH'],
        ...     'Mid Cap': ['510500.SH'],
        ...     'Small Cap': ['510700.SH']
        ... }
        >>> result = analyze_by_groups(factor_data, groups)
        >>> print(result.summary)
    """
    analyzer = CrossSectionAnalysis(method=method)
    return analyzer.analyze_by_groups(factor_data, groups, group_names)


def analyze_by_quantiles(factor_data: FactorData,
                        quantile_col: str,
                        n_quantiles: int = 5,
                        method: str = 'spearman') -> AnalysisResult:
    """
    Quick cross-sectional analysis by quantiles

    Args:
        factor_data: Factor data
        quantile_col: Column to split quantiles on
        n_quantiles: Number of quantiles
        method: Correlation method

    Returns:
        AnalysisResult with quantile analysis

    Example:
        >>> result = analyze_by_quantiles(
        ...     factor_data,
        ...     quantile_col='market_cap',
        ...     n_quantiles=5
        ... )
        >>> print(result.summary)
    """
    analyzer = CrossSectionAnalysis(method=method)
    return analyzer.analyze_by_quantiles(factor_data, quantile_col, n_quantiles)
