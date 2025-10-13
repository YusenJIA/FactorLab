"""
Robustness Testing

Test factor robustness through parameter sensitivity and stability analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable, Any
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from ..core import FactorData, AnalysisResult
from ..univariate.ic_analysis import ICAnalyzer


class RobustnessTest:
    """
    Robustness Testing

    Test factor stability under different conditions and parameter settings.
    """

    def __init__(self, method: str = 'spearman'):
        """
        Initialize robustness tester

        Args:
            method: Correlation method for IC calculation
        """
        self.method = method

    def parameter_sensitivity(self,
                            factor_data_list: List[FactorData],
                            param_values: List[Any],
                            param_name: str) -> AnalysisResult:
        """
        Test factor sensitivity to parameter changes

        Args:
            factor_data_list: List of FactorData with different parameter values
            param_values: List of parameter values (corresponding to factor_data_list)
            param_name: Name of the parameter being tested

        Returns:
            AnalysisResult with sensitivity analysis

        Example:
            >>> # Test window length sensitivity
            >>> factor_data_list = [
            >>>     compute_factor(window=10),
            >>>     compute_factor(window=20),
            >>>     compute_factor(window=30)
            >>> ]
            >>> result = tester.parameter_sensitivity(
            >>>     factor_data_list,
            >>>     param_values=[10, 20, 30],
            >>>     param_name='window_length'
            >>> )
        """
        if len(factor_data_list) != len(param_values):
            raise ValueError("factor_data_list and param_values must have same length")

        # Analyze each parameter setting
        param_results = []
        ic_analyzer = ICAnalyzer(method=self.method)

        for param_val, factor_data in zip(param_values, factor_data_list):
            try:
                result = ic_analyzer.analyze(factor_data)
                param_results.append({
                    param_name: param_val,
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir'],
                    'ic_positive_ratio': result.metrics['ic_positive_ratio']
                })
            except Exception as e:
                warnings.warn(f"Failed to analyze parameter {param_val}: {e}")
                continue

        # Create DataFrame
        param_df = pd.DataFrame(param_results)
        param_df.set_index(param_name, inplace=True)

        # Calculate metrics
        metrics = {
            'param_name': param_name,
            'n_params': len(param_df),
            'mean_ic': param_df['ic_mean'].mean(),
            'ic_range': param_df['ic_mean'].max() - param_df['ic_mean'].min(),
            'ic_std': param_df['ic_mean'].std(),
            'optimal_param': param_df['icir'].idxmax() if len(param_df) > 0 else None,
            'optimal_icir': param_df['icir'].max() if len(param_df) > 0 else None,
            'stability_score': 1.0 - (param_df['ic_mean'].std() / abs(param_df['ic_mean'].mean())) if param_df['ic_mean'].mean() != 0 else 0.0
        }

        # Create visualizations
        figures = self._create_sensitivity_plots(param_df, param_name)

        # Generate summary
        summary = self._generate_sensitivity_summary(param_df, metrics)

        return AnalysisResult(
            name=f'Parameter Sensitivity - {param_name}',
            metrics=metrics,
            data={
                'param_results': param_df
            },
            figures=figures,
            summary=summary
        )

    def subsample_stability(self,
                          factor_data: FactorData,
                          n_samples: int = 10,
                          sample_ratio: float = 0.8,
                          random_state: Optional[int] = None) -> AnalysisResult:
        """
        Test factor stability with random subsampling

        Args:
            factor_data: Factor data
            n_samples: Number of random samples
            sample_ratio: Ratio of data to sample (0-1)
            random_state: Random seed for reproducibility

        Returns:
            AnalysisResult with subsample stability analysis

        Example:
            >>> result = tester.subsample_stability(
            >>>     factor_data,
            >>>     n_samples=20,
            >>>     sample_ratio=0.8
            >>> )
        """
        if random_state is not None:
            np.random.seed(random_state)

        dates = sorted(factor_data.factor_values.index.get_level_values('timestamp').unique())
        n_dates = int(len(dates) * sample_ratio)

        subsample_results = []
        ic_analyzer = ICAnalyzer(method=self.method)

        for i in range(n_samples):
            # Random sample dates
            sampled_dates = np.random.choice(dates, size=n_dates, replace=False)
            sampled_dates = sorted(sampled_dates)

            # Filter data
            date_mask = factor_data.factor_values.index.get_level_values('timestamp').isin(sampled_dates)
            sampled_factors = factor_data.factor_values[date_mask]

            return_mask = factor_data.returns.index.get_level_values('timestamp').isin(sampled_dates)
            sampled_returns = factor_data.returns[return_mask]

            # Create FactorData
            sampled_factor_data = FactorData(
                factor_values=sampled_factors,
                returns=sampled_returns,
                factor_name=factor_data.factor_name,
                start_date=factor_data.start_date,
                end_date=factor_data.end_date,
                frequency=factor_data.frequency
            )

            # Calculate IC
            try:
                result = ic_analyzer.analyze(sampled_factor_data)
                subsample_results.append({
                    'sample': i + 1,
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir']
                })
            except:
                pass

        # Create DataFrame
        subsample_df = pd.DataFrame(subsample_results)

        # Calculate metrics
        metrics = {
            'n_samples': len(subsample_df),
            'sample_ratio': sample_ratio,
            'mean_ic': subsample_df['ic_mean'].mean(),
            'ic_std': subsample_df['ic_mean'].std(),
            'ic_min': subsample_df['ic_mean'].min(),
            'ic_max': subsample_df['ic_mean'].max(),
            'ic_range': subsample_df['ic_mean'].max() - subsample_df['ic_mean'].min(),
            'positive_ratio': (subsample_df['ic_mean'] > 0).mean(),
            'stability_score': 1.0 - subsample_df['ic_mean'].std() / abs(subsample_df['ic_mean'].mean()) if subsample_df['ic_mean'].mean() != 0 else 0.0
        }

        # Create visualizations
        figures = self._create_subsample_plots(subsample_df)

        # Generate summary
        summary = self._generate_subsample_summary(subsample_df, metrics)

        return AnalysisResult(
            name=f'Subsample Stability - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'subsample_results': subsample_df
            },
            figures=figures,
            summary=summary
        )

    def return_period_test(self,
                          factor_data: FactorData,
                          return_periods: List[int] = [1, 5, 10, 20]) -> AnalysisResult:
        """
        Test factor performance with different return periods

        Args:
            factor_data: Factor data (should have multiple return columns)
            return_periods: List of return periods to test

        Returns:
            AnalysisResult with return period analysis

        Example:
            >>> result = tester.return_period_test(
            >>>     factor_data,
            >>>     return_periods=[1, 5, 10, 20]
            >>> )
        """
        # Check available return columns
        return_cols = [col for col in factor_data.returns.columns if col.startswith('return_')]

        if len(return_cols) == 0:
            raise ValueError("No return columns found in factor_data.returns")

        period_results = []
        ic_analyzer = ICAnalyzer(method=self.method)

        for period in return_periods:
            return_col = f'return_{period}d'

            if return_col not in return_cols:
                warnings.warn(f"Return column {return_col} not found, skipping")
                continue

            # Create FactorData with specific return period
            period_returns = factor_data.returns[[return_col]].copy()
            period_returns.columns = ['return']

            period_factor_data = FactorData(
                factor_values=factor_data.factor_values,
                returns=period_returns,
                factor_name=factor_data.factor_name,
                start_date=factor_data.start_date,
                end_date=factor_data.end_date,
                frequency=factor_data.frequency
            )

            # Calculate IC
            try:
                result = ic_analyzer.analyze(period_factor_data)
                period_results.append({
                    'return_period': period,
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir'],
                    'ic_positive_ratio': result.metrics['ic_positive_ratio']
                })
            except Exception as e:
                warnings.warn(f"Failed to analyze return period {period}: {e}")
                continue

        # Create DataFrame
        period_df = pd.DataFrame(period_results)
        period_df.set_index('return_period', inplace=True)

        # Calculate metrics
        metrics = {
            'n_periods': len(period_df),
            'mean_ic': period_df['ic_mean'].mean(),
            'optimal_period': period_df['icir'].idxmax() if len(period_df) > 0 else None,
            'optimal_icir': period_df['icir'].max() if len(period_df) > 0 else None,
            'ic_decay_trend': self._calculate_trend(period_df['ic_mean'])
        }

        # Create visualizations
        figures = self._create_return_period_plots(period_df)

        # Generate summary
        summary = self._generate_return_period_summary(period_df, metrics)

        return AnalysisResult(
            name=f'Return Period Test - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'period_results': period_df
            },
            figures=figures,
            summary=summary
        )

    def asset_universe_test(self,
                          factor_data: FactorData,
                          universe_groups: Dict[str, List[str]]) -> AnalysisResult:
        """
        Test factor stability across different asset universes

        Args:
            factor_data: Factor data
            universe_groups: Dictionary of {universe_name: [asset_list]}

        Returns:
            AnalysisResult with universe stability analysis

        Example:
            >>> universes = {
            >>>     'Large Cap': ['510050.SH', '510300.SH'],
            >>>     'All ETFs': ['510050.SH', '510300.SH', '510500.SH']
            >>> }
            >>> result = tester.asset_universe_test(factor_data, universes)
        """
        universe_results = []
        ic_analyzer = ICAnalyzer(method=self.method)

        for universe_name, assets in universe_groups.items():
            # Filter by assets
            asset_mask = factor_data.factor_values.index.get_level_values('asset').isin(assets)
            universe_factors = factor_data.factor_values[asset_mask]

            return_mask = factor_data.returns.index.get_level_values('asset').isin(assets)
            universe_returns = factor_data.returns[return_mask]

            if len(universe_factors) == 0:
                warnings.warn(f"No data for universe {universe_name}")
                continue

            # Create FactorData
            universe_factor_data = FactorData(
                factor_values=universe_factors,
                returns=universe_returns,
                factor_name=factor_data.factor_name,
                start_date=factor_data.start_date,
                end_date=factor_data.end_date,
                frequency=factor_data.frequency
            )

            # Calculate IC
            try:
                result = ic_analyzer.analyze(universe_factor_data)
                universe_results.append({
                    'universe': universe_name,
                    'n_assets': len(assets),
                    'ic_mean': result.metrics['ic_mean'],
                    'ic_std': result.metrics['ic_std'],
                    'icir': result.metrics['icir']
                })
            except Exception as e:
                warnings.warn(f"Failed to analyze universe {universe_name}: {e}")
                continue

        # Create DataFrame
        universe_df = pd.DataFrame(universe_results)
        universe_df.set_index('universe', inplace=True)

        # Calculate metrics
        metrics = {
            'n_universes': len(universe_df),
            'mean_ic': universe_df['ic_mean'].mean(),
            'ic_consistency': 1.0 - universe_df['ic_mean'].std() / abs(universe_df['ic_mean'].mean()) if universe_df['ic_mean'].mean() != 0 else 0.0,
            'best_universe': universe_df['icir'].idxmax() if len(universe_df) > 0 else None
        }

        # Create visualizations
        figures = self._create_universe_plots(universe_df)

        # Generate summary
        summary = self._generate_universe_summary(universe_df, metrics)

        return AnalysisResult(
            name=f'Asset Universe Test - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'universe_results': universe_df
            },
            figures=figures,
            summary=summary
        )

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend (slope)"""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def _create_sensitivity_plots(self, param_df: pd.DataFrame, param_name: str) -> Dict:
        """Create parameter sensitivity plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = param_df.index

        # IC Mean sensitivity
        ax1.plot(x, param_df['ic_mean'], marker='o', linewidth=2, markersize=8)
        ax1.fill_between(x,
                        param_df['ic_mean'] - param_df['ic_std'],
                        param_df['ic_mean'] + param_df['ic_std'],
                        alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_xlabel(param_name, fontsize=12)
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title(f'IC Sensitivity to {param_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # ICIR sensitivity
        ax2.plot(x, param_df['icir'], marker='s', linewidth=2, markersize=8, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_xlabel(param_name, fontsize=12)
        ax2.set_ylabel('ICIR', fontsize=12)
        ax2.set_title(f'ICIR Sensitivity to {param_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Mark optimal parameter
        if len(param_df) > 0:
            optimal_idx = param_df['icir'].idxmax()
            ax2.scatter([optimal_idx], [param_df.loc[optimal_idx, 'icir']],
                       color='red', s=200, marker='*', zorder=5, label='Optimal')
            ax2.legend()

        fig.tight_layout()
        figures['parameter_sensitivity'] = fig

        return figures

    def _create_subsample_plots(self, subsample_df: pd.DataFrame) -> Dict:
        """Create subsample stability plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # IC distribution
        ax1.hist(subsample_df['ic_mean'], bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(x=subsample_df['ic_mean'].mean(), color='red',
                   linestyle='--', linewidth=2, label=f'Mean: {subsample_df["ic_mean"].mean():.4f}')
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('IC Mean', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('IC Distribution Across Subsamples', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # IC across samples
        ax2.plot(subsample_df['sample'], subsample_df['ic_mean'], marker='o', alpha=0.6)
        ax2.axhline(y=subsample_df['ic_mean'].mean(), color='red',
                   linestyle='--', linewidth=2, label='Mean IC')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(subsample_df['sample'],
                        subsample_df['ic_mean'].mean() - subsample_df['ic_mean'].std(),
                        subsample_df['ic_mean'].mean() + subsample_df['ic_mean'].std(),
                        alpha=0.3, color='red', label='±1 Std')
        ax2.set_xlabel('Sample Number', fontsize=12)
        ax2.set_ylabel('IC Mean', fontsize=12)
        ax2.set_title('IC Stability Across Subsamples', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        figures['subsample_stability'] = fig

        return figures

    def _create_return_period_plots(self, period_df: pd.DataFrame) -> Dict:
        """Create return period plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = period_df.index

        # IC by return period
        ax1.plot(x, period_df['ic_mean'], marker='o', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Return Period (days)', fontsize=12)
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title('IC vs Return Period', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # ICIR by return period
        ax2.bar(x, period_df['icir'], alpha=0.7, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Return Period (days)', fontsize=12)
        ax2.set_ylabel('ICIR', fontsize=12)
        ax2.set_title('ICIR vs Return Period', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        figures['return_period_test'] = fig

        return figures

    def _create_universe_plots(self, universe_df: pd.DataFrame) -> Dict:
        """Create asset universe plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(universe_df))

        # IC comparison
        ax1.bar(x, universe_df['ic_mean'], alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Universe', fontsize=12)
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title('IC Across Asset Universes', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(universe_df.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # ICIR comparison
        ax2.bar(x, universe_df['icir'], alpha=0.7, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Universe', fontsize=12)
        ax2.set_ylabel('ICIR', fontsize=12)
        ax2.set_title('ICIR Across Asset Universes', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(universe_df.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        figures['universe_test'] = fig

        return figures

    def _generate_sensitivity_summary(self, param_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate parameter sensitivity summary"""
        summary = f"""
Parameter Sensitivity Analysis
{'='*60}
Parameter:              {metrics['param_name']}
Number of Tests:        {metrics['n_params']}
Mean IC:                {metrics['mean_ic']:.4f}
IC Range:               {metrics['ic_range']:.4f}
IC Std:                 {metrics['ic_std']:.4f}
Optimal Parameter:      {metrics['optimal_param']}
Optimal ICIR:           {metrics['optimal_icir']:.4f}
Stability Score:        {metrics['stability_score']:.4f}

Parameter Details:
{'─'*60}
"""

        for param_val, row in param_df.iterrows():
            summary += f"\n{metrics['param_name']} = {param_val}:\n"
            summary += f"  IC Mean:      {row['ic_mean']:>7.4f}\n"
            summary += f"  ICIR:         {row['icir']:>7.4f}\n"

        summary += "\nInterpretation:\n"
        summary += "─" * 60 + "\n"

        if metrics['stability_score'] > 0.8:
            summary += "- Highly stable: Factor performance is robust to parameter changes\n"
        elif metrics['stability_score'] > 0.5:
            summary += "- Moderately stable: Some sensitivity to parameter changes\n"
        else:
            summary += "- Unstable: Factor is highly sensitive to parameter selection\n"

        return summary

    def _generate_subsample_summary(self, subsample_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate subsample stability summary"""
        summary = f"""
Subsample Stability Analysis
{'='*60}
Number of Samples:      {metrics['n_samples']}
Sample Ratio:           {metrics['sample_ratio']:.1%}
Mean IC:                {metrics['mean_ic']:.4f}
IC Std:                 {metrics['ic_std']:.4f}
IC Range:               [{metrics['ic_min']:.4f}, {metrics['ic_max']:.4f}]
Positive IC Ratio:      {metrics['positive_ratio']:.2%}
Stability Score:        {metrics['stability_score']:.4f}

Interpretation:
--------------
"""

        if metrics['positive_ratio'] > 0.9 and metrics['stability_score'] > 0.7:
            summary += "- Excellent: Factor is highly stable across subsamples\n"
        elif metrics['positive_ratio'] > 0.7 and metrics['stability_score'] > 0.5:
            summary += "- Good: Factor shows acceptable stability\n"
        else:
            summary += "- Warning: Factor performance is inconsistent across subsamples\n"

        return summary

    def _generate_return_period_summary(self, period_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate return period summary"""
        summary = f"""
Return Period Analysis
{'='*60}
Number of Periods:      {metrics['n_periods']}
Mean IC:                {metrics['mean_ic']:.4f}
Optimal Period:         {metrics['optimal_period']} days
Optimal ICIR:           {metrics['optimal_icir']:.4f}
IC Decay Trend:         {metrics['ic_decay_trend']:.6f}

Period Details:
{'─'*60}
"""

        for period, row in period_df.iterrows():
            summary += f"\n{int(period)}-day return:\n"
            summary += f"  IC Mean:      {row['ic_mean']:>7.4f}\n"
            summary += f"  ICIR:         {row['icir']:>7.4f}\n"

        summary += "\nInterpretation:\n"
        summary += "─" * 60 + "\n"

        if metrics['ic_decay_trend'] < -0.001:
            summary += "- Factor effectiveness decreases with longer holding periods\n"
        elif metrics['ic_decay_trend'] > 0.001:
            summary += "- Factor effectiveness increases with longer holding periods\n"
        else:
            summary += "- Factor effectiveness is stable across holding periods\n"

        return summary

    def _generate_universe_summary(self, universe_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate universe stability summary"""
        summary = f"""
Asset Universe Stability Analysis
{'='*60}
Number of Universes:    {metrics['n_universes']}
Mean IC:                {metrics['mean_ic']:.4f}
IC Consistency:         {metrics['ic_consistency']:.4f}
Best Universe:          {metrics['best_universe']}

Universe Details:
{'─'*60}
"""

        for universe, row in universe_df.iterrows():
            summary += f"\n{universe} ({int(row['n_assets'])} assets):\n"
            summary += f"  IC Mean:      {row['ic_mean']:>7.4f}\n"
            summary += f"  ICIR:         {row['icir']:>7.4f}\n"

        summary += "\nInterpretation:\n"
        summary += "─" * 60 + "\n"

        if metrics['ic_consistency'] > 0.8:
            summary += "- Highly consistent: Factor works well across different universes\n"
        elif metrics['ic_consistency'] > 0.5:
            summary += "- Moderately consistent: Some variation across universes\n"
        else:
            summary += "- Inconsistent: Factor performance varies significantly by universe\n"

        return summary


# Convenience functions
def robustness_test(factor_data: FactorData,
                   test_type: str = 'subsample',
                   method: str = 'spearman',
                   **kwargs) -> AnalysisResult:
    """
    Quick robustness test

    Args:
        factor_data: Factor data
        test_type: Type of test ('subsample', 'return_period', 'universe')
        method: Correlation method
        **kwargs: Additional arguments for specific test

    Returns:
        AnalysisResult with robustness test results

    Example:
        >>> result = robustness_test(
        ...     factor_data,
        ...     test_type='subsample',
        ...     n_samples=20,
        ...     sample_ratio=0.8
        ... )
    """
    tester = RobustnessTest(method=method)

    if test_type == 'subsample':
        return tester.subsample_stability(factor_data, **kwargs)
    elif test_type == 'return_period':
        return tester.return_period_test(factor_data, **kwargs)
    elif test_type == 'universe':
        return tester.asset_universe_test(factor_data, **kwargs)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
