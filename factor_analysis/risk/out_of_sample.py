"""
Out-of-Sample Testing

Test factor performance on unseen data to validate generalization.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..core import FactorData, AnalysisResult
from ..univariate.ic_analysis import ICAnalyzer


class OutOfSampleTest:
    """
    Out-of-Sample Testing

    Validate factor performance on test data after training.
    """

    def __init__(self, method: str = 'spearman'):
        """
        Initialize out-of-sample tester

        Args:
            method: Correlation method for IC calculation
        """
        self.method = method

    def train_test_split(self,
                        factor_data: FactorData,
                        train_ratio: float = 0.7,
                        split_date: Optional[str] = None) -> AnalysisResult:
        """
        Split data into training and testing periods and compare performance

        Args:
            factor_data: Factor data
            train_ratio: Ratio of training data (if split_date not provided)
            split_date: Explicit split date (overrides train_ratio)

        Returns:
            AnalysisResult with in-sample and out-of-sample comparison

        Example:
            >>> result = tester.train_test_split(factor_data, train_ratio=0.7)
            >>> print(result.summary)
        """
        # Determine split date
        dates = sorted(factor_data.factor_values.index.get_level_values('timestamp').unique())

        if split_date is None:
            split_idx = int(len(dates) * train_ratio)
            split_dt = dates[split_idx]
        else:
            split_dt = pd.to_datetime(split_date)

        # Split data
        train_data = self._filter_by_date(
            factor_data,
            dates[0].strftime('%Y-%m-%d'),
            split_dt.strftime('%Y-%m-%d')
        )

        test_data = self._filter_by_date(
            factor_data,
            split_dt.strftime('%Y-%m-%d'),
            dates[-1].strftime('%Y-%m-%d')
        )

        # Analyze both periods
        ic_analyzer = ICAnalyzer(method=self.method)

        train_result = ic_analyzer.analyze(train_data)
        test_result = ic_analyzer.analyze(test_data)

        # Calculate comparison metrics
        metrics = {
            'split_date': split_dt.strftime('%Y-%m-%d'),
            'train_days': len(train_result.data['ic_series']),
            'test_days': len(test_result.data['ic_series']),
            'train_ic_mean': train_result.metrics['ic_mean'],
            'train_ic_std': train_result.metrics['ic_std'],
            'train_icir': train_result.metrics['icir'],
            'test_ic_mean': test_result.metrics['ic_mean'],
            'test_ic_std': test_result.metrics['ic_std'],
            'test_icir': test_result.metrics['icir'],
            'ic_mean_diff': test_result.metrics['ic_mean'] - train_result.metrics['ic_mean'],
            'icir_diff': test_result.metrics['icir'] - train_result.metrics['icir'],
            'ic_mean_decay_pct': (test_result.metrics['ic_mean'] - train_result.metrics['ic_mean']) / abs(train_result.metrics['ic_mean']) if train_result.metrics['ic_mean'] != 0 else np.nan,
            'sign_consistency': np.sign(train_result.metrics['ic_mean']) == np.sign(test_result.metrics['ic_mean'])
        }

        # Create visualizations
        figures = self._create_split_plots(train_result, test_result, split_dt)

        # Generate summary
        summary = self._generate_split_summary(metrics)

        return AnalysisResult(
            name=f'Out-of-Sample Test - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'train_result': train_result,
                'test_result': test_result,
                'split_date': split_dt
            },
            figures=figures,
            summary=summary
        )

    def rolling_oos_test(self,
                        factor_data: FactorData,
                        train_window: int = 252,
                        test_window: int = 63,
                        step: int = 21) -> AnalysisResult:
        """
        Rolling out-of-sample test

        Args:
            factor_data: Factor data
            train_window: Training window size (days)
            test_window: Testing window size (days)
            step: Step size for rolling window

        Returns:
            AnalysisResult with rolling OOS performance

        Example:
            >>> # Train on 1 year, test on next quarter, roll by month
            >>> result = tester.rolling_oos_test(
            ...     factor_data,
            ...     train_window=252,
            ...     test_window=63,
            ...     step=21
            ... )
        """
        dates = sorted(factor_data.factor_values.index.get_level_values('timestamp').unique())

        if len(dates) < train_window + test_window:
            raise ValueError(f"Not enough data: {len(dates)} days < {train_window + test_window}")

        rolling_results = []
        ic_analyzer = ICAnalyzer(method=self.method)

        i = 0
        while i + train_window + test_window <= len(dates):
            # Define train and test periods
            train_start = dates[i]
            train_end = dates[i + train_window - 1]
            test_start = dates[i + train_window]
            test_end = dates[min(i + train_window + test_window - 1, len(dates) - 1)]

            # Filter data
            train_data = self._filter_by_date(
                factor_data,
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d')
            )

            test_data = self._filter_by_date(
                factor_data,
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            )

            # Calculate IC for both periods
            try:
                train_result = ic_analyzer.analyze(train_data)
                test_result = ic_analyzer.analyze(test_data)

                rolling_results.append({
                    'test_end_date': test_end,
                    'train_ic_mean': train_result.metrics['ic_mean'],
                    'train_icir': train_result.metrics['icir'],
                    'test_ic_mean': test_result.metrics['ic_mean'],
                    'test_icir': test_result.metrics['icir'],
                    'ic_decay': test_result.metrics['ic_mean'] - train_result.metrics['ic_mean'],
                    'sign_consistent': np.sign(train_result.metrics['ic_mean']) == np.sign(test_result.metrics['ic_mean'])
                })
            except:
                pass

            i += step

        # Create DataFrame
        rolling_df = pd.DataFrame(rolling_results)
        rolling_df.set_index('test_end_date', inplace=True)

        # Calculate metrics
        metrics = {
            'n_windows': len(rolling_df),
            'mean_train_ic': rolling_df['train_ic_mean'].mean(),
            'mean_test_ic': rolling_df['test_ic_mean'].mean(),
            'mean_ic_decay': rolling_df['ic_decay'].mean(),
            'ic_decay_std': rolling_df['ic_decay'].std(),
            'sign_consistency_ratio': rolling_df['sign_consistent'].mean(),
            'test_ic_stability': rolling_df['test_ic_mean'].std()
        }

        # Create visualizations
        figures = self._create_rolling_oos_plots(rolling_df)

        # Generate summary
        summary = self._generate_rolling_oos_summary(rolling_df, metrics, train_window, test_window)

        return AnalysisResult(
            name=f'Rolling OOS Test - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'rolling_results': rolling_df
            },
            figures=figures,
            summary=summary
        )

    def walk_forward_test(self,
                         factor_data: FactorData,
                         n_folds: int = 5) -> AnalysisResult:
        """
        Walk-forward out-of-sample test (expanding window)

        Args:
            factor_data: Factor data
            n_folds: Number of folds

        Returns:
            AnalysisResult with walk-forward performance

        Example:
            >>> result = tester.walk_forward_test(factor_data, n_folds=5)
        """
        dates = sorted(factor_data.factor_values.index.get_level_values('timestamp').unique())
        total_days = len(dates)

        if total_days < n_folds * 2:
            raise ValueError(f"Not enough data for {n_folds} folds")

        fold_results = []
        ic_analyzer = ICAnalyzer(method=self.method)

        # Initial training size
        initial_train_size = total_days // (n_folds + 1)

        for fold in range(n_folds):
            # Expanding window: train on all data up to this point
            train_end_idx = initial_train_size + fold * (total_days - initial_train_size) // n_folds
            test_start_idx = train_end_idx + 1
            test_end_idx = min(train_end_idx + (total_days - initial_train_size) // n_folds,
                              total_days - 1)

            train_start = dates[0]
            train_end = dates[train_end_idx]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            # Filter data
            train_data = self._filter_by_date(
                factor_data,
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d')
            )

            test_data = self._filter_by_date(
                factor_data,
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            )

            # Calculate IC
            try:
                train_result = ic_analyzer.analyze(train_data)
                test_result = ic_analyzer.analyze(test_data)

                fold_results.append({
                    'fold': fold + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_ic_mean': train_result.metrics['ic_mean'],
                    'train_icir': train_result.metrics['icir'],
                    'test_ic_mean': test_result.metrics['ic_mean'],
                    'test_icir': test_result.metrics['icir'],
                    'ic_decay': test_result.metrics['ic_mean'] - train_result.metrics['ic_mean']
                })
            except Exception as e:
                warnings.warn(f"Failed to analyze fold {fold + 1}: {e}")
                continue

        # Create DataFrame
        fold_df = pd.DataFrame(fold_results)

        # Calculate metrics
        metrics = {
            'n_folds': len(fold_df),
            'mean_train_ic': fold_df['train_ic_mean'].mean(),
            'mean_test_ic': fold_df['test_ic_mean'].mean(),
            'mean_ic_decay': fold_df['ic_decay'].mean(),
            'ic_decay_std': fold_df['ic_decay'].std(),
            'test_ic_std': fold_df['test_ic_mean'].std(),
            'all_positive_test_ic': (fold_df['test_ic_mean'] > 0).all() if len(fold_df) > 0 else False
        }

        # Create visualizations
        figures = self._create_walk_forward_plots(fold_df)

        # Generate summary
        summary = self._generate_walk_forward_summary(fold_df, metrics)

        return AnalysisResult(
            name=f'Walk-Forward Test - {factor_data.factor_name}',
            metrics=metrics,
            data={
                'fold_results': fold_df
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

    def _create_split_plots(self, train_result, test_result, split_date) -> Dict:
        """Create train-test split comparison plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Figure 1: IC time series
        fig1, ax1 = plt.subplots(figsize=(14, 6))

        train_ic = train_result.data['ic_series']
        test_ic = test_result.data['ic_series']

        # Concatenate for plotting
        all_ic = pd.concat([train_ic, test_ic])

        ax1.plot(train_ic.index, train_ic.values, label='In-Sample (Train)',
                color='blue', alpha=0.7, linewidth=1.5)
        ax1.plot(test_ic.index, test_ic.values, label='Out-of-Sample (Test)',
                color='red', alpha=0.7, linewidth=1.5)
        ax1.axvline(x=split_date, color='green', linestyle='--',
                   linewidth=2, label=f'Split Date: {split_date.strftime("%Y-%m-%d")}')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('IC', fontsize=12)
        ax1.set_title('In-Sample vs Out-of-Sample IC', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        fig1.tight_layout()
        figures['train_test_split'] = fig1

        # Figure 2: Metrics comparison
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

        metrics_names = ['IC Mean', 'IC Std', 'ICIR']
        train_metrics = [train_result.metrics['ic_mean'],
                        train_result.metrics['ic_std'],
                        train_result.metrics['icir']]
        test_metrics = [test_result.metrics['ic_mean'],
                       test_result.metrics['ic_std'],
                       test_result.metrics['icir']]

        x = np.arange(len(metrics_names))
        width = 0.35

        ax2.bar(x - width/2, train_metrics, width, label='In-Sample', alpha=0.7)
        ax2.bar(x + width/2, test_metrics, width, label='Out-of-Sample', alpha=0.7)
        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Distribution comparison
        ax3.hist(train_ic.values, bins=30, alpha=0.5, label='In-Sample', density=True)
        ax3.hist(test_ic.values, bins=30, alpha=0.5, label='Out-of-Sample', density=True)
        ax3.set_xlabel('IC', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('IC Distribution Comparison', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        fig2.tight_layout()
        figures['metrics_comparison'] = fig2

        return figures

    def _create_rolling_oos_plots(self, rolling_df: pd.DataFrame) -> Dict:
        """Create rolling OOS test plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # IC comparison
        ax1.plot(rolling_df.index, rolling_df['train_ic_mean'],
                label='Train IC', linewidth=2, marker='o', markersize=4)
        ax1.plot(rolling_df.index, rolling_df['test_ic_mean'],
                label='Test IC', linewidth=2, marker='s', markersize=4)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax1.fill_between(rolling_df.index, rolling_df['train_ic_mean'],
                        alpha=0.3, label='Train IC Area')
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title('Rolling Out-of-Sample Test: IC Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # IC decay
        ax2.bar(rolling_df.index, rolling_df['ic_decay'],
               alpha=0.7, color='orange', label='IC Decay (Test - Train)')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Test End Date', fontsize=12)
        ax2.set_ylabel('IC Decay', fontsize=12)
        ax2.set_title('IC Decay Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        figures['rolling_oos_test'] = fig

        return figures

    def _create_walk_forward_plots(self, fold_df: pd.DataFrame) -> Dict:
        """Create walk-forward test plots"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = fold_df['fold']

        # IC comparison by fold
        ax1.plot(x, fold_df['train_ic_mean'], marker='o', label='Train IC', linewidth=2)
        ax1.plot(x, fold_df['test_ic_mean'], marker='s', label='Test IC', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Fold', fontsize=12)
        ax1.set_ylabel('IC Mean', fontsize=12)
        ax1.set_title('Walk-Forward Test: IC by Fold', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # IC decay by fold
        ax2.bar(x, fold_df['ic_decay'], alpha=0.7, color='orange')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Fold', fontsize=12)
        ax2.set_ylabel('IC Decay', fontsize=12)
        ax2.set_title('IC Decay by Fold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        figures['walk_forward_test'] = fig

        return figures

    def _generate_split_summary(self, metrics: Dict) -> str:
        """Generate train-test split summary"""
        summary = f"""
Out-of-Sample Test Summary (Train-Test Split)
{'='*60}
Split Date:             {metrics['split_date']}
Train Days:             {metrics['train_days']}
Test Days:              {metrics['test_days']}

In-Sample (Train) Performance:
  IC Mean:              {metrics['train_ic_mean']:.4f}
  IC Std:               {metrics['train_ic_std']:.4f}
  ICIR:                 {metrics['train_icir']:.4f}

Out-of-Sample (Test) Performance:
  IC Mean:              {metrics['test_ic_mean']:.4f}
  IC Std:               {metrics['test_ic_std']:.4f}
  ICIR:                 {metrics['test_icir']:.4f}

Performance Degradation:
  IC Mean Diff:         {metrics['ic_mean_diff']:.4f}
  ICIR Diff:            {metrics['icir_diff']:.4f}
  IC Decay Rate:        {metrics['ic_mean_decay_pct']:.2%}
  Sign Consistency:     {'Yes' if metrics['sign_consistency'] else 'No'}

Interpretation:
--------------
"""

        # Interpret results
        if metrics['sign_consistency']:
            if abs(metrics['ic_mean_decay_pct']) < 0.2:
                summary += "- Excellent: Factor generalizes well with minimal performance decay\n"
            elif abs(metrics['ic_mean_decay_pct']) < 0.5:
                summary += "- Good: Factor shows acceptable out-of-sample performance\n"
            else:
                summary += "- Warning: Significant performance decay in out-of-sample period\n"
        else:
            summary += "- Critical: Factor sign flips in out-of-sample period (potential overfitting)\n"

        return summary

    def _generate_rolling_oos_summary(self, rolling_df: pd.DataFrame,
                                      metrics: Dict, train_window: int, test_window: int) -> str:
        """Generate rolling OOS summary"""
        summary = f"""
Rolling Out-of-Sample Test Summary
{'='*60}
Train Window:           {train_window} days
Test Window:            {test_window} days
Number of Windows:      {metrics['n_windows']}

Average Performance:
  Mean Train IC:        {metrics['mean_train_ic']:.4f}
  Mean Test IC:         {metrics['mean_test_ic']:.4f}
  Mean IC Decay:        {metrics['mean_ic_decay']:.4f}
  IC Decay Std:         {metrics['ic_decay_std']:.4f}

Consistency:
  Sign Consistency:     {metrics['sign_consistency_ratio']:.2%}
  Test IC Stability:    {metrics['test_ic_stability']:.4f}

Interpretation:
--------------
"""

        # Interpret results
        if metrics['sign_consistency_ratio'] > 0.8:
            summary += "- High sign consistency: Factor direction is stable\n"
        elif metrics['sign_consistency_ratio'] > 0.6:
            summary += "- Moderate sign consistency: Factor mostly stable\n"
        else:
            summary += "- Low sign consistency: Factor direction unstable\n"

        if metrics['mean_ic_decay'] > -0.01:
            summary += "- Minimal decay: Factor maintains performance out-of-sample\n"
        elif metrics['mean_ic_decay'] > -0.03:
            summary += "- Acceptable decay: Some performance degradation\n"
        else:
            summary += "- Significant decay: Factor may be overfitted\n"

        return summary

    def _generate_walk_forward_summary(self, fold_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate walk-forward summary"""
        summary = f"""
Walk-Forward Test Summary
{'='*60}
Number of Folds:        {metrics['n_folds']}

Average Performance:
  Mean Train IC:        {metrics['mean_train_ic']:.4f}
  Mean Test IC:         {metrics['mean_test_ic']:.4f}
  Mean IC Decay:        {metrics['mean_ic_decay']:.4f}
  IC Decay Std:         {metrics['ic_decay_std']:.4f}
  Test IC Std:          {metrics['test_ic_std']:.4f}

Consistency:
  All Positive Test IC: {'Yes' if metrics['all_positive_test_ic'] else 'No'}

Fold Details:
{'─'*60}
"""

        for _, row in fold_df.iterrows():
            summary += f"\nFold {int(row['fold'])}:\n"
            summary += f"  Train IC:     {row['train_ic_mean']:>7.4f}\n"
            summary += f"  Test IC:      {row['test_ic_mean']:>7.4f}\n"
            summary += f"  IC Decay:     {row['ic_decay']:>7.4f}\n"

        summary += "\nInterpretation:\n"
        summary += "─" * 60 + "\n"

        if metrics['all_positive_test_ic'] and metrics['mean_ic_decay'] > -0.02:
            summary += "- Excellent: Factor performs consistently across all folds\n"
        elif metrics['mean_test_ic'] > 0:
            summary += "- Good: Factor shows positive out-of-sample performance\n"
        else:
            summary += "- Warning: Factor performance is inconsistent or negative\n"

        return summary


# Convenience functions
def out_of_sample_test(factor_data: FactorData,
                      train_ratio: float = 0.7,
                      split_date: Optional[str] = None,
                      method: str = 'spearman') -> AnalysisResult:
    """
    Quick out-of-sample test

    Args:
        factor_data: Factor data
        train_ratio: Training data ratio
        split_date: Optional explicit split date
        method: Correlation method

    Returns:
        AnalysisResult with OOS test results

    Example:
        >>> result = out_of_sample_test(factor_data, train_ratio=0.7)
        >>> print(result.summary)
    """
    tester = OutOfSampleTest(method=method)
    return tester.train_test_split(factor_data, train_ratio, split_date)
