"""
Factor Correlation Analysis Module

Analyze correlations between multiple factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from ..core import FactorData, AnalysisResult


class FactorCorrelation:
    """
    Factor Correlation Analyzer

    Compute correlation matrix between multiple factors.
    """

    def __init__(self, method: str = 'pearson'):
        """
        Initialize correlation analyzer

        Args:
            method: 'pearson' or 'spearman'
        """
        if method not in ['pearson', 'spearman']:
            raise ValueError("method must be 'pearson' or 'spearman'")

        self.method = method

    def analyze(self, factors: Dict[str, FactorData]) -> AnalysisResult:
        """
        Analyze correlations between multiple factors

        Args:
            factors: Dictionary of {factor_name: FactorData}

        Returns:
            AnalysisResult with correlation matrix and visualizations
        """
        # 1. Align all factors
        factor_df = self._align_factors(factors)

        # 2. Compute correlation matrix
        corr_matrix = factor_df.corr(method=self.method)

        # 3. Find high correlation pairs
        high_corr_pairs = self._find_high_correlation(corr_matrix, threshold=0.7)

        # 4. Generate visualizations
        figures = self._create_plots(corr_matrix)

        # 5. Generate summary
        summary = self._generate_summary(corr_matrix, high_corr_pairs)

        return AnalysisResult(
            name='Factor Correlation Analysis',
            metrics={
                'n_factors': len(factors),
                'mean_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                'max_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                'high_corr_pairs_count': len(high_corr_pairs)
            },
            data={
                'correlation_matrix': corr_matrix,
                'high_corr_pairs': pd.DataFrame(high_corr_pairs),
                'factor_values': factor_df
            },
            figures=figures,
            summary=summary
        )

    def _align_factors(self, factors: Dict[str, FactorData]) -> pd.DataFrame:
        """Align all factor values to common index"""
        factor_dict = {}

        for name, factor_data in factors.items():
            # Extract factor values
            factor_values = factor_data.factor_values['factor_value']
            factor_dict[name] = factor_values

        # Create DataFrame and align
        factor_df = pd.DataFrame(factor_dict)

        # Drop rows with any NaN
        factor_df = factor_df.dropna()

        return factor_df

    def _find_high_correlation(self, corr_matrix: pd.DataFrame, threshold: float) -> list:
        """Find pairs with correlation above threshold"""
        high_corr_pairs = []

        n = len(corr_matrix)
        for i in range(n):
            for j in range(i+1, n):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'factor1': corr_matrix.index[i],
                        'factor2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        return high_corr_pairs

    def _create_plots(self, corr_matrix: pd.DataFrame) -> Dict:
        """Generate correlation visualizations"""
        figures = {}

        # Set style
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Figure 1: Heatmap
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, ax=ax1,
                   square=True, linewidths=0.5)
        ax1.set_title('Factor Correlation Heatmap', fontsize=14, fontweight='bold')
        fig1.tight_layout()
        figures['heatmap'] = fig1

        # Figure 2: Clustermap
        if len(corr_matrix) > 2:
            try:
                fig2 = sns.clustermap(corr_matrix, annot=True, fmt='.2f',
                                     cmap='coolwarm', center=0,
                                     figsize=(10, 8), linewidths=0.5)
                fig2.fig.suptitle('Factor Correlation Clustermap',
                                 fontsize=14, fontweight='bold', y=0.98)
                figures['clustermap'] = fig2.fig
            except:
                warnings.warn("Failed to create clustermap")

        # Figure 3: Dendrogram
        if len(corr_matrix) > 2:
            try:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                # Convert correlation to distance
                distance_matrix = 1 - corr_matrix.abs()
                condensed_dist = squareform(distance_matrix, checks=False)
                linkage = hierarchy.linkage(condensed_dist, method='ward')
                hierarchy.dendrogram(linkage, labels=corr_matrix.index,
                                   ax=ax3, leaf_font_size=10)
                ax3.set_title('Factor Hierarchical Clustering',
                            fontsize=14, fontweight='bold')
                ax3.set_xlabel('Factors', fontsize=12)
                ax3.set_ylabel('Distance', fontsize=12)
                fig3.tight_layout()
                figures['dendrogram'] = fig3
            except:
                warnings.warn("Failed to create dendrogram")

        return figures

    def _generate_summary(self, corr_matrix: pd.DataFrame,
                         high_corr_pairs: list) -> str:
        """Generate text summary"""
        n_factors = len(corr_matrix)

        # Calculate statistics
        upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        mean_corr = upper_tri.mean()
        max_corr = upper_tri.max()
        min_corr = upper_tri.min()

        summary = f"""
Factor Correlation Analysis Summary
{'='*60}
Number of Factors:    {n_factors}
Total Pairs:          {len(upper_tri)}

Correlation Statistics:
-----------------------
Mean Correlation:     {mean_corr:.4f}
Max Correlation:      {max_corr:.4f}
Min Correlation:      {min_corr:.4f}

High Correlation Pairs (|corr| >= 0.7):
----------------------------------------
"""

        if high_corr_pairs:
            for pair in high_corr_pairs:
                summary += f"{pair['factor1']:20s} <-> {pair['factor2']:20s}: {pair['correlation']:>7.4f}\n"
        else:
            summary += "No high correlation pairs found.\n"

        summary += "\nRecommendations:\n"
        summary += "----------------\n"
        if len(high_corr_pairs) > 0:
            summary += f"- Found {len(high_corr_pairs)} highly correlated factor pairs\n"
            summary += "- Consider removing redundant factors to reduce multicollinearity\n"
            summary += "- Use PCA or factor combination for dimensionality reduction\n"
        else:
            summary += "- All factors are relatively independent\n"
            summary += "- Good diversity for factor combination\n"

        return summary


def compute_correlation(factors: Dict[str, FactorData],
                       method: str = 'spearman') -> AnalysisResult:
    """
    Quick correlation analysis

    Args:
        factors: Dictionary of factors
        method: Correlation method

    Returns:
        AnalysisResult with correlation matrix

    Example:
        >>> factors = {
        ...     'momentum': momentum_data,
        ...     'reversal': reversal_data,
        ...     'volatility': volatility_data
        ... }
        >>> result = compute_correlation(factors)
        >>> print(result.summary)
        >>> result.figures['heatmap'].savefig('correlation.png')
    """
    analyzer = FactorCorrelation(method=method)
    return analyzer.analyze(factors)
