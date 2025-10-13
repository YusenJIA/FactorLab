"""
Principal Component Analysis (PCA) Module

Perform dimensionality reduction and factor extraction using PCA.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from ..core import FactorData, AnalysisResult
from ..utils.data_prep import standardize_by_date


class PCAAnalysis:
    """
    Principal Component Analysis for Factors

    Reduce factor dimensionality and extract principal components.
    """

    def __init__(self, n_components: Optional[int] = None,
                 variance_threshold: float = 0.95):
        """
        Initialize PCA analyzer

        Args:
            n_components: Number of components to keep (None = auto based on variance)
            variance_threshold: Cumulative variance threshold for auto selection
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def analyze(self, factors: Dict[str, FactorData]) -> AnalysisResult:
        """
        Perform PCA analysis on multiple factors

        Args:
            factors: Dictionary of {factor_name: FactorData}

        Returns:
            AnalysisResult with PCA components and analysis
        """
        # 1. Align and standardize factors
        factor_df = self._prepare_factors(factors)

        # 2. Perform PCA
        pca_result = self._perform_pca(factor_df)

        # 3. Calculate metrics
        metrics = self._calculate_metrics(pca_result)

        # 4. Create visualizations
        figures = self._create_plots(pca_result, list(factors.keys()))

        # 5. Generate summary
        summary = self._generate_summary(pca_result, metrics)

        # 6. Create PC factors
        pc_factors = self._create_pc_factors(pca_result, factors)

        return AnalysisResult(
            name='PCA Analysis',
            metrics=metrics,
            data={
                'pca_model': pca_result['model'],
                'components': pca_result['components'],
                'explained_variance': pca_result['explained_variance'],
                'explained_variance_ratio': pca_result['explained_variance_ratio'],
                'loadings': pca_result['loadings'],
                'pc_scores': pca_result['pc_scores'],
                'pc_factors': pc_factors
            },
            figures=figures,
            summary=summary
        )

    def _prepare_factors(self, factors: Dict[str, FactorData]) -> pd.DataFrame:
        """Align and standardize all factors"""
        factor_dict = {}

        for name, factor_data in factors.items():
            # Extract and standardize
            factor_values = factor_data.factor_values['factor_value']
            standardized = standardize_by_date(
                pd.DataFrame({'factor_value': factor_values}),
                method='zscore'
            )
            factor_dict[name] = standardized['factor_value']

        # Align all factors
        factor_df = pd.DataFrame(factor_dict)
        factor_df = factor_df.dropna()

        if len(factor_df) == 0:
            raise ValueError("No common dates between factors after alignment")

        return factor_df

    def _perform_pca(self, factor_df: pd.DataFrame) -> Dict:
        """Perform PCA decomposition"""
        # Determine number of components
        if self.n_components is None:
            # Use variance threshold
            pca_full = SklearnPCA()
            pca_full.fit(factor_df)

            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
            n_components = max(1, min(n_components, len(factor_df.columns)))
        else:
            n_components = min(self.n_components, len(factor_df.columns))

        # Fit PCA with selected components
        pca = SklearnPCA(n_components=n_components)
        pc_scores = pca.fit_transform(factor_df)

        # Create component names
        component_names = [f'PC{i+1}' for i in range(n_components)]

        # Create scores DataFrame
        pc_scores_df = pd.DataFrame(
            pc_scores,
            index=factor_df.index,
            columns=component_names
        )

        # Create loadings DataFrame (components)
        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=factor_df.columns,
            columns=component_names
        )

        return {
            'model': pca,
            'n_components': n_components,
            'components': pca.components_,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'loadings': loadings_df,
            'pc_scores': pc_scores_df,
            'feature_names': list(factor_df.columns)
        }

    def _calculate_metrics(self, pca_result: Dict) -> Dict[str, float]:
        """Calculate PCA metrics"""
        metrics = {
            'n_components': pca_result['n_components'],
            'n_original_factors': len(pca_result['feature_names']),
            'total_variance_explained': pca_result['explained_variance_ratio'].sum(),
        }

        # Add individual component variance
        for i, var_ratio in enumerate(pca_result['explained_variance_ratio']):
            metrics[f'PC{i+1}_variance_ratio'] = var_ratio

        return metrics

    def _create_plots(self, pca_result: Dict, factor_names: List[str]) -> Dict:
        """Create PCA visualizations"""
        figures = {}

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Figure 1: Scree plot (explained variance)
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        n_comp = pca_result['n_components']
        x = np.arange(1, n_comp + 1)
        var_ratio = pca_result['explained_variance_ratio']
        cumsum_var = np.cumsum(var_ratio)

        ax1.bar(x, var_ratio, alpha=0.6, label='Individual')
        ax1.plot(x, cumsum_var, 'ro-', label='Cumulative')
        ax1.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% threshold')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        figures['scree_plot'] = fig1

        # Figure 2: Loadings heatmap
        fig2, ax2 = plt.subplots(figsize=(10, max(6, len(factor_names) * 0.4)))

        loadings = pca_result['loadings']
        sns.heatmap(loadings, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax2, cbar_kws={'label': 'Loading'})
        ax2.set_title('Factor Loadings on Principal Components',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Principal Components', fontsize=12)
        ax2.set_ylabel('Original Factors', fontsize=12)
        fig2.tight_layout()
        figures['loadings_heatmap'] = fig2

        # Figure 3: Biplot (if 2+ components)
        if n_comp >= 2:
            fig3, ax3 = plt.subplots(figsize=(10, 8))

            # Plot loadings as arrows
            loadings_matrix = pca_result['loadings'].values
            for i, feature in enumerate(factor_names):
                ax3.arrow(0, 0, loadings_matrix[i, 0], loadings_matrix[i, 1],
                         head_width=0.05, head_length=0.05, fc='red', ec='red',
                         alpha=0.6)
                ax3.text(loadings_matrix[i, 0] * 1.15, loadings_matrix[i, 1] * 1.15,
                        feature, fontsize=10, ha='center')

            # Set limits
            max_val = np.abs(loadings_matrix[:, :2]).max() * 1.3
            ax3.set_xlim(-max_val, max_val)
            ax3.set_ylim(-max_val, max_val)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax3.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=12)
            ax3.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=12)
            ax3.set_title('PCA Biplot - Factor Loadings', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            fig3.tight_layout()
            figures['biplot'] = fig3

        return figures

    def _generate_summary(self, pca_result: Dict, metrics: Dict) -> str:
        """Generate text summary"""
        n_comp = pca_result['n_components']
        total_var = metrics['total_variance_explained']

        summary = f"""
PCA Analysis Summary
{'='*60}
Original Factors:          {metrics['n_original_factors']}
Components Selected:       {n_comp}
Total Variance Explained:  {total_var:.2%}

Component Variance Breakdown:
{'─'*60}
"""

        for i in range(n_comp):
            var_ratio = metrics[f'PC{i+1}_variance_ratio']
            summary += f"PC{i+1}:  {var_ratio:>6.2%}  "
            summary += "█" * int(var_ratio * 50) + "\n"

        summary += f"\nTop Factor Loadings per Component:\n"
        summary += "─" * 60 + "\n"

        loadings_df = pca_result['loadings']
        for i in range(min(n_comp, 3)):  # Show top 3 components
            pc_name = f'PC{i+1}'
            top_factors = loadings_df[pc_name].abs().nlargest(3)
            summary += f"\n{pc_name}:\n"
            for factor, loading in top_factors.items():
                actual_loading = loadings_df.loc[factor, pc_name]
                summary += f"  {factor:20s}: {actual_loading:>7.3f}\n"

        summary += "\nRecommendations:\n"
        summary += "─" * 60 + "\n"
        if total_var >= 0.95:
            summary += f"- Excellent: {n_comp} components explain {total_var:.1%} of variance\n"
            summary += "- Consider using PC factors instead of original factors\n"
        elif total_var >= 0.80:
            summary += f"- Good: {n_comp} components explain {total_var:.1%} of variance\n"
            summary += "- PC factors can be used for dimensionality reduction\n"
        else:
            summary += f"- Moderate: {n_comp} components explain only {total_var:.1%} of variance\n"
            summary += "- Original factors may be more informative\n"

        return summary

    def _create_pc_factors(self, pca_result: Dict,
                          original_factors: Dict[str, FactorData]) -> Dict[str, FactorData]:
        """Create FactorData objects for principal components"""
        pc_factors = {}

        pc_scores_df = pca_result['pc_scores']

        # Use first factor's metadata as template
        first_factor = list(original_factors.values())[0]

        for col in pc_scores_df.columns:
            # Create factor values DataFrame
            pc_values = pc_scores_df[col].to_frame('factor_value')

            # Align returns with PC scores
            common_index = pc_values.index.intersection(first_factor.returns.index)
            pc_values_aligned = pc_values.loc[common_index]
            returns_aligned = first_factor.returns.loc[common_index]

            # Get date range
            dates = pc_values_aligned.index.get_level_values('timestamp')
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = dates.max().strftime('%Y-%m-%d')

            # Create FactorData
            pc_factor = FactorData(
                factor_values=pc_values_aligned,
                returns=returns_aligned,
                factor_name=col,
                start_date=start_date,
                end_date=end_date,
                frequency=first_factor.frequency
            )

            pc_factors[col] = pc_factor

        return pc_factors


def perform_pca(factors: Dict[str, FactorData],
                n_components: Optional[int] = None,
                variance_threshold: float = 0.95) -> AnalysisResult:
    """
    Quick PCA analysis

    Args:
        factors: Dictionary of factors
        n_components: Number of components (None = auto)
        variance_threshold: Variance threshold for auto selection

    Returns:
        AnalysisResult with PCA components

    Example:
        >>> factors = {
        ...     'momentum': momentum_data,
        ...     'reversal': reversal_data,
        ...     'volatility': volatility_data
        ... }
        >>> result = perform_pca(factors, n_components=2)
        >>> print(result.summary)
        >>>
        >>> # Use PC factors for analysis
        >>> pc_factors = result.data['pc_factors']
        >>> pc1_factor = pc_factors['PC1']
    """
    analyzer = PCAAnalysis(n_components=n_components,
                          variance_threshold=variance_threshold)
    return analyzer.analyze(factors)
