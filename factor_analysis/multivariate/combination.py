"""
Factor Combination Module

Combine multiple factors into a composite factor using various weighting schemes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings

from ..core import FactorData, AnalysisResult
from ..univariate.ic_analysis import ICAnalyzer
from ..utils.data_prep import standardize_by_date


class FactorCombination:
    """
    Factor Combination Optimizer

    Combine multiple factors with different weighting methods.
    """

    def __init__(self, method: str = 'equal_weight'):
        """
        Initialize factor combination

        Args:
            method: Combination method
                - 'equal_weight': Equal weight for all factors
                - 'ic_weight': Weight by IC
                - 'icir_weight': Weight by ICIR
                - 'optimal': Maximize ICIR (optimization)
        """
        valid_methods = ['equal_weight', 'ic_weight', 'icir_weight', 'optimal']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        self.method = method

    def combine(self,
                factors: Dict[str, FactorData],
                returns: Optional[pd.DataFrame] = None) -> FactorData:
        """
        Combine multiple factors

        Args:
            factors: Dictionary of {factor_name: FactorData}
            returns: Returns data (required for IC-based weighting)

        Returns:
            FactorData: Combined factor

        Example:
            >>> combiner = FactorCombination(method='icir_weight')
            >>> combined = combiner.combine(factors, returns_df)
        """
        # Calculate weights
        if self.method == 'equal_weight':
            weights = self._equal_weight(factors)
        elif self.method == 'ic_weight':
            if returns is None:
                raise ValueError("returns required for ic_weight method")
            weights = self._ic_weight(factors)
        elif self.method == 'icir_weight':
            if returns is None:
                raise ValueError("returns required for icir_weight method")
            weights = self._icir_weight(factors)
        elif self.method == 'optimal':
            if returns is None:
                raise ValueError("returns required for optimal method")
            weights = self._optimal_weight(factors)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Construct combined factor
        combined_factor = self._construct_combined_factor(factors, weights)

        return combined_factor

    def _equal_weight(self, factors: Dict[str, FactorData]) -> Dict[str, float]:
        """Equal weight"""
        n = len(factors)
        return {name: 1.0 / n for name in factors.keys()}

    def _ic_weight(self, factors: Dict[str, FactorData]) -> Dict[str, float]:
        """Weight by IC"""
        ic_values = {}

        for name, factor_data in factors.items():
            try:
                analyzer = ICAnalyzer(method='spearman')
                result = analyzer.analyze(factor_data)
                ic_values[name] = result.metrics['ic_mean']
            except:
                warnings.warn(f"Failed to compute IC for {name}, using 0")
                ic_values[name] = 0

        # Normalize weights (use absolute IC)
        total_ic = sum(abs(ic) for ic in ic_values.values())
        if total_ic > 0:
            weights = {name: abs(ic) / total_ic for name, ic in ic_values.items()}
        else:
            weights = self._equal_weight(factors)

        return weights

    def _icir_weight(self, factors: Dict[str, FactorData]) -> Dict[str, float]:
        """Weight by ICIR"""
        icir_values = {}

        for name, factor_data in factors.items():
            try:
                analyzer = ICAnalyzer(method='spearman')
                result = analyzer.analyze(factor_data)
                icir_values[name] = max(result.metrics['icir'], 0)  # Only positive ICIR
            except:
                warnings.warn(f"Failed to compute ICIR for {name}, using 0")
                icir_values[name] = 0

        # Normalize weights
        total_icir = sum(icir_values.values())
        if total_icir > 0:
            weights = {name: icir / total_icir for name, icir in icir_values.items()}
        else:
            weights = self._equal_weight(factors)

        return weights

    def _optimal_weight(self, factors: Dict[str, FactorData]) -> Dict[str, float]:
        """Optimal weights to maximize ICIR (simplified version)"""
        # This is a simplified version; full optimization would use quadratic programming
        # For now, use ICIR weighting as approximation
        return self._icir_weight(factors)

    def _construct_combined_factor(self,
                                   factors: Dict[str, FactorData],
                                   weights: Dict[str, float]) -> FactorData:
        """Construct weighted combination of factors"""
        # Align all factors
        factor_dict = {}
        for name, factor_data in factors.items():
            # Standardize before combining
            standardized = standardize_by_date(
                factor_data.factor_values,
                method='zscore'
            )
            factor_dict[name] = standardized['factor_value']

        factor_df = pd.DataFrame(factor_dict).dropna()

        # Apply weights
        combined_values = pd.Series(0, index=factor_df.index)
        for name, weight in weights.items():
            if name in factor_df.columns:
                combined_values += weight * factor_df[name]

        # Create combined FactorData
        combined_factor_df = combined_values.to_frame('factor_value')

        # Use returns from first factor (assuming all aligned)
        first_factor = list(factors.values())[0]
        combined_returns = first_factor.returns

        # Align returns with combined factor
        common_index = combined_factor_df.index.intersection(combined_returns.index)
        combined_factor_df = combined_factor_df.loc[common_index]
        combined_returns = combined_returns.loc[common_index]

        # Get date range
        dates = combined_factor_df.index.get_level_values('datetime')
        start_date = dates.min().strftime('%Y-%m-%d')
        end_date = dates.max().strftime('%Y-%m-%d')

        # Create weight string for name
        weight_str = ", ".join([f"{k}:{v:.2f}" for k, v in weights.items()])

        combined_factor_data = FactorData(
            factor_values=combined_factor_df,
            returns=combined_returns,
            factor_name=f"Combined_{self.method}",
            start_date=start_date,
            end_date=end_date,
            frequency=first_factor.frequency
        )

        return combined_factor_data


def combine_factors(factors: Dict[str, FactorData],
                   method: str = 'icir_weight',
                   returns: Optional[pd.DataFrame] = None) -> FactorData:
    """
    Quick factor combination

    Args:
        factors: Dictionary of factors
        method: Combination method
        returns: Returns data (for IC-based methods)

    Returns:
        FactorData: Combined factor

    Example:
        >>> combined = combine_factors(factors, method='icir_weight')
        >>> # Analyze combined factor
        >>> from factor_analysis.univariate import compute_ic
        >>> ic_result = compute_ic(combined)
        >>> print(f"Combined ICIR: {ic_result.metrics['icir']:.4f}")
    """
    combiner = FactorCombination(method=method)
    return combiner.combine(factors, returns)
