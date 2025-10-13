"""
多因子分析模块

提供多因子相关性、组合优化和主成分分析功能。
"""

from .correlation import FactorCorrelation, compute_correlation
from .combination import FactorCombination, combine_factors
from .pca import PCAAnalysis, perform_pca

__all__ = [
    'FactorCorrelation',
    'compute_correlation',
    'FactorCombination',
    'combine_factors',
    'PCAAnalysis',
    'perform_pca'
]
