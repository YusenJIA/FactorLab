"""
单因子分析模块

提供单因子检验的核心功能：
- IC分析：因子与收益的相关性
- 分组回测：按因子值分组的收益表现
"""

from .ic_analysis import ICAnalyzer, compute_ic, batch_ic_analysis
from .portfolio_sorting import PortfolioSorter, portfolio_sorting_test

__all__ = [
    'ICAnalyzer',
    'compute_ic',
    'batch_ic_analysis',
    'PortfolioSorter',
    'portfolio_sorting_test'
]
