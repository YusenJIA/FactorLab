"""
因子分析模块

基于 factor_framework 的因子分析系统，提供完整的因子检验、评估和组合优化功能。

核心组件：
- FactorData: 标准化的因子数据容器（支持从FactorEngine输出自动转换）
- AnalysisResult: 统一的分析结果容器
- ICAnalyzer: IC分析器（因子与收益相关性）
- PortfolioSorter: 分组回测器（按因子值分组的收益表现）

快速开始：
    >>> from factor_framework import FactorEngine, FrequencyConfig, Close, Volume, Mean
    >>> from factor_analysis import FactorData, compute_ic, portfolio_sorting_test
    >>>
    >>> # 1. 使用 factor_framework 计算因子
    >>> factor = Close() / Mean(Volume(), window=20)
    >>> config = FrequencyConfig(input_freq='daily')
    >>> engine = FactorEngine()
    >>> factor_df = engine.compute_factor(factor, assets, dates, config)
    >>>
    >>> # 2. 转换为 FactorData
    >>> factor_data = FactorData.from_engine_output(
    ...     factor_df, returns_df, '价格成交量比'
    ... )
    >>>
    >>> # 3. 一行代码完成分析
    >>> ic_result = compute_ic(factor_data, method='spearman')
    >>> sorting_result = portfolio_sorting_test(factor_data, n_quantiles=5)
"""

from .core import FactorData, AnalysisResult
from .univariate import (
    ICAnalyzer, compute_ic, batch_ic_analysis,
    PortfolioSorter, portfolio_sorting_test
)
from .multivariate import (
    FactorCorrelation, compute_correlation,
    FactorCombination, combine_factors,
    PCAAnalysis, perform_pca
)
from .performance import (
    TimeSeriesAnalysis, analyze_by_period, rolling_analysis,
    CrossSectionAnalysis, analyze_by_groups, analyze_by_quantiles
)
from .risk import (
    OutOfSampleTest, out_of_sample_test,
    RobustnessTest, robustness_test
)

__version__ = "1.0.0"
__author__ = "ETF Strategy Framework"

__all__ = [
    # Core
    'FactorData',
    'AnalysisResult',
    # Univariate
    'ICAnalyzer',
    'compute_ic',
    'batch_ic_analysis',
    'PortfolioSorter',
    'portfolio_sorting_test',
    # Multivariate
    'FactorCorrelation',
    'compute_correlation',
    'FactorCombination',
    'combine_factors',
    'PCAAnalysis',
    'perform_pca',
    # Performance
    'TimeSeriesAnalysis',
    'analyze_by_period',
    'rolling_analysis',
    'CrossSectionAnalysis',
    'analyze_by_groups',
    'analyze_by_quantiles',
    # Risk
    'OutOfSampleTest',
    'out_of_sample_test',
    'RobustnessTest',
    'robustness_test'
]
