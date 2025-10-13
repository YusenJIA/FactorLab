"""
因子分析工具函数模块

提供数据准备和性能指标计算功能。
"""

from .data_prep import (
    compute_forward_returns,
    align_factor_and_returns,
    validate_multiindex,
    resample_to_frequency,
    split_train_test,
    winsorize_data,
    fill_missing_values,
    standardize_by_date,
    get_data_summary
)

from .metrics import (
    compute_ic_metrics,
    compute_returns_metrics,
    compute_max_drawdown,
    compute_turnover,
    compute_information_ratio,
    compute_sortino_ratio,
    compute_var,
    compute_cvar,
    compute_beta,
    compute_alpha,
    compute_all_metrics
)

__all__ = [
    # data_prep
    'compute_forward_returns',
    'align_factor_and_returns',
    'validate_multiindex',
    'resample_to_frequency',
    'split_train_test',
    'winsorize_data',
    'fill_missing_values',
    'standardize_by_date',
    'get_data_summary',
    # metrics
    'compute_ic_metrics',
    'compute_returns_metrics',
    'compute_max_drawdown',
    'compute_turnover',
    'compute_information_ratio',
    'compute_sortino_ratio',
    'compute_var',
    'compute_cvar',
    'compute_beta',
    'compute_alpha',
    'compute_all_metrics'
]
