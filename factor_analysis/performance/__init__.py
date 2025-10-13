"""
Performance Evaluation Module

Analyze factor performance across different dimensions.
"""

from .time_series import TimeSeriesAnalysis, analyze_by_period, rolling_analysis
from .cross_section import CrossSectionAnalysis, analyze_by_groups, analyze_by_quantiles

__all__ = [
    'TimeSeriesAnalysis',
    'analyze_by_period',
    'rolling_analysis',
    'CrossSectionAnalysis',
    'analyze_by_groups',
    'analyze_by_quantiles'
]
