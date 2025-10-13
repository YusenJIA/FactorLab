"""
统一因子计算框架

这个框架实现了三层架构设计：
1. 抽象因子定义层：定义因子的抽象表达式树，完全不关心频率
2. 频率配置层：为抽象表达式赋予具体的时间含义
3. 执行引擎层：读取抽象因子表达式和频率配置，执行DAG计算

主要组件：
- FrequencyConfig: 频率配置类
- FactorNode: 因子节点抽象基类
- FactorEngine: 因子计算引擎
- 各种运算节点：AtomicData, Mean, Div, Rank等
"""

from .config import FrequencyConfig
from .engine import FactorEngine
from .nodes import *
from .data_loader import get_data

__version__ = "1.0.0"
__author__ = "ETF Strategy Framework"

__all__ = [
    'FrequencyConfig',
    'FactorEngine',
    'FactorNode',
    'AtomicData',
    'Close', 'Volume', 'High', 'Low', 'Open', 'Amount', 'Constant',
    'Add', 'Div', 'Mul', 'Sub', 'Abs',
    'Mean', 'Std', 'Sum', 'Max', 'Min', 'Ref',
    'Rank', 'Zscore',
    'get_data'
]