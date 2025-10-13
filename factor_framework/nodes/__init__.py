"""
因子节点模块

包含所有类型的因子节点：
- base: FactorNode抽象基类
- atomic: 原子数据节点（Close, Volume等）
- math_ops: 数学运算节点（Add, Div等）
- time_ops: 时间聚合运算（Mean, Std等）
- cross_ops: 横截面运算（Rank, Zscore等）
"""

from .base import FactorNode
from .atomic import AtomicData, Close, Volume, High, Low, Open, Amount, Constant
from .math_ops import Add, Div, Mul, Sub, Abs, Log, Sqrt, Pow, Max as MathMax, Min as MathMin
from .time_ops import Mean, Std, Sum, Ref, Tsmax, Tsmin, EMA
from .cross_ops import Rank, Zscore

__all__ = [
    'FactorNode',
    'AtomicData',
    'Close', 'Volume', 'High', 'Low', 'Open', 'Amount', 'Constant',
    'Add', 'Div', 'Mul', 'Sub', 'Abs', 'Log', 'Sqrt', 'Pow',
    'Mean', 'Std', 'Sum', 'Ref', 'Tsmax', 'Tsmin', 'EMA',
    'Rank', 'Zscore',
    'Max', 'Min'  # 导出time_ops中的Max/Min
]

# 为了兼容，默认Max/Min使用时间序列版本
Max = Tsmax
Min = Tsmin