"""
价值因子定义

包含与估值相关的因子：
- PriceToMA: 价格与均线比率
- PercentFromHigh: 距离历史高点的百分比
"""

from factor_framework.nodes.atomic import Close, High, Low, Constant
from factor_framework.nodes.time_ops import Mean, Tsmax, Tsmin
from factor_framework.nodes.math_ops import Div, Sub, Add


def PriceToMA(window_length=20):
    """
    价格与移动平均线比率

    计算公式：
    PriceToMA = Close / Mean(Close, window_length)

    参数：
        window_length: 均线周期，默认20

    返回：
        FactorNode: 价格均线比因子

    示例：
        >>> ptma_factor = PriceToMA(window_length=20)
        >>> config = FrequencyConfig(input_freq='daily', window_length=20, calc_freq='daily')
    """
    close = Close()
    ma = Mean(close, window=window_length)

    return Div(close, ma)


def PercentFromHigh(window_length=252):
    """
    距离历史高点的百分比

    计算公式：
    PercentFromHigh = (Max(Close, window_length) - Close) / Max(Close, window_length)

    参数：
        window_length: 回看窗口，默认252（一年）

    返回：
        FactorNode: 距离高点百分比因子
    """
    close = Close()
    highest = Tsmax(close, window=window_length)

    return Div(Sub(highest, close), highest)


def PriceRange(window_length=20):
    """
    价格区间位置

    计算公式：
    PriceRange = (Close - Min(Low)) / (Max(High) - Min(Low))

    参数：
        window_length: 计算窗口，默认20

    返回：
        FactorNode: 价格在区间中的位置（0-1）
    """
    close = Close()
    highest = Tsmax(High(), window=window_length)
    lowest = Tsmin(Low(), window=window_length)

    price_range = Div(Sub(close, lowest), Add(Sub(highest, lowest), Constant(1e-10)))

    return price_range
