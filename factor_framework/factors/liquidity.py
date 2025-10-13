"""
流动性因子定义

包含与成交量和流动性相关的因子：
- PriceToVolume: 价格成交量比
- VolumeRatio: 成交量比率
- AmihudIlliquidity: Amihud非流动性指标
"""

from factor_framework.nodes.atomic import Close, Volume, Constant
from factor_framework.nodes.time_ops import Mean, Ref
from factor_framework.nodes.math_ops import Div, Abs, Mul


def PriceToVolume(window_length=20):
    """
    价格与平均成交量比率

    计算公式：
    PriceToVolume = 收盘价 / Mean(成交量, window_length)

    参数：
        window_length: 平均成交量计算窗口，默认20

    返回：
        FactorNode: 价格成交量比因子

    示例：
        >>> ptv_factor = PriceToVolume(window_length=20)
        >>> config = FrequencyConfig(input_freq='daily', window_length=20, calc_freq='daily')
    """
    price = Close()
    avg_volume = Mean(Volume(), window=window_length)

    return Div(price, avg_volume)


def VolumeRatio(short_window=5, long_window=20):
    """
    成交量比率

    计算公式：
    VolumeRatio = Mean(成交量, short_window) / Mean(成交量, long_window)

    参数：
        short_window: 短期窗口，默认5
        long_window: 长期窗口，默认20

    返回：
        FactorNode: 成交量比率因子
    """
    short_avg_vol = Mean(Volume(), window=short_window)
    long_avg_vol = Mean(Volume(), window=long_window)

    return Div(short_avg_vol, long_avg_vol)


def AmihudIlliquidity(window_length=20):
    """
    Amihud非流动性指标

    计算公式：
    Amihud = Mean(|收益率| / 成交额, window_length)

    参数：
        window_length: 计算窗口，默认20

    返回：
        FactorNode: Amihud非流动性因子
    """
    # 日收益率
    returns = Div(Sub(Close(), Ref(Close(), 1)), Ref(Close(), 1))

    # 成交额 = 价格 * 成交量
    turnover = Mul(Close(), Volume())

    # |收益率| / 成交额
    daily_illiq = Div(Abs(returns), Add(turnover, Constant(1e-10)))

    # 平均非流动性
    amihud = Mean(daily_illiq, window=window_length)

    return amihud


def RelativeVolume(window_length=20):
    """
    相对成交量

    计算公式：
    RelativeVolume = 当前成交量 / Mean(成交量, window_length)

    参数：
        window_length: 计算窗口，默认20

    返回：
        FactorNode: 相对成交量因子
    """
    current_vol = Volume()
    avg_vol = Mean(Volume(), window=window_length)

    return Div(current_vol, avg_vol)
