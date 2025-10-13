"""
波动率因子定义

包含波动率相关的技术指标：
- ATR: 平均真实波幅
- BollingerBands: 布林带
- HistoricalVolatility: 历史波动率
"""

from factor_framework.nodes.atomic import Close, High, Low, Constant
from factor_framework.nodes.time_ops import Mean, Std, Ref
from factor_framework.nodes.math_ops import Sub, Add, Mul, Div, Abs, Max, Log, Sqrt, Pow


def ATR(window_length=14):
    """
    平均真实波幅 (Average True Range)

    计算公式：
    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    ATR = Mean(TR, window_length)

    参数：
        window_length: 计算窗口，默认14

    返回：
        FactorNode: ATR因子

    示例：
        >>> atr_factor = ATR(window_length=14)
        >>> config = FrequencyConfig(input_freq='daily', window_length=14, calc_freq='daily')
    """
    high = High()
    low = Low()
    close_prev = Ref(Close(), 1)

    # TR的三个候选值
    hl = high - low
    hc = Abs(high - close_prev)
    lc = Abs(low - close_prev)

    # TR = max(HL, HC, LC)
    tr = Max(Max(hl, hc), lc)

    # ATR = Mean(TR, window_length)
    atr = Mean(tr, window=window_length)

    return atr


def BollingerBands(window_length=20, num_std=2):
    """
    布林带 (Bollinger Bands)

    计算公式：
    Middle = Mean(Close, window_length)
    Upper = Middle + num_std * Std(Close, window_length)
    Lower = Middle - num_std * Std(Close, window_length)
    BBWidth = (Upper - Lower) / Middle

    参数：
        window_length: 计算窗口，默认20
        num_std: 标准差倍数，默认2

    返回：
        dict: {'upper': 上轨, 'middle': 中轨, 'lower': 下轨, 'width': 带宽}
    """
    close = Close()

    # 中轨 = 移动平均
    middle = Mean(close, window=window_length)

    # 标准差
    std = Std(close, window=window_length)

    # 上轨和下轨
    upper = Add(middle, Mul(Constant(num_std), std))
    lower = Sub(middle, Mul(Constant(num_std), std))

    # 带宽 = (上轨 - 下轨) / 中轨
    width = Div(Sub(upper, lower), middle)

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width
    }


def HistoricalVolatility(window_length=20):
    """
    历史波动率

    计算公式：
    Returns = (Close - Close_prev) / Close_prev
    HV = Std(Returns, window_length) * sqrt(252)  # 年化

    参数：
        window_length: 计算窗口，默认20

    返回：
        FactorNode: 历史波动率因子（年化）
    """
    close = Close()
    close_prev = Ref(close, 1)

    # 收益率
    returns = Div(Sub(close, close_prev), close_prev)

    # 收益率标准差
    std_returns = Std(returns, window=window_length)

    # 年化波动率（假设252个交易日）
    annual_vol = Mul(std_returns, Constant(252 ** 0.5))

    return annual_vol


def ParkinsonVolatility(window_length=20):
    """
    Parkinson波动率（基于高低价）

    计算公式：
    PV = sqrt(Mean((ln(High/Low))^2 / (4*ln(2)), window_length)) * sqrt(252)

    参数：
        window_length: 计算窗口，默认20

    返回：
        FactorNode: Parkinson波动率因子（年化）
    """
    high = High()
    low = Low()

    # ln(High/Low)
    hl_ratio = Log(Div(high, low))

    # (ln(High/Low))^2 / (4*ln(2))
    squared_ratio = Div(Pow(hl_ratio, Constant(2)), Constant(4 * 0.693147))

    # Mean(...) * sqrt(252)
    parkinson_vol = Mul(Sqrt(Mean(squared_ratio, window=window_length)), Constant(252 ** 0.5))

    return parkinson_vol
