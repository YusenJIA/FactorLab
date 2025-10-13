"""
动量类因子定义

包含常用的动量技术指标：
- RSI: 相对强弱指标
- MACD: 指数移动平均线差异
- Momentum: 简单动量
"""

from factor_framework.nodes.base import FactorNode
from factor_framework.nodes.atomic import Close, Volume, Constant
from factor_framework.nodes.time_ops import Mean, Ref, EMA
from factor_framework.nodes.math_ops import Add, Sub, Div, Abs, Max, Min


def RSI(window_length=14):
    """
    相对强弱指标 (Relative Strength Index)

    计算公式：
    RSI = 100 - (100 / (1 + RS))
    其中 RS = 平均涨幅 / 平均跌幅

    参数：
        window_length: 计算窗口，默认14

    返回：
        FactorNode: RSI因子节点

    示例：
        >>> rsi_factor = RSI(window_length=14)
        >>> config = FrequencyConfig(input_freq='daily', window_length=14, calc_freq='daily')
        >>> results = engine.compute_factor(rsi_factor, ['510050.SH'], '2025-01-01', '2025-01-31', config)
    """
    # 价格变化
    delta = Close() - Ref(Close(), 1)

    # 涨幅和跌幅
    gain = Max(delta, Constant(0))
    loss = Abs(Min(delta, Constant(0)))

    # 平均涨幅和平均跌幅
    avg_gain = Mean(gain, window=window_length)
    avg_loss = Mean(loss, window=window_length)

    # RS = 平均涨幅 / 平均跌幅
    rs = avg_gain / (avg_loss + Constant(1e-10))  # 避免除零

    # RSI = 100 - (100 / (1 + RS))
    rsi = Constant(100) - (Constant(100) / (Constant(1) + rs))

    return rsi


def MACD(fast_period=12, slow_period=26, signal_period=9):
    """
    指数移动平均线差异 (Moving Average Convergence Divergence)

    计算公式：
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    参数：
        fast_period: 快速EMA周期，默认12
        slow_period: 慢速EMA周期，默认26
        signal_period: 信号线周期，默认9

    返回：
        dict: {'macd': MACD线, 'signal': 信号线, 'histogram': 柱状图}

    示例：
        >>> macd_dict = MACD(fast_period=12, slow_period=26, signal_period=9)
        >>> macd_factor = macd_dict['macd']
        >>> config = FrequencyConfig(input_freq='daily', window_length=26, calc_freq='daily')
    """
    # 快速和慢速EMA
    ema_fast = EMA(Close(), window=fast_period)
    ema_slow = EMA(Close(), window=slow_period)

    # MACD线
    macd_line = ema_fast - ema_slow

    # 信号线（MACD的EMA）
    signal_line = EMA(macd_line, window=signal_period)

    # 柱状图
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def Momentum(window_length=20):
    """
    简单动量因子

    计算公式：
    Momentum = (当前价格 / N天前价格) - 1

    参数：
        window_length: 回看周期，默认20

    返回：
        FactorNode: 动量因子节点

    示例：
        >>> mom_factor = Momentum(window_length=20)
        >>> config = FrequencyConfig(input_freq='daily', window_length=20, calc_freq='daily')
    """
    current_price = Close()
    past_price = Ref(Close(), window_length)

    # (P_t / P_{t-n}) - 1
    momentum = (current_price / past_price) - Constant(1)

    return momentum


def RateOfChange(window_length=10):
    """
    价格变化率 (Rate of Change)

    计算公式：
    ROC = ((当前价格 - N天前价格) / N天前价格) * 100

    参数：
        window_length: 回看周期，默认10

    返回：
        FactorNode: ROC因子节点
    """
    current_price = Close()
    past_price = Ref(Close(), window_length)

    roc = ((current_price - past_price) / past_price) * Constant(100)

    return roc
