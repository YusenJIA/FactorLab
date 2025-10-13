"""
预定义因子库

这个模块包含常用的技术指标和因子定义，所有因子都是基于 factor_framework.nodes 构建的。

分类：
- momentum: 动量类因子（RSI, MACD, MOM等）
- liquidity: 流动性因子（成交量相关）
- volatility: 波动率因子（ATR, Bollinger Bands等）
- value: 价值因子（PE相关等）
"""

from .momentum import RSI, MACD, Momentum
from .liquidity import PriceToVolume, VolumeRatio
from .volatility import ATR, BollingerBands, HistoricalVolatility

__all__ = [
    # 动量类
    'RSI',
    'MACD',
    'Momentum',

    # 流动性类
    'PriceToVolume',
    'VolumeRatio',

    # 波动率类
    'ATR',
    'BollingerBands',
    'HistoricalVolatility',
]
