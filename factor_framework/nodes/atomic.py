"""
原子数据节点实现

这些是设计文档中"原子数据式"的实现，代表基础市场数据。
原子节点是因子表达式树的叶子节点，直接从数据源获取数据。

支持的原子数据类型：
- Close: 收盘价
- Open: 开盘价
- High: 最高价
- Low: 最低价
- Volume: 成交量
- Amount: 成交额
- Constant: 常数值
"""

from typing import Any, Optional
import pandas as pd
import numpy as np
from .base import FactorNode
from ..config import FrequencyConfig


class AtomicData(FactorNode):
    """
    原子数据抽象基类

    所有原子数据节点的基类，代表从数据源直接获取的基础数据。
    按照设计文档，原子数据是因子表达式的基础构建块。
    """

    def __init__(self, field_name: str, name: Optional[str] = None):
        """
        初始化原子数据节点

        Args:
            field_name: 数据字段名（如'close', 'volume'等）
            name: 节点显示名称
        """
        super().__init__(name or field_name.capitalize())
        self.field_name = field_name.lower()

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        从数据中提取指定字段的值

        Args:
            data: 包含OHLCV数据的DataFrame
            config: 频率配置
            timestamp: 当前时间点
            **kwargs: 其他参数

        Returns:
            指定字段的数据Series
        """
        self.validate_inputs(data, config)

        # 检查字段是否存在
        if self.field_name not in data.columns:
            raise ValueError(f"数据中不存在字段: {self.field_name}")

        # 获取截止到当前时间点的数据
        if isinstance(data.index, pd.DatetimeIndex):
            # 如果数据已经按时间索引
            mask = data.index <= timestamp
            filtered_data = data.loc[mask]
        else:
            # 如果datetime是列
            mask = data['datetime'] <= timestamp
            filtered_data = data.loc[mask]

        if filtered_data.empty:
            raise ValueError(f"在时间点 {timestamp} 之前没有可用数据")

        # 如果有多个资产，需要返回截面数据
        if 'code' in filtered_data.columns:
            # 按资产分组，取每个资产的最新值
            latest_data = filtered_data.groupby('code')[self.field_name].last()
            return latest_data
        else:
            # 单资产情况，返回最新值
            return pd.Series([filtered_data[self.field_name].iloc[-1]],
                           index=[0])

    def __repr__(self) -> str:
        return f"{self.name}"


class Close(AtomicData):
    """收盘价节点"""

    def __init__(self):
        super().__init__('close', 'Close')


class Open(AtomicData):
    """开盘价节点"""

    def __init__(self):
        super().__init__('open', 'Open')


class High(AtomicData):
    """最高价节点"""

    def __init__(self):
        super().__init__('high', 'High')


class Low(AtomicData):
    """最低价节点"""

    def __init__(self):
        super().__init__('low', 'Low')


class Volume(AtomicData):
    """成交量节点"""

    def __init__(self):
        super().__init__('volume', 'Volume')


class Amount(AtomicData):
    """成交额节点"""

    def __init__(self):
        super().__init__('amount', 'Amount')


class Constant(FactorNode):
    """
    常数节点

    代表一个固定的数值常数，用于与其他因子进行数学运算。
    例如：Close() / Constant(100) 表示收盘价除以100
    """

    def __init__(self, value: float, name: Optional[str] = None):
        """
        初始化常数节点

        Args:
            value: 常数值
            name: 节点名称，默认为常数值的字符串表示
        """
        super().__init__(name or f"Const({value})")
        self.value = float(value)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        返回常数值

        对于常数节点，不管输入什么数据，都返回固定的常数值。
        返回的Series长度与数据中的资产数量匹配。
        """
        # 确定资产数量
        if 'code' in data.columns:
            unique_codes = data['code'].unique()
            return pd.Series([self.value] * len(unique_codes), index=unique_codes)
        else:
            # 单资产情况
            return pd.Series([self.value], index=[0])

    def __repr__(self) -> str:
        return f"Const({self.value})"


class Returns(FactorNode):
    """
    收益率节点

    计算价格的收益率，支持不同周期的收益率计算。
    这是一个组合节点，内部使用Close和Ref节点。
    """

    def __init__(self, periods: int = 1, name: Optional[str] = None):
        """
        初始化收益率节点

        Args:
            periods: 收益率周期，1表示单期收益率
            name: 节点名称
        """
        super().__init__(name or f"Returns({periods})")
        self.periods = periods

        # 创建依赖节点
        self.close = Close()
        from .time_ops import Ref
        self.close_lag = Ref(self.close, periods)

        # 添加依赖关系
        self.add_dependency(self.close)
        self.add_dependency(self.close_lag)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        计算收益率

        Returns = (Close_t - Close_{t-periods}) / Close_{t-periods}
        """
        # 计算当前收盘价和滞后收盘价
        current_close = self.close.compute_with_cache(data, config, timestamp, **kwargs)
        lag_close = self.close_lag.compute_with_cache(data, config, timestamp, **kwargs)

        # 计算收益率
        returns = (current_close - lag_close) / lag_close

        # 处理除零情况
        returns = returns.replace([np.inf, -np.inf], np.nan)

        return returns

    def __repr__(self) -> str:
        return f"Returns({self.periods})"


# 预定义的原子数据实例
# 这样用户可以直接导入使用，避免重复创建
_CLOSE = Close()
_OPEN = Open()
_HIGH = High()
_LOW = Low()
_VOLUME = Volume()
_AMOUNT = Amount()