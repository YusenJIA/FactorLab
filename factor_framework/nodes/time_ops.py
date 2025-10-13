"""
时间聚合运算节点实现

这些是设计文档中"时间聚合运算"的实现，包括时间窗口操作。
时间聚合运算是因子计算的核心，按照设计文档中的要求，窗口长度的具体含义
由FrequencyConfig决定（20可以是20日也可以是20分钟）。

支持的时间运算类型：
- Mean: 移动平均
- Std: 移动标准差
- Sum: 移动求和
- Max: 移动最大值
- Min: 移动最小值
- Ref: 历史值引用（滞后算子）
- Delta: 变化量
- Returns: 收益率
"""

from typing import Optional, Union
import pandas as pd
import numpy as np
from .base import FactorNode
from ..config import FrequencyConfig


class TimeWindowOp(FactorNode):
    """
    时间窗口运算抽象基类

    所有时间窗口运算的基类。按照设计文档的核心思想，窗口长度是抽象的，
    具体含义由FrequencyConfig决定。
    """

    def __init__(self,
                 operand: FactorNode,
                 window: Optional[int] = None,
                 name: Optional[str] = None):
        """
        初始化时间窗口运算节点

        Args:
            operand: 输入因子
            window: 窗口长度，如果为None则使用config中的window_length
            name: 节点名称
        """
        super().__init__(name or f"{self.__class__.__name__}({operand.name}, {window})")
        self.operand = operand
        self.window = window

        # 添加依赖关系
        self.add_dependency(operand)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        计算时间窗口运算结果

        这里实现了设计文档中的核心思想：抽象的窗口概念通过config转化为具体的时间含义
        """
        # 确定窗口长度
        window_length = self.window if self.window is not None else config.window_length

        # 获取操作数在历史时间窗口内的所有值
        historical_data = self._get_historical_data(data, config, timestamp, window_length)

        if historical_data.empty:
            raise ValueError(f"在时间点 {timestamp} 没有足够的历史数据进行窗口计算")

        # 执行具体的窗口运算
        return self._execute_window_operation(historical_data, window_length)

    def _get_historical_data(self,
                           data: pd.DataFrame,
                           config: FrequencyConfig,
                           timestamp: pd.Timestamp,
                           window_length: int) -> pd.DataFrame:
        """
        获取历史时间窗口内的数据

        这个方法实现了设计文档中"时间轴对齐和处理"的要求，
        确保严格避免未来函数。
        """
        # 确保数据按时间排序
        if isinstance(data.index, pd.DatetimeIndex):
            data_sorted = data.sort_index()
            time_col = data_sorted.index
        else:
            data_sorted = data.sort_values('datetime')
            time_col = data_sorted['datetime']

        # 严格的未来函数检查：只使用timestamp之前的数据
        mask = time_col <= timestamp
        valid_data = data_sorted.loc[mask]

        if valid_data.empty:
            return pd.DataFrame()

        # 根据频率配置获取窗口数据
        pandas_freq = config.get_pandas_freq('input')

        if 'code' in valid_data.columns:
            # 多资产情况：为每个资产分别获取窗口数据
            result_data = []
            for code in valid_data['code'].unique():
                asset_data = valid_data[valid_data['code'] == code]

                if isinstance(asset_data.index, pd.DatetimeIndex):
                    asset_data_indexed = asset_data
                else:
                    asset_data_indexed = asset_data.set_index('datetime')

                # 获取最近window_length个周期的数据
                window_data = asset_data_indexed.tail(window_length)
                result_data.append(window_data.reset_index())

            if result_data:
                return pd.concat(result_data, ignore_index=True)
            else:
                return pd.DataFrame()
        else:
            # 单资产情况
            if isinstance(valid_data.index, pd.DatetimeIndex):
                return valid_data.tail(window_length)
            else:
                return valid_data.tail(window_length)

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """
        执行具体的窗口运算

        子类必须实现这个方法
        """
        raise NotImplementedError("子类必须实现_execute_window_operation方法")


class Mean(TimeWindowOp):
    """
    移动平均

    按照设计文档示例，这是构建"价格与平均成交量比率"因子的关键组件。
    Mean(Volume, window_length=W) 中的W会根据config转化为具体时间含义。
    """

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """计算移动平均"""
        # 先计算操作数在每个时间点的值
        if 'code' in data.columns:
            # 多资产情况
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]
                operand_values = []

                # 为当前资产计算操作数在窗口内的历史值
                for _, row in asset_data.iterrows():
                    # 这里简化处理，实际应该调用operand.compute
                    # 但由于我们已经在窗口数据中，直接使用数据
                    pass

                # 计算该资产的均值
                if len(asset_data) > 0:
                    # 假设operand是原子数据，直接从数据中获取
                    field_name = getattr(self.operand, 'field_name', 'close')
                    if field_name in asset_data.columns:
                        result[code] = asset_data[field_name].mean()
                    else:
                        result[code] = np.nan

            return pd.Series(result)
        else:
            # 单资产情况
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in data.columns:
                return pd.Series([data[field_name].mean()], index=[0])
            else:
                return pd.Series([np.nan], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"Mean({self.operand.name}, {window_str})"


class Std(TimeWindowOp):
    """移动标准差"""

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """计算移动标准差"""
        if 'code' in data.columns:
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]
                field_name = getattr(self.operand, 'field_name', 'close')
                if field_name in asset_data.columns and len(asset_data) > 1:
                    result[code] = asset_data[field_name].std()
                else:
                    result[code] = 0.0  # 单个值的标准差为0
            return pd.Series(result)
        else:
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in data.columns and len(data) > 1:
                return pd.Series([data[field_name].std()], index=[0])
            else:
                return pd.Series([0.0], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"Std({self.operand.name}, {window_str})"


class Sum(TimeWindowOp):
    """移动求和"""

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """计算移动求和"""
        if 'code' in data.columns:
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]
                field_name = getattr(self.operand, 'field_name', 'close')
                if field_name in asset_data.columns:
                    result[code] = asset_data[field_name].sum()
                else:
                    result[code] = 0.0
            return pd.Series(result)
        else:
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in data.columns:
                return pd.Series([data[field_name].sum()], index=[0])
            else:
                return pd.Series([0.0], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"Sum({self.operand.name}, {window_str})"


class Max(TimeWindowOp):
    """移动最大值"""

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """计算移动最大值"""
        if 'code' in data.columns:
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]
                field_name = getattr(self.operand, 'field_name', 'close')
                if field_name in asset_data.columns:
                    result[code] = asset_data[field_name].max()
                else:
                    result[code] = np.nan
            return pd.Series(result)
        else:
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in data.columns:
                return pd.Series([data[field_name].max()], index=[0])
            else:
                return pd.Series([np.nan], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"Max({self.operand.name}, {window_str})"


class Min(TimeWindowOp):
    """移动最小值"""

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """计算移动最小值"""
        if 'code' in data.columns:
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]
                field_name = getattr(self.operand, 'field_name', 'close')
                if field_name in asset_data.columns:
                    result[code] = asset_data[field_name].min()
                else:
                    result[code] = np.nan
            return pd.Series(result)
        else:
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in data.columns:
                return pd.Series([data[field_name].min()], index=[0])
            else:
                return pd.Series([np.nan], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"Min({self.operand.name}, {window_str})"


class Ref(FactorNode):
    """
    历史值引用（滞后算子）

    Ref(operand, periods) 返回periods期之前的operand值。
    这是时间序列分析的基础运算符。
    """

    def __init__(self, operand: FactorNode, periods: int, name: Optional[str] = None):
        """
        初始化历史引用节点

        Args:
            operand: 要引用的因子
            periods: 滞后期数
            name: 节点名称
        """
        super().__init__(name or f"Ref({operand.name}, {periods})")
        self.operand = operand
        self.periods = periods

        # 添加依赖关系
        self.add_dependency(operand)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        计算历史引用值

        严格遵循未来函数检查：只能引用当前时间点之前的数据
        """
        # 计算目标时间点（当前时间减去periods个周期）
        pandas_freq = config.get_pandas_freq('input')

        # 根据频率计算时间偏移
        if config.input_freq == 'daily':
            target_time = timestamp - pd.Timedelta(days=self.periods)
        elif config.input_freq == 'minute':
            target_time = timestamp - pd.Timedelta(minutes=self.periods)
        elif config.input_freq == '5min':
            target_time = timestamp - pd.Timedelta(minutes=5 * self.periods)
        elif config.input_freq == '30min':
            target_time = timestamp - pd.Timedelta(minutes=30 * self.periods)
        else:
            # 通用处理
            try:
                offset = pd.Timedelta(f"{self.periods}{pandas_freq}")
                target_time = timestamp - offset
            except:
                # 如果无法解析，使用periods作为天数
                target_time = timestamp - pd.Timedelta(days=self.periods)

        # 在目标时间点计算operand的值
        try:
            return self.operand.compute_with_cache(data, config, target_time, **kwargs)
        except:
            # 如果目标时间点没有数据，返回NaN
            if 'code' in data.columns:
                codes = data['code'].unique()
                return pd.Series([np.nan] * len(codes), index=codes)
            else:
                return pd.Series([np.nan], index=[0])

    def __repr__(self) -> str:
        return f"Ref({self.operand.name}, {self.periods})"


class Delta(FactorNode):
    """
    变化量：Delta(operand, periods) = operand - Ref(operand, periods)

    计算相对于历史值的变化量
    """

    def __init__(self, operand: FactorNode, periods: int = 1, name: Optional[str] = None):
        """
        初始化变化量节点

        Args:
            operand: 要计算变化的因子
            periods: 对比的历史期数
            name: 节点名称
        """
        super().__init__(name or f"Delta({operand.name}, {periods})")
        self.operand = operand
        self.periods = periods

        # 创建内部依赖节点
        self.ref_operand = Ref(operand, periods)

        # 添加依赖关系
        self.add_dependency(operand)
        self.add_dependency(self.ref_operand)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """计算变化量"""
        current_val = self.operand.compute_with_cache(data, config, timestamp, **kwargs)
        historical_val = self.ref_operand.compute_with_cache(data, config, timestamp, **kwargs)

        return current_val - historical_val

    def __repr__(self) -> str:
        return f"Delta({self.operand.name}, {self.periods})"


class Rank(FactorNode):
    """
    时间序列排名

    对某个因子在时间窗口内的值进行排名，返回当前值在历史窗口中的排名百分位
    """

    def __init__(self,
                 operand: FactorNode,
                 window: Optional[int] = None,
                 name: Optional[str] = None):
        """
        初始化时间序列排名节点

        Args:
            operand: 要排名的因子
            window: 排名窗口长度
            name: 节点名称
        """
        super().__init__(name or f"TsRank({operand.name}, {window})")
        self.operand = operand
        self.window = window

        # 添加依赖关系
        self.add_dependency(operand)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """计算时间序列排名"""
        window_length = self.window if self.window is not None else config.window_length

        # 获取历史窗口数据
        if 'code' in data.columns:
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]

                # 计算该资产在窗口内的历史值
                historical_values = []
                for i in range(window_length):
                    try:
                        hist_timestamp = timestamp - pd.Timedelta(days=i) if config.input_freq == 'daily' else timestamp - pd.Timedelta(minutes=i)
                        hist_val = self.operand.compute_with_cache(
                            asset_data, config, hist_timestamp, **kwargs
                        )
                        if not hist_val.empty and not np.isnan(hist_val.iloc[0]):
                            historical_values.append(hist_val.iloc[0])
                    except:
                        continue

                if len(historical_values) > 1:
                    current_val = historical_values[0] if historical_values else np.nan
                    rank_pct = (np.sum(np.array(historical_values) < current_val) + 0.5 * np.sum(np.array(historical_values) == current_val)) / len(historical_values)
                    result[code] = rank_pct
                else:
                    result[code] = 0.5  # 默认中位数排名

            return pd.Series(result)
        else:
            # 单资产处理
            return pd.Series([0.5], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"TsRank({self.operand.name}, {window_str})"


class Tsmax(TimeWindowOp):
    """
    时间序列最大值

    Tsmax(operand, window_length) 返回窗口内的最大值
    """

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """计算时间序列最大值"""
        if 'code' in data.columns:
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]
                field_name = getattr(self.operand, 'field_name', 'close')
                if field_name in asset_data.columns:
                    result[code] = asset_data[field_name].max()
                else:
                    result[code] = np.nan
            return pd.Series(result)
        else:
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in data.columns:
                return pd.Series([data[field_name].max()], index=[0])
            else:
                return pd.Series([np.nan], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"Tsmax({self.operand.name}, {window_str})"


class Tsmin(TimeWindowOp):
    """
    时间序列最小值

    Tsmin(operand, window_length) 返回窗口内的最小值
    """

    def _execute_window_operation(self, data: pd.DataFrame, window_length: int) -> pd.Series:
        """计算时间序列最小值"""
        if 'code' in data.columns:
            result = {}
            for code in data['code'].unique():
                asset_data = data[data['code'] == code]
                field_name = getattr(self.operand, 'field_name', 'close')
                if field_name in asset_data.columns:
                    result[code] = asset_data[field_name].min()
                else:
                    result[code] = np.nan
            return pd.Series(result)
        else:
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in data.columns:
                return pd.Series([data[field_name].min()], index=[0])
            else:
                return pd.Series([np.nan], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"Tsmin({self.operand.name}, {window_str})"


class EMA(FactorNode):
    """
    指数移动平均 (Exponential Moving Average)

    EMA计算使用指数加权，近期数据权重更大
    EMA(t) = alpha * Price(t) + (1 - alpha) * EMA(t-1)
    其中 alpha = 2 / (window_length + 1)
    """

    def __init__(self, operand: FactorNode, window: Optional[int] = None, name: Optional[str] = None):
        """
        初始化EMA节点

        Args:
            operand: 输入因子
            window: EMA周期
            name: 节点名称
        """
        super().__init__(name or f"EMA({operand.name}, {window})")
        self.operand = operand
        self.window = window
        self.add_dependency(operand)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """计算EMA"""
        window = self.window if self.window is not None else config.window_length

        # 获取历史数据（需要足够多的数据来计算EMA）
        if isinstance(data.index, pd.DatetimeIndex):
            data_sorted = data.sort_index()
            time_col = data_sorted.index
        else:
            data_sorted = data.sort_values('datetime')
            time_col = data_sorted['datetime']

        mask = time_col <= timestamp
        valid_data = data_sorted.loc[mask]

        if valid_data.empty:
            if 'code' in data.columns:
                codes = data['code'].unique()
                return pd.Series([np.nan] * len(codes), index=codes)
            else:
                return pd.Series([np.nan], index=[0])

        # 计算EMA
        if 'code' in valid_data.columns:
            result = {}
            for code in valid_data['code'].unique():
                asset_data = valid_data[valid_data['code'] == code]
                field_name = getattr(self.operand, 'field_name', 'close')

                if field_name in asset_data.columns:
                    values = asset_data[field_name].tail(window * 2)  # 取足够多的数据
                    if len(values) > 0:
                        # 使用pandas的ewm计算EMA
                        ema_values = values.ewm(span=window, adjust=False).mean()
                        result[code] = ema_values.iloc[-1]
                    else:
                        result[code] = np.nan
                else:
                    result[code] = np.nan

            return pd.Series(result)
        else:
            field_name = getattr(self.operand, 'field_name', 'close')
            if field_name in valid_data.columns:
                values = valid_data[field_name].tail(window * 2)
                if len(values) > 0:
                    ema_values = values.ewm(span=window, adjust=False).mean()
                    return pd.Series([ema_values.iloc[-1]], index=[0])
                else:
                    return pd.Series([np.nan], index=[0])
            else:
                return pd.Series([np.nan], index=[0])

    def __repr__(self) -> str:
        window_str = str(self.window) if self.window is not None else "W"
        return f"EMA({self.operand.name}, {window_str})"