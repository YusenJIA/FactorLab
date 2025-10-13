"""
横截面运算节点实现

这些是设计文档中"横截面运算"的实现，包括排名和评分运算。
横截面运算在某个时间点对所有资产的因子值进行比较和变换。

支持的横截面运算类型：
- Rank: 横截面排名
- Zscore: 横截面标准化
- Quantile: 分位数
- Winsorize: 缩尾处理
- Neutralize: 中性化处理
"""

from typing import Optional, Union, List
import pandas as pd
import numpy as np
from .base import FactorNode
from ..config import FrequencyConfig


class CrossSectionOp(FactorNode):
    """
    横截面运算抽象基类

    横截面运算在某个时间点对所有资产进行操作，不涉及时间维度。
    """

    def __init__(self, operand: FactorNode, name: Optional[str] = None):
        """
        初始化横截面运算节点

        Args:
            operand: 输入因子
            name: 节点名称
        """
        super().__init__(name or f"{self.__class__.__name__}({operand.name})")
        self.operand = operand

        # 添加依赖关系
        self.add_dependency(operand)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        计算横截面运算结果

        横截面运算的特点是在单个时间点对所有资产进行操作
        """
        # 计算输入因子在当前时间点的值
        operand_values = self.operand.compute_with_cache(data, config, timestamp, **kwargs)

        # 执行横截面运算
        return self._execute_cross_section_operation(operand_values)

    def _execute_cross_section_operation(self, values: pd.Series) -> pd.Series:
        """
        执行具体的横截面运算

        子类必须实现这个方法
        """
        raise NotImplementedError("子类必须实现_execute_cross_section_operation方法")


class Rank(CrossSectionOp):
    """
    横截面排名

    将所有资产的因子值进行排名，返回排名百分位（0-1之间）。
    这是设计文档中提到的核心横截面运算之一。
    """

    def __init__(self,
                 operand: FactorNode,
                 ascending: bool = True,
                 method: str = 'average',
                 name: Optional[str] = None):
        """
        初始化排名节点

        Args:
            operand: 要排名的因子
            ascending: 是否升序排名（True表示数值越大排名越高）
            method: 排名方法（'average', 'min', 'max', 'first', 'dense'）
            name: 节点名称
        """
        super().__init__(operand, name or f"Rank({operand.name})")
        self.ascending = ascending
        self.method = method

    def _execute_cross_section_operation(self, values: pd.Series) -> pd.Series:
        """计算横截面排名"""
        # 过滤掉NaN值
        valid_mask = ~values.isna()
        if not valid_mask.any():
            return pd.Series(np.nan, index=values.index)

        valid_values = values[valid_mask]

        # 计算排名
        ranks = valid_values.rank(method=self.method, ascending=self.ascending)

        # 转换为百分位（0-1之间）
        if len(ranks) > 1:
            rank_pct = (ranks - 1) / (len(ranks) - 1)
        else:
            rank_pct = pd.Series([0.5], index=ranks.index)

        # 将结果填充回原始索引
        result = pd.Series(np.nan, index=values.index)
        result.loc[valid_mask] = rank_pct

        return result

    def __repr__(self) -> str:
        return f"Rank({self.operand.name})"


class Zscore(CrossSectionOp):
    """
    横截面标准化（Z-score）

    将因子值标准化为均值0、标准差1的分布。
    这是设计文档中提到的另一个核心横截面运算。
    """

    def __init__(self,
                 operand: FactorNode,
                 robust: bool = False,
                 name: Optional[str] = None):
        """
        初始化标准化节点

        Args:
            operand: 要标准化的因子
            robust: 是否使用鲁棒标准化（中位数和MAD）
            name: 节点名称
        """
        super().__init__(operand, name or f"Zscore({operand.name})")
        self.robust = robust

    def _execute_cross_section_operation(self, values: pd.Series) -> pd.Series:
        """计算横截面标准化"""
        # 过滤掉NaN值
        valid_mask = ~values.isna()
        if not valid_mask.any():
            return pd.Series(np.nan, index=values.index)

        valid_values = values[valid_mask]

        if len(valid_values) <= 1:
            # 只有一个值时，标准化结果为0
            result = pd.Series(np.nan, index=values.index)
            result.loc[valid_mask] = 0.0
            return result

        if self.robust:
            # 鲁棒标准化：使用中位数和MAD
            median = valid_values.median()
            mad = np.median(np.abs(valid_values - median))
            if mad > 0:
                standardized = (valid_values - median) / mad
            else:
                standardized = pd.Series(0.0, index=valid_values.index)
        else:
            # 普通标准化：使用均值和标准差
            mean = valid_values.mean()
            std = valid_values.std()
            if std > 0:
                standardized = (valid_values - mean) / std
            else:
                standardized = pd.Series(0.0, index=valid_values.index)

        # 将结果填充回原始索引
        result = pd.Series(np.nan, index=values.index)
        result.loc[valid_mask] = standardized

        return result

    def __repr__(self) -> str:
        return f"Zscore({self.operand.name})"


class Quantile(CrossSectionOp):
    """
    横截面分位数

    将因子值分为指定数量的分位数组别
    """

    def __init__(self,
                 operand: FactorNode,
                 n_quantiles: int = 5,
                 labels: bool = False,
                 name: Optional[str] = None):
        """
        初始化分位数节点

        Args:
            operand: 要分组的因子
            n_quantiles: 分位数数量
            labels: 是否返回标签（False返回0-1之间的值）
            name: 节点名称
        """
        super().__init__(operand, name or f"Quantile({operand.name}, {n_quantiles})")
        self.n_quantiles = n_quantiles
        self.labels = labels

    def _execute_cross_section_operation(self, values: pd.Series) -> pd.Series:
        """计算分位数分组"""
        # 过滤掉NaN值
        valid_mask = ~values.isna()
        if not valid_mask.any():
            return pd.Series(np.nan, index=values.index)

        valid_values = values[valid_mask]

        # 计算分位数
        try:
            if self.labels:
                # 返回分组标签（1, 2, 3, ...）
                quantiles = pd.qcut(valid_values, self.n_quantiles, labels=False) + 1
            else:
                # 返回分位数值（0-1之间）
                quantiles = pd.qcut(valid_values, self.n_quantiles, labels=False) / (self.n_quantiles - 1)
        except ValueError:
            # 如果无法分组（值太少或重复值太多），返回均匀分布
            if self.labels:
                quantiles = pd.Series(1, index=valid_values.index)
            else:
                quantiles = pd.Series(0.5, index=valid_values.index)

        # 将结果填充回原始索引
        result = pd.Series(np.nan, index=values.index)
        result.loc[valid_mask] = quantiles

        return result

    def __repr__(self) -> str:
        return f"Quantile({self.operand.name}, {self.n_quantiles})"


class Winsorize(CrossSectionOp):
    """
    横截面缩尾处理

    将极端值限制在指定的百分位数范围内
    """

    def __init__(self,
                 operand: FactorNode,
                 lower: float = 0.05,
                 upper: float = 0.95,
                 name: Optional[str] = None):
        """
        初始化缩尾处理节点

        Args:
            operand: 要处理的因子
            lower: 下分位数
            upper: 上分位数
            name: 节点名称
        """
        super().__init__(operand, name or f"Winsorize({operand.name}, {lower}, {upper})")
        self.lower = lower
        self.upper = upper

    def _execute_cross_section_operation(self, values: pd.Series) -> pd.Series:
        """执行缩尾处理"""
        # 过滤掉NaN值
        valid_mask = ~values.isna()
        if not valid_mask.any():
            return values

        valid_values = values[valid_mask]

        # 计算分位数
        lower_bound = valid_values.quantile(self.lower)
        upper_bound = valid_values.quantile(self.upper)

        # 执行缩尾
        winsorized = valid_values.clip(lower=lower_bound, upper=upper_bound)

        # 将结果填充回原始索引
        result = values.copy()
        result.loc[valid_mask] = winsorized

        return result

    def __repr__(self) -> str:
        return f"Winsorize({self.operand.name}, {self.lower}, {self.upper})"


class Neutralize(CrossSectionOp):
    """
    横截面中性化处理

    根据指定的基准因子对目标因子进行中性化，去除基准因子的影响
    """

    def __init__(self,
                 operand: FactorNode,
                 benchmark: FactorNode,
                 name: Optional[str] = None):
        """
        初始化中性化节点

        Args:
            operand: 要中性化的因子
            benchmark: 基准因子
            name: 节点名称
        """
        super().__init__(operand, name or f"Neutralize({operand.name}, {benchmark.name})")
        self.benchmark = benchmark

        # 添加基准因子依赖
        self.add_dependency(benchmark)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        计算中性化结果

        重写compute方法，因为需要同时计算operand和benchmark
        """
        # 计算目标因子和基准因子的值
        operand_values = self.operand.compute_with_cache(data, config, timestamp, **kwargs)
        benchmark_values = self.benchmark.compute_with_cache(data, config, timestamp, **kwargs)

        # 对齐索引
        operand_values, benchmark_values = operand_values.align(benchmark_values, fill_value=np.nan)

        # 执行中性化
        return self._execute_neutralization(operand_values, benchmark_values)

    def _execute_neutralization(self, operand_values: pd.Series, benchmark_values: pd.Series) -> pd.Series:
        """执行中性化处理"""
        # 过滤掉NaN值
        valid_mask = ~(operand_values.isna() | benchmark_values.isna())
        if not valid_mask.any() or valid_mask.sum() < 2:
            return operand_values

        valid_operand = operand_values[valid_mask]
        valid_benchmark = benchmark_values[valid_mask]

        # 计算线性回归残差
        try:
            # 简化的线性回归：y = alpha + beta * x + residual
            X = np.column_stack([np.ones(len(valid_benchmark)), valid_benchmark])
            beta = np.linalg.lstsq(X, valid_operand, rcond=None)[0]

            # 计算残差
            predicted = beta[0] + beta[1] * valid_benchmark
            residuals = valid_operand - predicted

            # 将结果填充回原始索引
            result = pd.Series(np.nan, index=operand_values.index)
            result.loc[valid_mask] = residuals

            return result
        except:
            # 如果回归失败，返回原值
            return operand_values

    def __repr__(self) -> str:
        return f"Neutralize({self.operand.name}, {self.benchmark.name})"


class Scale(CrossSectionOp):
    """
    横截面缩放

    将因子值缩放到指定范围
    """

    def __init__(self,
                 operand: FactorNode,
                 target_sum: Optional[float] = None,
                 target_range: Optional[tuple] = None,
                 name: Optional[str] = None):
        """
        初始化缩放节点

        Args:
            operand: 要缩放的因子
            target_sum: 目标总和（如果指定，将缩放到该总和）
            target_range: 目标范围（如果指定，将缩放到该范围）
            name: 节点名称
        """
        super().__init__(operand, name or f"Scale({operand.name})")
        self.target_sum = target_sum
        self.target_range = target_range

    def _execute_cross_section_operation(self, values: pd.Series) -> pd.Series:
        """执行缩放处理"""
        # 过滤掉NaN值
        valid_mask = ~values.isna()
        if not valid_mask.any():
            return values

        valid_values = values[valid_mask]

        if self.target_sum is not None:
            # 缩放到指定总和
            current_sum = valid_values.sum()
            if current_sum != 0:
                scaled = valid_values * (self.target_sum / current_sum)
            else:
                scaled = valid_values
        elif self.target_range is not None:
            # 缩放到指定范围
            min_val, max_val = valid_values.min(), valid_values.max()
            if min_val != max_val:
                # 标准化到0-1
                normalized = (valid_values - min_val) / (max_val - min_val)
                # 缩放到目标范围
                target_min, target_max = self.target_range
                scaled = normalized * (target_max - target_min) + target_min
            else:
                # 所有值相同，设为中间值
                target_min, target_max = self.target_range
                scaled = pd.Series((target_min + target_max) / 2, index=valid_values.index)
        else:
            # 默认：标准化到总和为1
            current_sum = valid_values.sum()
            if current_sum != 0:
                scaled = valid_values / current_sum
            else:
                scaled = pd.Series(1.0 / len(valid_values), index=valid_values.index)

        # 将结果填充回原始索引
        result = values.copy()
        result.loc[valid_mask] = scaled

        return result

    def __repr__(self) -> str:
        if self.target_sum is not None:
            return f"Scale({self.operand.name}, sum={self.target_sum})"
        elif self.target_range is not None:
            return f"Scale({self.operand.name}, range={self.target_range})"
        else:
            return f"Scale({self.operand.name})"