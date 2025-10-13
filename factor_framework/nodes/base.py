"""
因子节点抽象基类

这是设计文档中"抽象因子定义层"的核心组件。
FactorNode定义了因子计算的抽象接口，所有具体的因子节点都继承自这个基类。

关键设计原则：
1. 抽象性：基类完全不关心频率，只定义计算逻辑
2. 组合性：节点可以组合成复杂的因子表达式树
3. 延迟计算：节点只定义计算逻辑，实际计算由引擎调度
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Set
import pandas as pd
from ..config import FrequencyConfig


class FactorNode(ABC):
    """
    因子节点抽象基类

    所有因子节点的基类，定义了因子计算的统一接口。
    按照设计文档，这个层次完全不关心频率，只关心抽象的计算逻辑。

    核心设计：
    - 每个节点代表因子表达式树中的一个节点
    - 节点可以有输入（dependencies）和输出（计算结果）
    - 支持运算符重载，使得因子表达式可以用数学符号写出
    """

    def __init__(self, name: Optional[str] = None):
        """
        初始化因子节点

        Args:
            name: 节点名称，用于调试和可视化
        """
        self.name = name or self.__class__.__name__
        self.dependencies: List[FactorNode] = []  # 依赖的节点
        self._cache: Dict[str, Any] = {}  # 计算结果缓存
        self._cache_key: Optional[str] = None  # 当前缓存键

    @abstractmethod
    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        计算因子值的抽象方法

        这是所有因子节点必须实现的核心方法。按照设计文档，这个方法：
        1. 接收抽象的数据、配置和时间戳
        2. 返回因子值的Series
        3. 不关心具体的频率，由config参数提供频率信息

        Args:
            data: 输入数据DataFrame，包含OHLCV等基础数据
            config: 频率配置，提供时间维度的具体含义
            timestamp: 当前计算时间点
            **kwargs: 其他参数

        Returns:
            因子值Series，index为资产代码，values为因子值
        """
        pass

    def add_dependency(self, node: 'FactorNode'):
        """添加依赖节点"""
        if node not in self.dependencies:
            self.dependencies.append(node)

    def get_dependencies(self) -> List['FactorNode']:
        """获取所有依赖节点"""
        return self.dependencies.copy()

    def get_all_dependencies(self) -> Set['FactorNode']:
        """递归获取所有依赖节点（包括间接依赖）"""
        all_deps = set()
        for dep in self.dependencies:
            all_deps.add(dep)
            all_deps.update(dep.get_all_dependencies())
        return all_deps

    def _generate_cache_key(self, config: FrequencyConfig, timestamp: pd.Timestamp) -> str:
        """生成缓存键"""
        deps_key = "_".join(sorted([dep.name for dep in self.dependencies]))
        return f"{self.name}_{config.input_freq}_{config.window_length}_{timestamp}_{deps_key}"

    def compute_with_cache(self,
                          data: pd.DataFrame,
                          config: FrequencyConfig,
                          timestamp: pd.Timestamp,
                          **kwargs) -> pd.Series:
        """
        带缓存的计算方法

        实现了计算结果缓存，避免重复计算相同的因子值。
        这是性能优化的重要组件。
        """
        cache_key = self._generate_cache_key(config, timestamp)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self.compute(data, config, timestamp, **kwargs)
        self._cache[cache_key] = result
        self._cache_key = cache_key

        return result

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_key = None

    def validate_inputs(self, data: pd.DataFrame, config: FrequencyConfig) -> bool:
        """
        验证输入数据的有效性

        检查数据是否包含必要的列，时间范围是否足够等。
        """
        # 检查数据是否为空
        if data is None or data.empty:
            raise ValueError(f"节点 {self.name} 接收到空数据")

        # 检查是否有datetime索引或列
        if not isinstance(data.index, pd.DatetimeIndex) and 'datetime' not in data.columns:
            raise ValueError(f"节点 {self.name} 需要datetime索引或datetime列")

        return True

    # 运算符重载，支持因子表达式的数学运算
    def __add__(self, other: Union['FactorNode', float, int]) -> 'FactorNode':
        """加法运算符重载"""
        from .math_ops import Add
        if isinstance(other, (int, float)):
            from .atomic import Constant
            other = Constant(other)
        return Add(self, other)

    def __radd__(self, other: Union[float, int]) -> 'FactorNode':
        """右加法运算符重载"""
        return self.__add__(other)

    def __sub__(self, other: Union['FactorNode', float, int]) -> 'FactorNode':
        """减法运算符重载"""
        from .math_ops import Sub
        if isinstance(other, (int, float)):
            from .atomic import Constant
            other = Constant(other)
        return Sub(self, other)

    def __rsub__(self, other: Union[float, int]) -> 'FactorNode':
        """右减法运算符重载"""
        from .math_ops import Sub
        from .atomic import Constant
        return Sub(Constant(other), self)

    def __mul__(self, other: Union['FactorNode', float, int]) -> 'FactorNode':
        """乘法运算符重载"""
        from .math_ops import Mul
        if isinstance(other, (int, float)):
            from .atomic import Constant
            other = Constant(other)
        return Mul(self, other)

    def __rmul__(self, other: Union[float, int]) -> 'FactorNode':
        """右乘法运算符重载"""
        return self.__mul__(other)

    def __truediv__(self, other: Union['FactorNode', float, int]) -> 'FactorNode':
        """除法运算符重载"""
        from .math_ops import Div
        if isinstance(other, (int, float)):
            from .atomic import Constant
            other = Constant(other)
        return Div(self, other)

    def __rtruediv__(self, other: Union[float, int]) -> 'FactorNode':
        """右除法运算符重载"""
        from .math_ops import Div
        from .atomic import Constant
        return Div(Constant(other), self)

    def __neg__(self) -> 'FactorNode':
        """负号运算符重载"""
        from .math_ops import Mul
        from .atomic import Constant
        return Mul(Constant(-1), self)

    def __abs__(self) -> 'FactorNode':
        """绝对值运算符重载"""
        from .math_ops import Abs
        return Abs(self)

    def __pow__(self, other: Union['FactorNode', float, int]) -> 'FactorNode':
        """幂运算符重载"""
        from .math_ops import Pow
        if isinstance(other, (int, float)):
            from .atomic import Constant
            other = Constant(other)
        return Pow(self, other)

    def rank(self) -> 'FactorNode':
        """横截面排名"""
        from .cross_ops import Rank
        return Rank(self)

    def zscore(self) -> 'FactorNode':
        """横截面标准化"""
        from .cross_ops import Zscore
        return Zscore(self)

    def rolling_mean(self, window: int) -> 'FactorNode':
        """滚动平均"""
        from .time_ops import Mean
        return Mean(self, window)

    def rolling_std(self, window: int) -> 'FactorNode':
        """滚动标准差"""
        from .time_ops import Std
        return Std(self, window)

    def ref(self, periods: int) -> 'FactorNode':
        """历史值引用"""
        from .time_ops import Ref
        return Ref(self, periods)

    def __repr__(self) -> str:
        """字符串表示"""
        deps_str = ", ".join([dep.name for dep in self.dependencies])
        if deps_str:
            return f"{self.name}({deps_str})"
        return self.name

    def __str__(self) -> str:
        """用户友好的字符串表示"""
        return self.__repr__()

    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典表示，用于序列化"""
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'dependencies': [dep.to_dict() for dep in self.dependencies]
        }

    def visualize_tree(self, indent: int = 0) -> str:
        """可视化因子表达式树"""
        result = "  " * indent + f"{self.name}\n"
        for dep in self.dependencies:
            result += dep.visualize_tree(indent + 1)
        return result