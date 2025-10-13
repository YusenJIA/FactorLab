"""
数学运算节点实现

这些是设计文档中"基本运算"的实现，包括各种数学运算符。
数学运算节点接收一个或多个因子作为输入，对其进行数学变换。

支持的运算类型：
- 二元运算：Add(+), Sub(-), Mul(*), Div(/), Pow(**)
- 一元运算：Abs(绝对值), Log(对数), Sqrt(平方根), Neg(取负)
- 条件运算：Max, Min, Clip
"""

from typing import Union, Optional
import pandas as pd
import numpy as np
from .base import FactorNode
from ..config import FrequencyConfig


class BinaryOp(FactorNode):
    """
    二元运算抽象基类

    所有二元运算的基类，接收两个因子作为输入。
    """

    def __init__(self, left: FactorNode, right: FactorNode, name: Optional[str] = None):
        """
        初始化二元运算节点

        Args:
            left: 左操作数
            right: 右操作数
            name: 节点名称
        """
        super().__init__(name or f"{self.__class__.__name__}({left.name}, {right.name})")
        self.left = left
        self.right = right

        # 添加依赖关系
        self.add_dependency(left)
        self.add_dependency(right)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """
        计算二元运算结果

        子类需要实现具体的运算逻辑
        """
        # 计算左右操作数的值
        left_val = self.left.compute_with_cache(data, config, timestamp, **kwargs)
        right_val = self.right.compute_with_cache(data, config, timestamp, **kwargs)

        # 对齐索引（确保两个Series的索引一致）
        left_val, right_val = left_val.align(right_val, fill_value=np.nan)

        # 执行具体运算
        return self._execute_operation(left_val, right_val)

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """
        执行具体的运算操作

        子类必须实现这个方法
        """
        raise NotImplementedError("子类必须实现_execute_operation方法")


class UnaryOp(FactorNode):
    """
    一元运算抽象基类

    所有一元运算的基类，接收一个因子作为输入。
    """

    def __init__(self, operand: FactorNode, name: Optional[str] = None):
        """
        初始化一元运算节点

        Args:
            operand: 操作数
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
        计算一元运算结果
        """
        # 计算操作数的值
        operand_val = self.operand.compute_with_cache(data, config, timestamp, **kwargs)

        # 执行具体运算
        return self._execute_operation(operand_val)

    def _execute_operation(self, operand: pd.Series) -> pd.Series:
        """
        执行具体的运算操作

        子类必须实现这个方法
        """
        raise NotImplementedError("子类必须实现_execute_operation方法")


# 二元运算实现
class Add(BinaryOp):
    """加法运算：left + right"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left + right

    def __repr__(self) -> str:
        return f"({self.left.name} + {self.right.name})"


class Sub(BinaryOp):
    """减法运算：left - right"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left - right

    def __repr__(self) -> str:
        return f"({self.left.name} - {self.right.name})"


class Mul(BinaryOp):
    """乘法运算：left * right"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left * right

    def __repr__(self) -> str:
        return f"({self.left.name} * {self.right.name})"


class Div(BinaryOp):
    """
    除法运算：left / right

    按照设计文档示例，这是构建"价格与平均成交量比率"因子的关键运算。
    """

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        # 处理除零情况
        result = left / right
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    def __repr__(self) -> str:
        return f"({self.left.name} / {self.right.name})"


class Pow(BinaryOp):
    """幂运算：left ** right"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return np.power(left, right)

    def __repr__(self) -> str:
        return f"({self.left.name} ** {self.right.name})"


# 一元运算实现
class Abs(UnaryOp):
    """绝对值运算：abs(operand)"""

    def _execute_operation(self, operand: pd.Series) -> pd.Series:
        return np.abs(operand)

    def __repr__(self) -> str:
        return f"Abs({self.operand.name})"


class Log(UnaryOp):
    """自然对数运算：ln(operand)"""

    def _execute_operation(self, operand: pd.Series) -> pd.Series:
        # 处理非正数情况
        result = np.log(operand)
        return result.replace([np.inf, -np.inf], np.nan)

    def __repr__(self) -> str:
        return f"Log({self.operand.name})"


class Sqrt(UnaryOp):
    """平方根运算：sqrt(operand)"""

    def _execute_operation(self, operand: pd.Series) -> pd.Series:
        return np.sqrt(np.maximum(operand, 0))  # 负数时返回0

    def __repr__(self) -> str:
        return f"Sqrt({self.operand.name})"


class Neg(UnaryOp):
    """取负运算：-operand"""

    def _execute_operation(self, operand: pd.Series) -> pd.Series:
        return -operand

    def __repr__(self) -> str:
        return f"-{self.operand.name}"


class Sign(UnaryOp):
    """符号函数：sign(operand)"""

    def _execute_operation(self, operand: pd.Series) -> pd.Series:
        return np.sign(operand)

    def __repr__(self) -> str:
        return f"Sign({self.operand.name})"


# 条件运算
class Max(BinaryOp):
    """最大值运算：max(left, right)"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return np.maximum(left, right)

    def __repr__(self) -> str:
        return f"Max({self.left.name}, {self.right.name})"


class Min(BinaryOp):
    """最小值运算：min(left, right)"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return np.minimum(left, right)

    def __repr__(self) -> str:
        return f"Min({self.left.name}, {self.right.name})"


class Clip(FactorNode):
    """
    裁剪运算：将值限制在指定范围内

    clip(operand, lower, upper) = max(lower, min(operand, upper))
    """

    def __init__(self,
                 operand: FactorNode,
                 lower: Union[FactorNode, float, None] = None,
                 upper: Union[FactorNode, float, None] = None,
                 name: Optional[str] = None):
        """
        初始化裁剪运算节点

        Args:
            operand: 要裁剪的因子
            lower: 下界，可以是因子或常数
            upper: 上界，可以是因子或常数
            name: 节点名称
        """
        super().__init__(name or f"Clip({operand.name}, {lower}, {upper})")
        self.operand = operand
        self.lower = lower
        self.upper = upper

        # 添加依赖关系
        self.add_dependency(operand)
        if isinstance(lower, FactorNode):
            self.add_dependency(lower)
        if isinstance(upper, FactorNode):
            self.add_dependency(upper)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """计算裁剪结果"""
        # 计算操作数的值
        operand_val = self.operand.compute_with_cache(data, config, timestamp, **kwargs)

        # 计算下界
        if self.lower is None:
            lower_val = None
        elif isinstance(self.lower, FactorNode):
            lower_val = self.lower.compute_with_cache(data, config, timestamp, **kwargs)
        else:
            lower_val = self.lower

        # 计算上界
        if self.upper is None:
            upper_val = None
        elif isinstance(self.upper, FactorNode):
            upper_val = self.upper.compute_with_cache(data, config, timestamp, **kwargs)
        else:
            upper_val = self.upper

        # 执行裁剪
        result = operand_val.copy()
        if lower_val is not None:
            if isinstance(lower_val, pd.Series):
                result = np.maximum(result, lower_val)
            else:
                result = np.maximum(result, lower_val)

        if upper_val is not None:
            if isinstance(upper_val, pd.Series):
                result = np.minimum(result, upper_val)
            else:
                result = np.minimum(result, upper_val)

        return result


class IfElse(FactorNode):
    """
    条件运算：if condition then true_val else false_val

    这是一个三元运算符，根据条件选择不同的值。
    """

    def __init__(self,
                 condition: FactorNode,
                 true_val: FactorNode,
                 false_val: FactorNode,
                 name: Optional[str] = None):
        """
        初始化条件运算节点

        Args:
            condition: 条件因子（True/False）
            true_val: 条件为真时的值
            false_val: 条件为假时的值
            name: 节点名称
        """
        super().__init__(name or f"IfElse({condition.name}, {true_val.name}, {false_val.name})")
        self.condition = condition
        self.true_val = true_val
        self.false_val = false_val

        # 添加依赖关系
        self.add_dependency(condition)
        self.add_dependency(true_val)
        self.add_dependency(false_val)

    def compute(self,
                data: pd.DataFrame,
                config: FrequencyConfig,
                timestamp: pd.Timestamp,
                **kwargs) -> pd.Series:
        """计算条件运算结果"""
        # 计算各个因子的值
        cond_val = self.condition.compute_with_cache(data, config, timestamp, **kwargs)
        true_val = self.true_val.compute_with_cache(data, config, timestamp, **kwargs)
        false_val = self.false_val.compute_with_cache(data, config, timestamp, **kwargs)

        # 对齐索引
        cond_val, true_val = cond_val.align(true_val, fill_value=np.nan)
        cond_val, false_val = cond_val.align(false_val, fill_value=np.nan)

        # 执行条件选择
        result = np.where(cond_val, true_val, false_val)
        return pd.Series(result, index=cond_val.index)


# 比较运算
class Greater(BinaryOp):
    """大于运算：left > right"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left > right

    def __repr__(self) -> str:
        return f"({self.left.name} > {self.right.name})"


class Less(BinaryOp):
    """小于运算：left < right"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left < right

    def __repr__(self) -> str:
        return f"({self.left.name} < {self.right.name})"


class Equal(BinaryOp):
    """等于运算：left == right"""

    def _execute_operation(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left == right

    def __repr__(self) -> str:
        return f"({self.left.name} == {self.right.name})"