"""
因子分析核心数据结构

这个模块定义了因子分析系统的核心数据结构，提供标准化的数据容器。
完美集成 factor_framework 的输出格式。

核心类：
- FactorData: 因子数据容器（支持从 FactorEngine 输出自动转换）
- AnalysisResult: 分析结果容器
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime


@dataclass
class FactorData:
    """
    因子数据容器

    标准化的因子数据格式，用于所有分析模块。
    内部使用 MultiIndex 存储，便于横截面和时间序列操作。

    Attributes:
        factor_values: 因子值 DataFrame (MultiIndex: datetime, code)
        returns: 收益率 DataFrame (MultiIndex: datetime, code)
        factor_name: 因子名称
        start_date: 开始日期
        end_date: 结束日期
        frequency: 数据频率
        attributes: 可选的资产属性（行业、市值等）
    """

    factor_values: pd.DataFrame
    returns: pd.DataFrame
    factor_name: str
    start_date: str
    end_date: str
    frequency: str = 'daily'
    attributes: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """数据验证"""
        self.validate()

    @classmethod
    def from_engine_output(cls,
                          factor_df: pd.DataFrame,
                          returns_df: pd.DataFrame,
                          factor_name: str,
                          frequency: str = 'daily',
                          attributes: Optional[pd.DataFrame] = None) -> 'FactorData':
        """
        从 FactorEngine 输出创建 FactorData

        自动将 FactorEngine 的输出格式转换为 MultiIndex 格式。

        Args:
            factor_df: FactorEngine.compute_factor() 的输出
                      必需列: ['timestamp', 'asset', 'factor_value']
            returns_df: 收益率 DataFrame
                       必需列: ['timestamp', 'asset', 'return']
            factor_name: 因子名称
            frequency: 数据频率
            attributes: 可选的资产属性

        Returns:
            FactorData 实例

        Example:
            >>> from factor_framework import FactorEngine, FrequencyConfig
            >>> from factor_framework.nodes import Close, Volume, Mean
            >>>
            >>> # 使用 factor_framework 计算因子
            >>> factor = Close() / Mean(Volume(), window=20)
            >>> config = FrequencyConfig(input_freq='daily')
            >>> engine = FactorEngine()
            >>>
            >>> factor_df = engine.compute_factor(
            ...     factor, ['510050.SH'], '2024-01-01', '2024-12-31', config
            ... )
            >>>
            >>> # 转换为 FactorData
            >>> factor_data = FactorData.from_engine_output(
            ...     factor_df=factor_df,
            ...     returns_df=returns_df,
            ...     factor_name='价格成交量比'
            ... )
        """
        # 验证输入格式
        required_factor_cols = ['timestamp', 'asset', 'factor_value']
        required_returns_cols = ['timestamp', 'asset', 'return']

        if not all(col in factor_df.columns for col in required_factor_cols):
            raise ValueError(f"factor_df 必须包含列: {required_factor_cols}")

        if not all(col in returns_df.columns for col in required_returns_cols):
            raise ValueError(f"returns_df 必须包含列: {required_returns_cols}")

        # 转换为 MultiIndex 格式
        factor_pivot = (factor_df
                       .set_index(['timestamp', 'asset'])['factor_value']
                       .to_frame('factor_value'))

        returns_pivot = (returns_df
                        .set_index(['timestamp', 'asset'])['return']
                        .to_frame('return'))

        # 确定日期范围
        start_date = pd.to_datetime(factor_df['timestamp'].min()).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(factor_df['timestamp'].max()).strftime('%Y-%m-%d')

        return cls(
            factor_values=factor_pivot,
            returns=returns_pivot,
            factor_name=factor_name,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            attributes=attributes
        )

    @classmethod
    def from_multiindex(cls,
                       factor_values: pd.DataFrame,
                       returns: pd.DataFrame,
                       factor_name: str,
                       start_date: str,
                       end_date: str,
                       frequency: str = 'daily',
                       attributes: Optional[pd.DataFrame] = None) -> 'FactorData':
        """
        从 MultiIndex DataFrame 直接创建

        适用于已经是 MultiIndex 格式的数据。

        Args:
            factor_values: MultiIndex DataFrame (datetime, code)
            returns: MultiIndex DataFrame (datetime, code)
            factor_name: 因子名称
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            attributes: 可选属性

        Returns:
            FactorData 实例
        """
        return cls(
            factor_values=factor_values,
            returns=returns,
            factor_name=factor_name,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            attributes=attributes
        )

    def validate(self) -> bool:
        """
        验证数据完整性

        Returns:
            bool: 数据是否有效

        Raises:
            ValueError: 如果数据无效
        """
        # 检查索引类型
        if not isinstance(self.factor_values.index, pd.MultiIndex):
            raise ValueError("factor_values 必须是 MultiIndex DataFrame")

        if not isinstance(self.returns.index, pd.MultiIndex):
            raise ValueError("returns 必须是 MultiIndex DataFrame")

        # 检查索引名称
        if self.factor_values.index.names != ['timestamp', 'asset']:
            # 尝试重命名
            if len(self.factor_values.index.names) == 2:
                self.factor_values.index.names = ['timestamp', 'asset']
            else:
                raise ValueError("factor_values 索引必须是 (timestamp, asset)")

        if self.returns.index.names != ['timestamp', 'asset']:
            if len(self.returns.index.names) == 2:
                self.returns.index.names = ['timestamp', 'asset']
            else:
                raise ValueError("returns 索引必须是 (timestamp, asset)")

        # 检查数据是否为空
        if self.factor_values.empty:
            raise ValueError("factor_values 不能为空")

        if self.returns.empty:
            raise ValueError("returns 不能为空")

        # 统计缺失值
        factor_missing_ratio = self.factor_values.isnull().sum().sum() / len(self.factor_values)
        returns_missing_ratio = self.returns.isnull().sum().sum() / len(self.returns)

        if factor_missing_ratio > 0.5:
            import warnings
            warnings.warn(f"因子值缺失比例较高: {factor_missing_ratio:.2%}")

        if returns_missing_ratio > 0.5:
            import warnings
            warnings.warn(f"收益率缺失比例较高: {returns_missing_ratio:.2%}")

        return True

    def get_cross_section(self, date: str) -> pd.Series:
        """
        获取某日的横截面数据

        Args:
            date: 日期字符串 'YYYY-MM-DD' 或 Timestamp

        Returns:
            该日所有资产的因子值 Series (index: asset)
        """
        date_ts = pd.to_datetime(date)
        try:
            return self.factor_values.xs(date_ts, level='timestamp')['factor_value']
        except KeyError:
            raise ValueError(f"日期 {date} 不在数据范围内")

    def get_time_series(self, code: str) -> pd.Series:
        """
        获取某只资产的时间序列

        Args:
            code: 资产代码

        Returns:
            该资产的因子值时间序列 (index: timestamp)
        """
        try:
            return self.factor_values.xs(code, level='asset')['factor_value']
        except KeyError:
            raise ValueError(f"资产 {code} 不在数据中")

    def get_date_range(self) -> List[pd.Timestamp]:
        """获取所有日期列表"""
        return sorted(self.factor_values.index.get_level_values('timestamp').unique())

    def get_assets(self) -> List[str]:
        """获取所有资产代码列表"""
        return sorted(self.factor_values.index.get_level_values('asset').unique())

    def to_long_format(self) -> pd.DataFrame:
        """
        转换回 FactorEngine 输出格式

        将 MultiIndex 格式转换回 long format，便于保存或传递给其他系统。

        Returns:
            DataFrame with columns: ['timestamp', 'asset', 'factor_value']
        """
        return self.factor_values.reset_index()

    def summary(self) -> str:
        """
        生成数据摘要

        Returns:
            格式化的摘要字符串
        """
        n_dates = len(self.get_date_range())
        n_assets = len(self.get_assets())
        n_obs = len(self.factor_values)

        factor_stats = self.factor_values['factor_value'].describe()
        returns_stats = self.returns['return'].describe()

        summary = f"""
FactorData Summary
==================
Factor Name:  {self.factor_name}
Frequency:    {self.frequency}
Date Range:   {self.start_date} to {self.end_date}
Dates:        {n_dates}
Assets:       {n_assets}
Observations: {n_obs}

Factor Statistics:
------------------
Mean:     {factor_stats['mean']:.6f}
Std:      {factor_stats['std']:.6f}
Min:      {factor_stats['min']:.6f}
25%:      {factor_stats['25%']:.6f}
50%:      {factor_stats['50%']:.6f}
75%:      {factor_stats['75%']:.6f}
Max:      {factor_stats['max']:.6f}

Returns Statistics:
-------------------
Mean:     {returns_stats['mean']:.6f}
Std:      {returns_stats['std']:.6f}
Min:      {returns_stats['min']:.6f}
Max:      {returns_stats['max']:.6f}
"""
        return summary

    def __repr__(self) -> str:
        return f"FactorData('{self.factor_name}', {self.start_date} to {self.end_date}, {len(self.get_assets())} assets)"


@dataclass
class AnalysisResult:
    """
    分析结果容器

    统一的分析结果格式，用于所有分析模块的输出。

    Attributes:
        name: 结果名称（如 'IC Analysis - 动量因子'）
        metrics: 核心指标字典（如 {'ic_mean': 0.05, 'icir': 1.2}）
        data: 详细数据字典（如 {'ic_series': pd.Series}）
        figures: 图表对象字典（如 {'ic_timeseries': matplotlib.figure.Figure}）
        summary: 文字摘要
        timestamp: 分析生成时间
    """

    name: str
    metrics: Dict[str, float]
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_dict(self) -> Dict:
        """
        转换为字典（用于序列化）

        Returns:
            包含所有非图表数据的字典
        """
        return {
            'name': self.name,
            'metrics': self.metrics,
            'data': {k: v.to_dict() for k, v in self.data.items()},
            'summary': self.summary,
            'timestamp': self.timestamp
        }

    def save(self, path: str, format: str = 'json', save_figures: bool = True):
        """
        保存结果

        Args:
            path: 保存路径（不含扩展名）
            format: 格式 ('json' 或 'pickle')
            save_figures: 是否保存图表
        """
        if format == 'json':
            # 保存指标和摘要
            with open(f"{path}.json", 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        elif format == 'pickle':
            # 保存完整对象（包括图表）
            import pickle
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"不支持的格式: {format}")

        # 保存图表
        if save_figures and self.figures:
            import os
            figures_dir = f"{path}_figures"
            os.makedirs(figures_dir, exist_ok=True)

            for fig_name, fig in self.figures.items():
                if fig is not None:
                    fig_path = os.path.join(figures_dir, f"{fig_name}.png")
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')

    @classmethod
    def load(cls, path: str, format: str = 'pickle') -> 'AnalysisResult':
        """
        加载结果

        Args:
            path: 文件路径（不含扩展名）
            format: 格式 ('pickle')

        Returns:
            AnalysisResult 实例
        """
        if format == 'pickle':
            import pickle
            with open(f"{path}.pkl", 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"加载仅支持 pickle 格式")

    def display(self):
        """显示结果摘要"""
        print("=" * 60)
        print(self.name)
        print("=" * 60)

        if self.summary:
            print(self.summary)
        else:
            print("\nMetrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.4f}")
                else:
                    print(f"  {key:20s}: {value}")

        print(f"\nGenerated at: {self.timestamp}")

        if self.figures:
            print(f"\nFigures: {', '.join(self.figures.keys())}")

    def __repr__(self) -> str:
        return f"AnalysisResult('{self.name}', {len(self.metrics)} metrics, {len(self.figures)} figures)"
