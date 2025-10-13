"""
数据准备工具函数

提供因子分析所需的各种数据准备和转换功能。

主要功能：
- compute_forward_returns: 计算前向收益率
- align_factor_and_returns: 对齐因子值和收益率
- validate_multiindex: 验证 MultiIndex 格式
- resample_to_frequency: 重采样到指定频率
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import warnings


def compute_forward_returns(returns: pd.DataFrame,
                            periods: int = 1) -> pd.DataFrame:
    """
    计算前向收益率

    对每只资产，shift负值获取未来N期的收益率。
    严格防止未来函数：在时刻t，计算的是t+periods的收益率。

    Args:
        returns: 收益率 DataFrame (MultiIndex: datetime, asset)
                列名: 'return'
        periods: 前向期数

    Returns:
        前向收益率 DataFrame (相同格式)

    Example:
        >>> # 假设 returns 包含每日收益率
        >>> forward_returns = compute_forward_returns(returns, periods=1)
        >>> # forward_returns[t, asset] = returns[t+1, asset]
    """
    if not isinstance(returns.index, pd.MultiIndex):
        raise ValueError("returns 必须是 MultiIndex DataFrame")

    # 按资产分组，对每组进行 shift
    forward_returns = (returns
                      .groupby(level='asset')
                      .shift(-periods))

    return forward_returns


def align_factor_and_returns(factor_values: pd.DataFrame,
                            returns: pd.DataFrame,
                            forward_periods: int = 1,
                            dropna: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐因子值和前向收益率

    确保因子值和收益率的索引一致，便于后续分析。

    Args:
        factor_values: 因子值 DataFrame (MultiIndex: datetime, asset)
        returns: 收益率 DataFrame (MultiIndex: datetime, asset)
        forward_periods: 前向期数
        dropna: 是否删除缺失值

    Returns:
        (aligned_factor_values, aligned_forward_returns): 对齐后的数据

    Example:
        >>> factor_aligned, returns_aligned = align_factor_and_returns(
        ...     factor_values, returns, forward_periods=1
        ... )
    """
    # 计算前向收益率
    forward_returns = compute_forward_returns(returns, forward_periods)

    # 对齐索引
    factor_values, forward_returns = factor_values.align(
        forward_returns,
        join='inner',
        axis=0
    )

    # 删除缺失值
    if dropna:
        # 合并后删除任一为NaN的行
        valid_mask = (~factor_values.isnull().any(axis=1) &
                     ~forward_returns.isnull().any(axis=1))
        factor_values = factor_values[valid_mask]
        forward_returns = forward_returns[valid_mask]

    return factor_values, forward_returns


def validate_multiindex(df: pd.DataFrame,
                       expected_levels: Optional[List[str]] = None) -> bool:
    """
    验证 DataFrame 是否为合法的 MultiIndex 格式

    Args:
        df: 要验证的 DataFrame
        expected_levels: 期望的索引层级名称列表

    Returns:
        bool: 是否为合法格式

    Raises:
        ValueError: 如果格式不合法
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame 必须是 MultiIndex 格式")

    if expected_levels is not None:
        if df.index.names != expected_levels:
            raise ValueError(
                f"索引名称不匹配。期望: {expected_levels}, "
                f"实际: {df.index.names}"
            )

    return True


def resample_to_frequency(df: pd.DataFrame,
                         target_freq: str,
                         method: str = 'last') -> pd.DataFrame:
    """
    重采样到目标频率

    Args:
        df: MultiIndex DataFrame (datetime, asset)
        target_freq: 目标频率 ('D', 'W', 'M' 等)
        method: 聚合方法
                'last': 取最后一个值
                'first': 取第一个值
                'mean': 取平均值
                'sum': 求和

    Returns:
        重采样后的 DataFrame

    Example:
        >>> # 将分钟数据重采样为日频
        >>> daily_df = resample_to_frequency(minute_df, 'D', method='last')
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df 必须是 MultiIndex DataFrame")

    # 按资产分组，对每组进行重采样
    result = []
    for asset, group in df.groupby(level='asset'):
        # 重置索引，只保留时间
        group_ts = group.droplevel('asset')

        # 重采样
        if method == 'last':
            resampled = group_ts.resample(target_freq).last()
        elif method == 'first':
            resampled = group_ts.resample(target_freq).first()
        elif method == 'mean':
            resampled = group_ts.resample(target_freq).mean()
        elif method == 'sum':
            resampled = group_ts.resample(target_freq).sum()
        else:
            raise ValueError(f"不支持的聚合方法: {method}")

        # 添加资产列
        resampled['asset'] = asset
        result.append(resampled.reset_index())

    # 合并并重建 MultiIndex
    result_df = pd.concat(result, ignore_index=True)
    result_df = result_df.set_index(['datetime', 'asset'])

    return result_df


def split_train_test(df: pd.DataFrame,
                    train_ratio: float = 0.7,
                    by_time: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    划分训练集和测试集

    Args:
        df: MultiIndex DataFrame (datetime, asset)
        train_ratio: 训练集比例
        by_time: 是否按时间划分（True: 前70%时间为训练，False: 随机划分）

    Returns:
        (train_df, test_df): 训练集和测试集

    Example:
        >>> train, test = split_train_test(factor_data, train_ratio=0.7)
    """
    if by_time:
        # 按时间顺序划分
        dates = sorted(df.index.get_level_values('timestamp').unique())
        split_idx = int(len(dates) * train_ratio)
        train_dates = dates[:split_idx]

        train_df = df[df.index.get_level_values('timestamp').isin(train_dates)]
        test_df = df[~df.index.get_level_values('timestamp').isin(train_dates)]
    else:
        # 随机划分
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, train_size=train_ratio, random_state=42
        )

    return train_df, test_df


def winsorize_data(df: pd.DataFrame,
                  lower: float = 0.01,
                  upper: float = 0.99,
                  by_date: bool = True) -> pd.DataFrame:
    """
    数据缩尾处理

    限制极端值，提高数据稳健性。

    Args:
        df: DataFrame
        lower: 下分位数
        upper: 上分位数
        by_date: 是否按日期进行横截面缩尾

    Returns:
        缩尾后的 DataFrame

    Example:
        >>> # 对因子值进行缩尾（1%和99%分位数）
        >>> winsorized = winsorize_data(factor_values, lower=0.01, upper=0.99)
    """
    result = df.copy()

    if by_date and isinstance(df.index, pd.MultiIndex):
        # 按日期横截面缩尾
        dates = df.index.get_level_values('timestamp').unique()

        for date in dates:
            cross_section = df.xs(date, level='timestamp')

            for col in cross_section.columns:
                lower_bound = cross_section[col].quantile(lower)
                upper_bound = cross_section[col].quantile(upper)

                # 缩尾
                result.loc[(date, slice(None)), col] = (
                    cross_section[col].clip(lower=lower_bound, upper=upper_bound)
                )
    else:
        # 全局缩尾
        for col in df.columns:
            lower_bound = df[col].quantile(lower)
            upper_bound = df[col].quantile(upper)
            result[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return result


def fill_missing_values(df: pd.DataFrame,
                       method: str = 'forward',
                       limit: Optional[int] = None) -> pd.DataFrame:
    """
    填充缺失值

    Args:
        df: MultiIndex DataFrame
        method: 填充方法
                'forward': 前向填充
                'backward': 后向填充
                'mean': 均值填充
                'zero': 填充0
        limit: 最大填充数量

    Returns:
        填充后的 DataFrame

    Example:
        >>> filled = fill_missing_values(factor_values, method='forward', limit=5)
    """
    result = df.copy()

    if isinstance(df.index, pd.MultiIndex):
        # 按资产分组填充
        filled_groups = []
        for asset, group in result.groupby(level='asset'):
            if method == 'forward':
                filled = group.fillna(method='ffill', limit=limit)
            elif method == 'backward':
                filled = group.fillna(method='bfill', limit=limit)
            elif method == 'mean':
                filled = group.fillna(group.mean())
            elif method == 'zero':
                filled = group.fillna(0)
            else:
                raise ValueError(f"不支持的填充方法: {method}")

            filled_groups.append(filled)

        result = pd.concat(filled_groups)
    else:
        # 普通 DataFrame
        if method == 'forward':
            result = result.fillna(method='ffill', limit=limit)
        elif method == 'backward':
            result = result.fillna(method='bfill', limit=limit)
        elif method == 'mean':
            result = result.fillna(result.mean())
        elif method == 'zero':
            result = result.fillna(0)

    return result


def standardize_by_date(df: pd.DataFrame,
                       method: str = 'zscore') -> pd.DataFrame:
    """
    按日期进行横截面标准化

    Args:
        df: MultiIndex DataFrame (datetime, asset)
        method: 标准化方法
                'zscore': Z-score标准化
                'minmax': Min-Max标准化到[0,1]
                'rank': 排名标准化到[0,1]

    Returns:
        标准化后的 DataFrame

    Example:
        >>> standardized = standardize_by_date(factor_values, method='zscore')
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df 必须是 MultiIndex DataFrame")

    result = df.copy()
    dates = df.index.get_level_values('timestamp').unique()

    for date in dates:
        cross_section = df.xs(date, level='timestamp')

        for col in cross_section.columns:
            values = cross_section[col]

            if method == 'zscore':
                mean = values.mean()
                std = values.std()
                if std > 0:
                    standardized = (values - mean) / std
                else:
                    standardized = values
            elif method == 'minmax':
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    standardized = (values - min_val) / (max_val - min_val)
                else:
                    standardized = pd.Series(0.5, index=values.index)
            elif method == 'rank':
                standardized = values.rank() / len(values)
            else:
                raise ValueError(f"不支持的标准化方法: {method}")

            result.loc[(date, slice(None)), col] = standardized.values

    return result


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取数据摘要统计

    Args:
        df: MultiIndex DataFrame

    Returns:
        摘要统计 DataFrame

    Example:
        >>> summary = get_data_summary(factor_values)
        >>> print(summary)
    """
    summary = {
        'count': df.count(),
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        '25%': df.quantile(0.25),
        '50%': df.quantile(0.50),
        '75%': df.quantile(0.75),
        'max': df.max(),
        'missing_ratio': df.isnull().sum() / len(df)
    }

    return pd.DataFrame(summary)
