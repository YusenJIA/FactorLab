"""
工具函数模块

这个模块包含设计文档中提到的时间对齐、数据验证和其他辅助功能。
主要解决"时间轴对齐和处理"的技术挑战。

功能包括：
1. 时间对齐工具
2. 数据验证工具
3. 因子表达式工具
4. 性能优化工具
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from .config import FrequencyConfig
from .nodes.base import FactorNode


def align_timestamps(data: pd.DataFrame,
                    target_freq: str,
                    trading_calendar: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """
    时间戳对齐

    将数据对齐到指定的时间频率，确保时间轴的一致性

    Args:
        data: 输入数据
        target_freq: 目标频率 ('D', '5min', '30min' 等)
        trading_calendar: 交易日历（可选）

    Returns:
        对齐后的数据
    """
    if data.empty:
        return data

    # 确保有datetime列或索引
    if 'datetime' in data.columns:
        data = data.set_index('datetime')
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("数据必须包含datetime列或datetime索引")

    # 生成目标时间轴
    start_time = data.index.min()
    end_time = data.index.max()

    if trading_calendar is not None:
        # 使用提供的交易日历
        target_index = trading_calendar[(trading_calendar >= start_time) &
                                      (trading_calendar <= end_time)]
    else:
        # 生成标准时间轴
        target_index = generate_trading_calendar(start_time, end_time, target_freq)

    # 按资产分组对齐
    if 'code' in data.columns:
        aligned_data = []
        for code in data['code'].unique():
            asset_data = data[data['code'] == code]

            # 重新索引到目标时间轴
            aligned_asset = asset_data.reindex(target_index, method='ffill')
            aligned_asset['code'] = code

            aligned_data.append(aligned_asset)

        result = pd.concat(aligned_data)
    else:
        # 单资产情况
        result = data.reindex(target_index, method='ffill')

    return result.reset_index()


def generate_trading_calendar(start_date: pd.Timestamp,
                            end_date: pd.Timestamp,
                            freq: str,
                            market: str = 'CN') -> pd.DatetimeIndex:
    """
    生成交易日历

    根据市场和频率生成交易时间

    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率
        market: 市场代码 ('CN' for China)

    Returns:
        交易时间索引
    """
    if market == 'CN':
        # 中国A股交易时间
        return generate_cn_trading_calendar(start_date, end_date, freq)
    else:
        # 默认：工作日
        return pd.date_range(start_date, end_date, freq=freq)


def generate_cn_trading_calendar(start_date: pd.Timestamp,
                               end_date: pd.Timestamp,
                               freq: str) -> pd.DatetimeIndex:
    """
    生成中国A股交易日历

    考虑交易时间段：
    - 上午：9:30-11:30
    - 下午：13:00-15:00
    """
    trading_times = []

    if freq in ['D', '1D', 'daily']:
        # 日频：每个交易日15:00
        current = start_date.normalize()
        while current <= end_date:
            if current.weekday() < 5:  # 工作日
                trading_times.append(current.replace(hour=15, minute=0))
            current += timedelta(days=1)

    elif freq in ['30min', '30T']:
        # 30分钟频率
        current = start_date.normalize()
        while current <= end_date:
            if current.weekday() < 5:  # 工作日
                # 上午时段
                morning_times = pd.date_range(
                    current.replace(hour=9, minute=30),
                    current.replace(hour=11, minute=30),
                    freq='30min'
                )
                # 下午时段
                afternoon_times = pd.date_range(
                    current.replace(hour=13, minute=0),
                    current.replace(hour=15, minute=0),
                    freq='30min'
                )
                trading_times.extend(morning_times)
                trading_times.extend(afternoon_times)
            current += timedelta(days=1)

    elif freq in ['5min', '5T']:
        # 5分钟频率
        current = start_date.normalize()
        while current <= end_date:
            if current.weekday() < 5:  # 工作日
                # 上午时段
                morning_times = pd.date_range(
                    current.replace(hour=9, minute=30),
                    current.replace(hour=11, minute=30),
                    freq='5min'
                )
                # 下午时段
                afternoon_times = pd.date_range(
                    current.replace(hour=13, minute=0),
                    current.replace(hour=15, minute=0),
                    freq='5min'
                )
                trading_times.extend(morning_times)
                trading_times.extend(afternoon_times)
            current += timedelta(days=1)

    elif freq in ['1min', '1T', 'minute']:
        # 分钟频率
        current = start_date.normalize()
        while current <= end_date:
            if current.weekday() < 5:  # 工作日
                # 上午时段
                morning_times = pd.date_range(
                    current.replace(hour=9, minute=30),
                    current.replace(hour=11, minute=30),
                    freq='1min'
                )
                # 下午时段
                afternoon_times = pd.date_range(
                    current.replace(hour=13, minute=0),
                    current.replace(hour=15, minute=0),
                    freq='1min'
                )
                trading_times.extend(morning_times)
                trading_times.extend(afternoon_times)
            current += timedelta(days=1)

    else:
        # 其他频率：使用pandas默认
        return pd.date_range(start_date, end_date, freq=freq)

    return pd.DatetimeIndex(trading_times).sort_values()


def validate_factor_data(data: pd.DataFrame,
                        required_columns: Optional[List[str]] = None,
                        check_duplicates: bool = True,
                        check_missing: bool = True) -> Dict[str, Any]:
    """
    验证因子数据的完整性和正确性

    Args:
        data: 要验证的数据
        required_columns: 必需的列名列表
        check_duplicates: 是否检查重复数据
        check_missing: 是否检查缺失值

    Returns:
        验证报告字典
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    # 基本检查
    if data.empty:
        report['valid'] = False
        report['errors'].append("数据为空")
        return report

    # 检查必需列
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            report['valid'] = False
            report['errors'].append(f"缺少必需列: {missing_columns}")

    # 检查datetime列
    if 'datetime' in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
            try:
                data['datetime'] = pd.to_datetime(data['datetime'])
            except:
                report['valid'] = False
                report['errors'].append("datetime列无法转换为时间类型")
    elif not isinstance(data.index, pd.DatetimeIndex):
        report['warnings'].append("数据没有datetime列或datetime索引")

    # 检查重复数据
    if check_duplicates and 'datetime' in data.columns and 'code' in data.columns:
        duplicates = data.duplicated(subset=['datetime', 'code'])
        if duplicates.any():
            dup_count = duplicates.sum()
            report['warnings'].append(f"发现 {dup_count} 条重复数据")

    # 检查缺失值
    if check_missing:
        missing_stats = data.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]
        if not missing_cols.empty:
            report['warnings'].append(f"缺失值统计: {missing_cols.to_dict()}")

    # 统计信息
    report['statistics'] = {
        'total_records': len(data),
        'unique_assets': data['code'].nunique() if 'code' in data.columns else 1,
        'date_range': {
            'start': data['datetime'].min() if 'datetime' in data.columns else data.index.min(),
            'end': data['datetime'].max() if 'datetime' in data.columns else data.index.max()
        },
        'columns': list(data.columns)
    }

    return report


def detect_future_bias(factor_node: FactorNode,
                      data: pd.DataFrame,
                      test_timestamps: List[pd.Timestamp],
                      config: FrequencyConfig) -> Dict[str, Any]:
    """
    检测因子计算中的未来偏差

    通过改变计算时间点来检测是否使用了未来数据

    Args:
        factor_node: 要检测的因子节点
        data: 测试数据
        test_timestamps: 测试时间点列表
        config: 频率配置

    Returns:
        检测报告
    """
    report = {
        'has_future_bias': False,
        'test_results': [],
        'problematic_timestamps': []
    }

    for i, timestamp in enumerate(test_timestamps[:-1]):  # 排除最后一个时间点
        try:
            # 在当前时间点计算因子值
            current_value = factor_node.compute(data, config, timestamp)

            # 在未来时间点重新计算相同历史时间点的因子值
            future_timestamp = test_timestamps[i + 1]

            # 截取到当前时间点的数据
            if 'datetime' in data.columns:
                historical_data = data[data['datetime'] <= timestamp]
            else:
                historical_data = data[data.index <= timestamp]

            # 用未来时间点的计算环境重新计算
            recalc_value = factor_node.compute(historical_data, config, timestamp)

            # 比较结果
            if not current_value.equals(recalc_value):
                report['has_future_bias'] = True
                report['problematic_timestamps'].append(timestamp)

                test_result = {
                    'timestamp': timestamp,
                    'original_value': current_value,
                    'recalculated_value': recalc_value,
                    'difference': (current_value - recalc_value).abs().max()
                }
                report['test_results'].append(test_result)

        except Exception as e:
            continue

    return report


def optimize_factor_computation(factor_node: FactorNode,
                              sample_data: pd.DataFrame,
                              config: FrequencyConfig) -> Dict[str, Any]:
    """
    优化因子计算性能

    分析计算瓶颈并提供优化建议

    Args:
        factor_node: 要优化的因子节点
        sample_data: 样本数据
        config: 频率配置

    Returns:
        优化报告和建议
    """
    import time

    report = {
        'total_nodes': 0,
        'computation_times': {},
        'memory_usage': {},
        'bottlenecks': [],
        'recommendations': []
    }

    # 分析因子图结构
    all_nodes = factor_node.get_all_dependencies()
    all_nodes.add(factor_node)
    report['total_nodes'] = len(all_nodes)

    # 测试各节点的计算时间
    test_timestamp = sample_data['datetime'].max() if 'datetime' in sample_data.columns else sample_data.index.max()

    for node in all_nodes:
        start_time = time.time()
        try:
            result = node.compute(sample_data, config, test_timestamp)
            end_time = time.time()

            computation_time = end_time - start_time
            report['computation_times'][node.name] = computation_time

            # 估算内存使用
            if hasattr(result, 'memory_usage'):
                memory_usage = result.memory_usage(deep=True).sum()
                report['memory_usage'][node.name] = memory_usage

        except Exception as e:
            report['computation_times'][node.name] = float('inf')
            continue

    # 识别瓶颈
    if report['computation_times']:
        max_time = max(report['computation_times'].values())
        avg_time = np.mean(list(report['computation_times'].values()))

        for node_name, comp_time in report['computation_times'].items():
            if comp_time > avg_time * 2:  # 超过平均时间2倍
                report['bottlenecks'].append({
                    'node': node_name,
                    'time': comp_time,
                    'relative_slowness': comp_time / avg_time
                })

    # 生成优化建议
    if len(all_nodes) > 10:
        report['recommendations'].append("考虑拆分复杂的因子表达式以提高可读性和调试性")

    if report['bottlenecks']:
        report['recommendations'].append("优化瓶颈节点的计算逻辑，考虑使用向量化操作")

    if config.window_length > 100:
        report['recommendations'].append("对于大窗口计算，考虑使用增量计算或滚动窗口优化")

    return report


def create_factor_summary(factor_node: FactorNode) -> Dict[str, Any]:
    """
    创建因子摘要信息

    生成因子的结构化描述和依赖关系

    Args:
        factor_node: 因子节点

    Returns:
        因子摘要信息
    """
    summary = {
        'name': factor_node.name,
        'type': factor_node.__class__.__name__,
        'dependencies': [],
        'depth': 0,
        'total_nodes': 0,
        'expression_tree': factor_node.visualize_tree(),
        'serialized': factor_node.to_dict()
    }

    # 分析依赖关系
    all_deps = factor_node.get_all_dependencies()
    summary['total_nodes'] = len(all_deps) + 1

    for dep in factor_node.dependencies:
        dep_summary = {
            'name': dep.name,
            'type': dep.__class__.__name__,
            'is_atomic': len(dep.dependencies) == 0
        }
        summary['dependencies'].append(dep_summary)

    # 计算深度
    def calculate_depth(node):
        if not node.dependencies:
            return 0
        return 1 + max(calculate_depth(dep) for dep in node.dependencies)

    summary['depth'] = calculate_depth(factor_node)

    return summary


def compare_factors(factor1: FactorNode,
                   factor2: FactorNode,
                   data: pd.DataFrame,
                   config: FrequencyConfig,
                   test_timestamps: List[pd.Timestamp]) -> Dict[str, Any]:
    """
    比较两个因子的计算结果

    Args:
        factor1: 第一个因子
        factor2: 第二个因子
        data: 测试数据
        config: 频率配置
        test_timestamps: 测试时间点

    Returns:
        比较报告
    """
    comparison = {
        'correlation': {},
        'differences': {},
        'statistics': {},
        'identical_points': 0,
        'total_points': 0
    }

    factor1_values = []
    factor2_values = []

    for timestamp in test_timestamps:
        try:
            val1 = factor1.compute(data, config, timestamp)
            val2 = factor2.compute(data, config, timestamp)

            # 确保数据对齐
            val1, val2 = val1.align(val2, fill_value=np.nan)

            factor1_values.append(val1)
            factor2_values.append(val2)

            # 检查是否完全相同
            if val1.equals(val2):
                comparison['identical_points'] += 1

            comparison['total_points'] += 1

        except Exception as e:
            continue

    if factor1_values and factor2_values:
        # 合并所有时间点的数据
        all_val1 = pd.concat(factor1_values)
        all_val2 = pd.concat(factor2_values)

        # 计算相关性
        valid_mask = ~(all_val1.isna() | all_val2.isna())
        if valid_mask.any():
            correlation = all_val1[valid_mask].corr(all_val2[valid_mask])
            comparison['correlation']['pearson'] = correlation

            # 计算统计差异
            differences = all_val1[valid_mask] - all_val2[valid_mask]
            comparison['differences'] = {
                'mean_diff': differences.mean(),
                'std_diff': differences.std(),
                'max_abs_diff': differences.abs().max(),
                'rmse': np.sqrt((differences ** 2).mean())
            }

            # 描述性统计
            comparison['statistics'] = {
                'factor1': {
                    'mean': all_val1[valid_mask].mean(),
                    'std': all_val1[valid_mask].std(),
                    'min': all_val1[valid_mask].min(),
                    'max': all_val1[valid_mask].max()
                },
                'factor2': {
                    'mean': all_val2[valid_mask].mean(),
                    'std': all_val2[valid_mask].std(),
                    'min': all_val2[valid_mask].min(),
                    'max': all_val2[valid_mask].max()
                }
            }

    return comparison


def resample_factor_data(data: pd.DataFrame,
                        from_freq: str,
                        to_freq: str,
                        method: str = 'last') -> pd.DataFrame:
    """
    重采样因子数据到不同频率

    Args:
        data: 原始数据
        from_freq: 原始频率
        to_freq: 目标频率
        method: 重采样方法 ('last', 'mean', 'sum' 等)

    Returns:
        重采样后的数据
    """
    if data.empty:
        return data

    # 确保有datetime索引
    if 'datetime' in data.columns:
        data = data.set_index('datetime')
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("数据必须包含datetime索引")

    # 按资产分组重采样
    if 'code' in data.columns:
        resampled_data = []
        for code in data['code'].unique():
            asset_data = data[data['code'] == code]

            if method == 'last':
                resampled = asset_data.resample(to_freq).last()
            elif method == 'mean':
                resampled = asset_data.resample(to_freq).mean()
            elif method == 'sum':
                resampled = asset_data.resample(to_freq).sum()
            elif method == 'ohlc':
                # 特殊处理OHLC数据
                resampled = asset_data.resample(to_freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'amount': 'sum'
                })
            else:
                resampled = asset_data.resample(to_freq).last()

            resampled['code'] = code
            resampled_data.append(resampled)

        result = pd.concat(resampled_data)
    else:
        # 单资产重采样
        if method == 'last':
            result = data.resample(to_freq).last()
        elif method == 'mean':
            result = data.resample(to_freq).mean()
        elif method == 'sum':
            result = data.resample(to_freq).sum()
        else:
            result = data.resample(to_freq).last()

    return result.dropna().reset_index()