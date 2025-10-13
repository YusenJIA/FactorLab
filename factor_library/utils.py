"""
因子库管理系统工具函数

提供序列化、反序列化等工具函数。
"""

import pickle
import json
import hashlib
from typing import Dict, Any, List
from datetime import datetime

from factor_framework.nodes.base import FactorNode
from factor_framework.config import FrequencyConfig

def serialize_factor_node(node: FactorNode) -> str:
    """序列化因子节点为十六进制字符串

    Args:
        node: 因子节点实例

    Returns:
        str: 序列化后的十六进制字符串
    """
    try:
        return pickle.dumps(node).hex()
    except Exception as e:
        raise ValueError(f"Failed to serialize factor node: {e}")

def deserialize_factor_node(data: str) -> FactorNode:
    """反序列化因子节点

    Args:
        data: 序列化的十六进制字符串

    Returns:
        FactorNode: 因子节点实例
    """
    try:
        return pickle.loads(bytes.fromhex(data))
    except Exception as e:
        raise ValueError(f"Failed to deserialize factor node: {e}")

def serialize_frequency_config(config: FrequencyConfig) -> Dict:
    """序列化频率配置为字典

    Args:
        config: 频率配置实例

    Returns:
        Dict: 配置字典
    """
    return {
        'input_freq': config.input_freq,
        'window_length': config.window_length,
        'calc_freq': config.calc_freq
    }

def deserialize_frequency_config(data: Dict) -> FrequencyConfig:
    """反序列化频率配置

    Args:
        data: 配置字典

    Returns:
        FrequencyConfig: 频率配置实例
    """
    return FrequencyConfig(**data)

def serialize_frequency_configs(configs: List[FrequencyConfig]) -> List[Dict]:
    """序列化频率配置列表"""
    return [serialize_frequency_config(config) for config in configs]

def deserialize_frequency_configs(data: List[Dict]) -> List[FrequencyConfig]:
    """反序列化频率配置列表"""
    return [deserialize_frequency_config(config_dict) for config_dict in data]

def generate_factor_id(factor_node: FactorNode, metadata: Dict) -> str:
    """生成因子唯一标识

    Args:
        factor_node: 因子节点
        metadata: 元数据字典

    Returns:
        str: 因子ID
    """
    # 使用因子名称和表达式哈希生成ID
    name = metadata.get('name', 'unnamed_factor')
    expression = str(factor_node)

    # 创建哈希
    content = f"{name}_{expression}_{datetime.now().strftime('%Y%m%d')}"
    hash_object = hashlib.md5(content.encode())
    hash_hex = hash_object.hexdigest()[:8]

    # 生成可读性好的ID
    safe_name = "".join(c.lower() if c.isalnum() else "_" for c in name)
    return f"{safe_name}_{hash_hex}"

def extract_factor_expression(factor_node: FactorNode) -> str:
    """提取因子表达式字符串表示

    Args:
        factor_node: 因子节点

    Returns:
        str: 因子表达式
    """
    try:
        return str(factor_node)
    except Exception:
        return f"<{type(factor_node).__name__}>"

def extract_dependencies(factor_node: FactorNode) -> List[str]:
    """提取因子依赖的原子数据

    Args:
        factor_node: 因子节点

    Returns:
        List[str]: 依赖的原子数据列表
    """
    dependencies = set()

    def _extract_recursive(node):
        """递归提取依赖"""
        # 如果是原子数据节点，添加到依赖中
        if hasattr(node, '__class__') and node.__class__.__name__ in [
            'Close', 'Volume', 'High', 'Low', 'Open', 'Turnover'
        ]:
            dependencies.add(node.__class__.__name__)

        # 递归处理子节点
        if hasattr(node, 'inputs') and node.inputs:
            for input_node in node.inputs:
                _extract_recursive(input_node)
        elif hasattr(node, 'input_node') and node.input_node:
            _extract_recursive(node.input_node)

    try:
        _extract_recursive(factor_node)
    except Exception:
        # 如果提取失败，返回空列表
        pass

    return list(dependencies)

def validate_metadata(metadata: Dict) -> Dict:
    """验证和标准化元数据

    Args:
        metadata: 原始元数据

    Returns:
        Dict: 验证后的元数据

    Raises:
        ValueError: 如果必要字段缺失
    """
    required_fields = ['name', 'description', 'category']
    for field in required_fields:
        if field not in metadata or not metadata[field]:
            raise ValueError(f"Required field '{field}' is missing or empty")

    # 标准化数据
    validated = metadata.copy()

    # 确保tags是列表
    if 'tags' not in validated:
        validated['tags'] = []
    elif not isinstance(validated['tags'], list):
        validated['tags'] = [validated['tags']]

    # 设置默认作者
    if 'author' not in validated:
        validated['author'] = 'unknown'

    # 设置默认状态
    if 'status' not in validated:
        validated['status'] = 'active'

    return validated

def save_json(data: Any, filepath: str):
    """保存数据到JSON文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

def load_json(filepath: str) -> Any:
    """从JSON文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        Any: 加载的数据
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file {filepath}: {e}")

def calculate_ic(factor_values, returns, method='pearson'):
    """计算信息系数(IC)

    Args:
        factor_values: 因子值 (pandas Series)
        returns: 收益率 (pandas Series)
        method: 计算方法 ('pearson' 或 'spearman')

    Returns:
        float: IC值
    """
    import pandas as pd

    # 确保索引对齐
    aligned_data = pd.concat([factor_values, returns], axis=1, join='inner')
    if aligned_data.empty:
        return 0.0

    factor_clean = aligned_data.iloc[:, 0].dropna()
    returns_clean = aligned_data.iloc[:, 1].dropna()

    if len(factor_clean) < 2 or len(returns_clean) < 2:
        return 0.0

    try:
        if method == 'pearson':
            return factor_clean.corr(returns_clean)
        elif method == 'spearman':
            return factor_clean.corr(returns_clean, method='spearman')
        else:
            raise ValueError(f"Unknown IC calculation method: {method}")
    except Exception:
        return 0.0

def calculate_factor_metrics(factor_values, returns):
    """计算因子性能指标

    Args:
        factor_values: 因子值时间序列 (pandas DataFrame)
        returns: 收益率时间序列 (pandas DataFrame)

    Returns:
        Dict: 性能指标字典
    """
    import pandas as pd
    import numpy as np

    metrics = {
        'ic_mean': 0.0,
        'ic_std': 0.0,
        'ir': 0.0,
        'rank_ic': 0.0,
        'coverage': 0.0,
        'periods_calculated': 0
    }

    try:
        # 确保数据对齐
        aligned_factor = factor_values.reindex(returns.index).fillna(method='ffill')
        aligned_returns = returns.copy()

        # 计算逐期IC
        ic_series = []
        rank_ic_series = []

        for date in aligned_factor.index:
            if date in aligned_returns.index:
                factor_cross_section = aligned_factor.loc[date].dropna()
                returns_cross_section = aligned_returns.loc[date].dropna()

                # 找到共同资产
                common_assets = factor_cross_section.index.intersection(returns_cross_section.index)
                if len(common_assets) >= 5:  # 至少需要5个资产
                    f_vals = factor_cross_section[common_assets]
                    r_vals = returns_cross_section[common_assets]

                    # 计算IC
                    ic = calculate_ic(f_vals, r_vals, method='pearson')
                    rank_ic = calculate_ic(f_vals, r_vals, method='spearman')

                    if not np.isnan(ic):
                        ic_series.append(ic)
                    if not np.isnan(rank_ic):
                        rank_ic_series.append(rank_ic)

        # 计算汇总指标
        if ic_series:
            ic_series = np.array(ic_series)
            metrics['ic_mean'] = np.mean(ic_series)
            metrics['ic_std'] = np.std(ic_series)
            metrics['ir'] = metrics['ic_mean'] / metrics['ic_std'] if metrics['ic_std'] > 0 else 0.0
            metrics['periods_calculated'] = len(ic_series)

        if rank_ic_series:
            metrics['rank_ic'] = np.mean(rank_ic_series)

        # 计算覆盖率
        total_periods = len(aligned_factor.index)
        calculated_periods = metrics['periods_calculated']
        metrics['coverage'] = calculated_periods / total_periods if total_periods > 0 else 0.0

    except Exception as e:
        print(f"Warning: Failed to calculate factor metrics: {e}")

    return metrics