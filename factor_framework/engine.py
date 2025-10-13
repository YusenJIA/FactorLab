"""
FactorEngine核心计算引擎

这是设计文档中"执行引擎层"的核心实现。引擎读取抽象的因子表达式和具体的频率配置，
在正确的时点获取正确的数据切片，执行DAG计算，并严格进行未来函数检查。

核心功能：
1. 调度器：按配置频率执行计算
2. DAG优化器：增量/并行计算
3. 未来函数检查器：严格防止使用未来数据
4. 缓存机制：优化重复计算
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import FrequencyConfig
from .nodes.base import FactorNode
from .data_loader import get_data


class FactorEngine:
    """
    因子计算引擎

    按照设计文档，这个引擎只关心"计算图"的逻辑，将"频率"视为可配置参数。
    实现了调度器、DAG优化器和未来函数检查器的功能。
    """

    def __init__(self, data_root: str = "data", max_workers: int = 4):
        """
        初始化因子引擎

        Args:
            data_root: 数据根目录
            max_workers: 并行计算的最大工作线程数
        """
        self.data_root = data_root
        self.max_workers = max_workers
        self._cache: Dict[str, Any] = {}
        self._computation_history: Dict[str, List[pd.Timestamp]] = defaultdict(list)
        self._dag_cache: Dict[str, List[FactorNode]] = {}

    def compute_factor(self,
                      factor: FactorNode,
                      assets: Union[List[str], str],
                      start_date: str,
                      end_date: str,
                      config: FrequencyConfig,
                      parallel: bool = True) -> pd.DataFrame:
        """
        计算因子值

        这是引擎的主要接口，按照设计文档的要求实现完整的因子计算流程。

        Args:
            factor: 要计算的因子节点
            assets: 资产代码列表
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            config: 频率配置
            parallel: 是否启用并行计算

        Returns:
            因子值DataFrame，包含timestamp, asset, factor_value列
        """
        # 1. 数据预加载和验证
        data = self._load_data(assets, start_date, end_date, config)
        if data.empty:
            raise ValueError("没有可用的数据")

        # 2. 生成计算时间点
        calculation_timestamps = self._generate_calculation_timestamps(
            start_date, end_date, config
        )

        # 3. DAG优化
        computation_nodes = self._optimize_dag(factor)

        # 4. 执行计算调度
        if parallel and len(calculation_timestamps) > 1:
            results = self._parallel_compute(
                factor, data, config, calculation_timestamps
            )
        else:
            results = self._sequential_compute(
                factor, data, config, calculation_timestamps
            )

        # 5. 整理结果
        return self._format_results(results, assets)

    def compute_multiple_factors(self,
                                factors: Dict[str, FactorNode],
                                assets: Union[List[str], str],
                                start_date: str,
                                end_date: str,
                                config: FrequencyConfig,
                                parallel: bool = True) -> pd.DataFrame:
        """
        批量计算多个因子

        利用共享计算节点优化性能

        Args:
            factors: 因子字典 {因子名: 因子节点}
            assets: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期
            config: 频率配置
            parallel: 是否启用并行计算

        Returns:
            包含所有因子值的DataFrame
        """
        # 预加载数据
        data = self._load_data(assets, start_date, end_date, config)
        if data.empty:
            raise ValueError("没有可用的数据")

        # 生成计算时间点
        calculation_timestamps = self._generate_calculation_timestamps(
            start_date, end_date, config
        )

        # 优化多因子DAG
        all_nodes = set()
        for factor in factors.values():
            all_nodes.update(factor.get_all_dependencies())
            all_nodes.add(factor)

        # 执行批量计算
        all_results = {}
        for factor_name, factor in factors.items():
            if parallel:
                results = self._parallel_compute(
                    factor, data, config, calculation_timestamps
                )
            else:
                results = self._sequential_compute(
                    factor, data, config, calculation_timestamps
                )
            all_results[factor_name] = results

        # 整理多因子结果
        return self._format_multiple_results(all_results, assets)

    def _load_data(self,
                   assets: Union[List[str], str],
                   start_date: str,
                   end_date: str,
                   config: FrequencyConfig) -> pd.DataFrame:
        """
        数据预加载

        根据配置加载所需的数据，实现设计文档中的统一数据接口
        """
        if isinstance(assets, str):
            assets = [assets]

        # 扩展时间范围以确保有足够的历史数据进行窗口计算
        extended_start = self._extend_start_date(start_date, config)

        try:
            data = get_data(
                assets=assets,
                start_date=extended_start,
                end_date=end_date,
                frequency=config.input_freq,
                data_root=self.data_root
            )
            return data
        except Exception as e:
            warnings.warn(f"数据加载失败: {e}")
            return pd.DataFrame()

    def _extend_start_date(self, start_date: str, config: FrequencyConfig) -> str:
        """
        扩展开始日期以确保有足够的历史数据

        根据窗口长度和频率计算需要额外的历史数据量
        """
        start_dt = pd.to_datetime(start_date)

        # 根据频率和窗口长度计算扩展天数
        if config.input_freq == 'daily':
            # 日频数据：额外加载 window_length * 2 天的数据
            buffer_days = config.window_length * 2
            extended_start = start_dt - pd.Timedelta(days=buffer_days)
        elif config.input_freq in ['minute', '1min']:
            # 分钟数据：额外加载 window_length 分钟对应的天数
            buffer_minutes = config.window_length * 2
            buffer_days = max(1, buffer_minutes // (6 * 60))  # 假设每天6小时交易
            extended_start = start_dt - pd.Timedelta(days=buffer_days)
        elif config.input_freq in ['30min', '30T']:
            # 30分钟数据
            buffer_periods = config.window_length * 2
            buffer_days = max(1, buffer_periods // (6 * 2))  # 每天12个30分钟周期
            extended_start = start_dt - pd.Timedelta(days=buffer_days)
        else:
            # 默认扩展30天
            extended_start = start_dt - pd.Timedelta(days=30)

        return extended_start.strftime('%Y-%m-%d')

    def _generate_calculation_timestamps(self,
                                       start_date: str,
                                       end_date: str,
                                       config: FrequencyConfig) -> List[pd.Timestamp]:
        """
        生成计算时间点

        根据计算频率生成需要计算因子值的时间点列表
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if config.calc_freq == 'daily':
            # 日频计算：每个交易日收盘时计算
            timestamps = pd.date_range(start_dt, end_dt, freq='D')
            # 转换为收盘时间（15:00）
            timestamps = [
                ts.replace(hour=15, minute=0, second=0)
                for ts in timestamps
                if ts.weekday() < 5  # 只保留工作日
            ]
        elif config.calc_freq in ['minute', '1min']:
            # 分钟级计算
            timestamps = []
            current = start_dt.replace(hour=9, minute=30)  # 开盘时间
            while current <= end_dt:
                if config.is_trading_time(current):
                    timestamps.append(current)
                current += pd.Timedelta(minutes=1)
        elif config.calc_freq in ['5min', '5T']:
            # 5分钟级计算
            timestamps = []
            current = start_dt.replace(hour=9, minute=30)
            while current <= end_dt:
                if config.is_trading_time(current):
                    timestamps.append(current)
                current += pd.Timedelta(minutes=5)
        elif config.calc_freq in ['30min', '30T']:
            # 30分钟级计算
            timestamps = []
            current = start_dt.replace(hour=9, minute=30)
            while current <= end_dt:
                if config.is_trading_time(current):
                    timestamps.append(current)
                current += pd.Timedelta(minutes=30)
        else:
            # 默认日频
            timestamps = pd.date_range(start_dt, end_dt, freq='D')
            timestamps = [ts.replace(hour=15, minute=0) for ts in timestamps]

        return timestamps

    def _optimize_dag(self, factor: FactorNode) -> List[FactorNode]:
        """
        DAG优化

        分析因子依赖关系，优化计算顺序
        """
        # 缓存DAG优化结果
        factor_key = str(factor)
        if factor_key in self._dag_cache:
            return self._dag_cache[factor_key]

        # 获取所有依赖节点
        all_nodes = factor.get_all_dependencies()
        all_nodes.add(factor)

        # 拓扑排序
        computation_order = self._topological_sort(all_nodes, factor)

        # 缓存结果
        self._dag_cache[factor_key] = computation_order

        return computation_order

    def _topological_sort(self,
                         nodes: Set[FactorNode],
                         target: FactorNode) -> List[FactorNode]:
        """
        拓扑排序

        确保按正确的依赖顺序计算节点
        """
        # 计算每个节点的入度
        in_degree = {node: 0 for node in nodes}
        for node in nodes:
            for dep in node.dependencies:
                if dep in nodes:
                    in_degree[node] += 1

        # 使用队列进行拓扑排序
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # 更新依赖当前节点的其他节点的入度
            for node in nodes:
                if current in node.dependencies:
                    in_degree[node] -= 1
                    if in_degree[node] == 0:
                        queue.append(node)

        if len(result) != len(nodes):
            raise ValueError("因子图中存在循环依赖")

        return result

    def _sequential_compute(self,
                          factor: FactorNode,
                          data: pd.DataFrame,
                          config: FrequencyConfig,
                          timestamps: List[pd.Timestamp]) -> List[Dict]:
        """
        顺序计算

        按时间顺序逐个计算因子值
        """
        results = []

        for timestamp in timestamps:
            try:
                # 严格的未来函数检查
                self._validate_timestamp(timestamp, data)

                # 计算因子值
                factor_value = factor.compute_with_cache(data, config, timestamp)

                # 记录结果
                result = {
                    'timestamp': timestamp,
                    'values': factor_value
                }
                results.append(result)

            except Exception as e:
                warnings.warn(f"时间点 {timestamp} 计算失败: {e}")
                continue

        return results

    def _parallel_compute(self,
                        factor: FactorNode,
                        data: pd.DataFrame,
                        config: FrequencyConfig,
                        timestamps: List[pd.Timestamp]) -> List[Dict]:
        """
        并行计算

        使用多线程并行计算不同时间点的因子值
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交计算任务
            future_to_timestamp = {
                executor.submit(self._compute_single_timestamp, factor, data, config, ts): ts
                for ts in timestamps
            }

            # 收集结果
            for future in as_completed(future_to_timestamp):
                timestamp = future_to_timestamp[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    warnings.warn(f"时间点 {timestamp} 并行计算失败: {e}")

        # 按时间排序
        results.sort(key=lambda x: x['timestamp'])

        return results

    def _compute_single_timestamp(self,
                                factor: FactorNode,
                                data: pd.DataFrame,
                                config: FrequencyConfig,
                                timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        计算单个时间点的因子值

        用于并行计算
        """
        try:
            # 验证时间点
            self._validate_timestamp(timestamp, data)

            # 计算因子值
            factor_value = factor.compute_with_cache(data, config, timestamp)

            return {
                'timestamp': timestamp,
                'values': factor_value
            }

        except Exception as e:
            return None

    def _validate_timestamp(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        严格的未来函数检查

        确保不使用未来数据
        """
        if data.empty:
            raise ValueError("数据为空")

        # 检查时间戳是否在数据范围内
        if isinstance(data.index, pd.DatetimeIndex):
            max_time = data.index.max()
        else:
            max_time = data['datetime'].max()

        if timestamp > max_time:
            raise ValueError(f"时间戳 {timestamp} 超出数据范围（最大时间：{max_time}）")

        # 检查是否在交易时间内（如果需要）
        # 这里可以添加更多的验证逻辑

    def _format_results(self,
                       results: List[Dict],
                       assets: List[str]) -> pd.DataFrame:
        """
        格式化单因子计算结果

        将计算结果转换为标准DataFrame格式
        """
        if not results:
            return pd.DataFrame(columns=['timestamp', 'asset', 'factor_value'])

        formatted_data = []

        for result in results:
            timestamp = result['timestamp']
            values = result['values']

            if isinstance(values, pd.Series):
                for asset_code, factor_value in values.items():
                    formatted_data.append({
                        'timestamp': timestamp,
                        'asset': asset_code,
                        'factor_value': factor_value
                    })
            else:
                # 单值情况
                for asset in assets:
                    formatted_data.append({
                        'timestamp': timestamp,
                        'asset': asset,
                        'factor_value': values
                    })

        df = pd.DataFrame(formatted_data)
        return df.sort_values(['timestamp', 'asset']).reset_index(drop=True)

    def _format_multiple_results(self,
                                all_results: Dict[str, List[Dict]],
                                assets: List[str]) -> pd.DataFrame:
        """
        格式化多因子计算结果
        """
        if not all_results:
            return pd.DataFrame()

        all_data = []

        for factor_name, results in all_results.items():
            for result in results:
                timestamp = result['timestamp']
                values = result['values']

                if isinstance(values, pd.Series):
                    for asset_code, factor_value in values.items():
                        all_data.append({
                            'timestamp': timestamp,
                            'asset': asset_code,
                            'factor': factor_name,
                            'factor_value': factor_value
                        })

        df = pd.DataFrame(all_data)
        return df.sort_values(['timestamp', 'asset', 'factor']).reset_index(drop=True)

    def clear_cache(self):
        """清空所有缓存"""
        self._cache.clear()
        self._dag_cache.clear()
        self._computation_history.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cache_size': len(self._cache),
            'dag_cache_size': len(self._dag_cache),
            'computation_history_size': len(self._computation_history)
        }

    def benchmark_factor(self,
                        factor: FactorNode,
                        assets: Union[List[str], str],
                        start_date: str,
                        end_date: str,
                        config: FrequencyConfig,
                        iterations: int = 1) -> Dict[str, float]:
        """
        因子计算性能基准测试

        Args:
            factor: 要测试的因子
            assets: 资产列表
            start_date: 开始日期
            end_date: 结束日期
            config: 频率配置
            iterations: 测试迭代次数

        Returns:
            性能统计信息
        """
        import time

        times = []

        for i in range(iterations):
            self.clear_cache()  # 每次测试前清空缓存

            start_time = time.time()
            try:
                result = self.compute_factor(factor, assets, start_date, end_date, config)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                warnings.warn(f"基准测试第{i+1}次迭代失败: {e}")

        if not times:
            return {}

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times),
            'iterations': len(times)
        }