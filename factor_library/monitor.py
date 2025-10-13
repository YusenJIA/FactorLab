"""
因子性能监控模块

使用 FactorEngine 计算因子值，然后评估性能指标。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from factor_framework.engine import FactorEngine
from factor_framework.config import FrequencyConfig
from factor_framework.data_loader import get_data

from .registry import FactorRegistry
from .metadata import FactorMetadata
from .utils import calculate_factor_metrics, save_json, load_json

class FactorMonitor:
    """因子性能监控 - 使用 FactorEngine 计算因子值，然后评估性能"""

    def __init__(self, registry: FactorRegistry, engine: FactorEngine = None):
        """初始化性能监控器

        Args:
            registry: 因子注册中心
            engine: 因子计算引擎，如果不提供则创建新实例
        """
        self.registry = registry
        self.engine = engine or FactorEngine()

        # 性能缓存路径
        self.performance_cache_dir = './data/factor_registry/performance'

    def calc_performance(self,
                        factor_id: str,
                        assets: List[str],
                        start_date: str,
                        end_date: str,
                        config: FrequencyConfig,
                        returns_data: pd.DataFrame = None) -> Dict:
        """计算因子性能指标

        工作流程:
        1. 从注册中心获取 FactorNode
        2. 使用 FactorEngine 计算因子值
        3. 计算性能指标（IC, IR等）

        Args:
            factor_id: 因子ID
            assets: 资产列表
            start_date: 开始日期
            end_date: 结束日期
            config: 频率配置
            returns_data: 收益率数据，如果不提供则自动计算

        Returns:
            metrics: {
                'ic_mean': IC均值,
                'ic_std': IC标准差,
                'ir': 信息比率,
                'rank_ic': Rank IC,
                'turnover': 换手率,
                'coverage': 覆盖率
            }
        """
        try:
            # 1. 获取因子定义
            factor_node, factor_configs = self.registry.get_factor(factor_id)

            # 检查配置是否支持
            if config not in factor_configs:
                print(f"Warning: Config {config} not in registered configs for factor {factor_id}")

            # 2. 计算因子值
            print(f"Computing factor values for {factor_id}...")
            factor_values = self.engine.compute_factor(
                factor_node, assets, start_date, end_date, config
            )

            if factor_values is None or factor_values.empty:
                print(f"Warning: No factor values computed for {factor_id}")
                return self._get_empty_metrics()

            # 3. 获取或计算收益率数据
            if returns_data is None:
                returns_data = self._calculate_returns(assets, start_date, end_date, config)

            if returns_data is None or returns_data.empty:
                print(f"Warning: No returns data available for performance calculation")
                return self._get_empty_metrics()

            # 4. 计算性能指标
            print(f"Calculating performance metrics for {factor_id}...")
            metrics = calculate_factor_metrics(factor_values, returns_data)

            # 5. 添加额外信息
            metrics.update({
                'calculation_date': datetime.now().isoformat(),
                'assets_count': len(assets),
                'start_date': start_date,
                'end_date': end_date,
                'frequency': config.input_freq,
                'factor_id': factor_id
            })

            # 6. 更新元数据中的性能信息
            self._update_factor_performance(factor_id, config.input_freq, metrics)

            # 7. 缓存结果
            self._cache_performance(factor_id, config.input_freq, metrics)

            return metrics

        except Exception as e:
            print(f"Error calculating performance for factor {factor_id}: {e}")
            return self._get_empty_metrics()

    def monitor_multi_freq(self,
                          factor_id: str,
                          assets: List[str],
                          date_range: Tuple[str, str]) -> pd.DataFrame:
        """监控因子在不同频率下的表现

        Args:
            factor_id: 因子ID
            assets: 资产列表
            date_range: 日期范围 (start_date, end_date)

        Returns:
            DataFrame with columns: ['frequency', 'ic_mean', 'ir', 'coverage', ...]
        """
        start_date, end_date = date_range
        results = []

        try:
            # 获取因子支持的所有配置
            _, factor_configs = self.registry.get_factor(factor_id)

            for config in factor_configs:
                print(f"Evaluating {factor_id} at frequency: {config.input_freq}")

                # 计算性能
                metrics = self.calc_performance(
                    factor_id, assets, start_date, end_date, config
                )

                # 添加频率信息
                result = {
                    'frequency': config.input_freq,
                    'window_length': config.window_length,
                    **metrics
                }
                results.append(result)

            # 转换为DataFrame
            df = pd.DataFrame(results)
            return df

        except Exception as e:
            print(f"Error in multi-frequency monitoring for {factor_id}: {e}")
            return pd.DataFrame()

    def detect_decay(self,
                    factor_id: str,
                    config: FrequencyConfig,
                    window: int = 90) -> Dict:
        """检测因子衰减

        Args:
            factor_id: 因子ID
            config: 频率配置
            window: 检测窗口（天数）

        Returns:
            Dict: 衰减检测结果
        """
        try:
            # 获取历史性能数据
            cached_performance = self._load_cached_performance(factor_id, config.input_freq)

            if not cached_performance:
                return {'decay_detected': False, 'reason': 'No historical data'}

            # 分析性能趋势（这里简化实现）
            recent_ic = cached_performance.get('ic_mean', 0)
            historical_ic = 0.05  # 假设的历史基准

            decay_ratio = recent_ic / historical_ic if historical_ic > 0 else 0

            return {
                'decay_detected': decay_ratio < 0.7,  # 如果IC下降超过30%认为衰减
                'decay_ratio': decay_ratio,
                'recent_ic': recent_ic,
                'historical_ic': historical_ic,
                'detection_date': datetime.now().isoformat()
            }

        except Exception as e:
            return {'decay_detected': False, 'error': str(e)}

    def batch_evaluate(self,
                      factor_ids: List[str],
                      config: FrequencyConfig,
                      assets: List[str],
                      start_date: str,
                      end_date: str) -> pd.DataFrame:
        """批量评估多个因子

        Args:
            factor_ids: 因子ID列表
            config: 频率配置
            assets: 资产列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 所有因子的性能对比
        """
        results = []

        # 预先计算收益率数据
        returns_data = self._calculate_returns(assets, start_date, end_date, config)

        for factor_id in factor_ids:
            print(f"Evaluating factor: {factor_id}")

            try:
                # 获取因子元数据
                metadata = self.registry.get_metadata(factor_id)

                # 计算性能
                metrics = self.calc_performance(
                    factor_id, assets, start_date, end_date, config, returns_data
                )

                # 合并结果
                result = {
                    'factor_id': factor_id,
                    'factor_name': metadata.name,
                    'category': metadata.category,
                    **metrics
                }
                results.append(result)

            except Exception as e:
                print(f"Error evaluating factor {factor_id}: {e}")
                # 添加错误记录
                results.append({
                    'factor_id': factor_id,
                    'factor_name': 'Error',
                    'category': 'Error',
                    'ic_mean': np.nan,
                    'ir': np.nan,
                    'error': str(e)
                })

        # 转换为DataFrame并排序
        df = pd.DataFrame(results)
        if not df.empty and 'ir' in df.columns:
            df = df.sort_values('ir', ascending=False, na_position='last')

        return df

    def evaluate_factor(self,
                       factor_id: str,
                       assets: List[str],
                       start_date: str,
                       end_date: str,
                       returns_data: pd.DataFrame = None) -> Dict:
        """评估因子在所有支持频率下的性能

        Args:
            factor_id: 因子ID
            assets: 资产列表
            start_date: 开始日期
            end_date: 结束日期
            returns_data: 收益率数据

        Returns:
            Dict: 各频率下的性能结果
        """
        results = {}

        try:
            _, factor_configs = self.registry.get_factor(factor_id)

            for config in factor_configs:
                freq_key = config.input_freq
                print(f"Evaluating {factor_id} at {freq_key}...")

                performance = self.calc_performance(
                    factor_id, assets, start_date, end_date, config, returns_data
                )

                results[freq_key] = performance

            return results

        except Exception as e:
            print(f"Error evaluating factor {factor_id}: {e}")
            return {}

    def _calculate_returns(self,
                          assets: List[str],
                          start_date: str,
                          end_date: str,
                          config: FrequencyConfig) -> pd.DataFrame:
        """计算收益率数据

        Args:
            assets: 资产列表
            start_date: 开始日期
            end_date: 结束日期
            config: 频率配置

        Returns:
            DataFrame: 收益率数据
        """
        try:
            # 获取价格数据
            price_data = get_data(
                assets, start_date, end_date, config.input_freq
            )

            if price_data is None or price_data.empty:
                return pd.DataFrame()

            # 计算收益率
            returns = price_data.groupby('asset').apply(
                lambda x: x.set_index('date')['close'].pct_change()
            ).unstack(level=0)

            # 移除第一行（NaN）
            returns = returns.iloc[1:]

            return returns

        except Exception as e:
            print(f"Error calculating returns: {e}")
            return pd.DataFrame()

    def _update_factor_performance(self,
                                  factor_id: str,
                                  frequency: str,
                                  metrics: Dict):
        """更新因子元数据中的性能信息

        Args:
            factor_id: 因子ID
            frequency: 频率
            metrics: 性能指标
        """
        try:
            metadata = self.registry.get_metadata(factor_id)
            metadata.update_performance(frequency, metrics)

            # 更新注册表
            self.registry.update_metadata(factor_id, {
                'performance_by_freq': metadata.performance_by_freq,
                'last_calc_time': metadata.last_calc_time
            })

        except Exception as e:
            print(f"Warning: Failed to update factor performance metadata: {e}")

    def _cache_performance(self,
                          factor_id: str,
                          frequency: str,
                          metrics: Dict):
        """缓存性能结果

        Args:
            factor_id: 因子ID
            frequency: 频率
            metrics: 性能指标
        """
        try:
            import os
            os.makedirs(self.performance_cache_dir, exist_ok=True)

            cache_file = f"{self.performance_cache_dir}/{factor_id}_{frequency}.json"
            save_json(metrics, cache_file)

        except Exception as e:
            print(f"Warning: Failed to cache performance data: {e}")

    def _load_cached_performance(self,
                                factor_id: str,
                                frequency: str) -> Dict:
        """加载缓存的性能数据

        Args:
            factor_id: 因子ID
            frequency: 频率

        Returns:
            Dict: 缓存的性能数据
        """
        try:
            cache_file = f"{self.performance_cache_dir}/{factor_id}_{frequency}.json"
            return load_json(cache_file)
        except Exception:
            return {}

    def _get_empty_metrics(self) -> Dict:
        """返回空的性能指标"""
        return {
            'ic_mean': 0.0,
            'ic_std': 0.0,
            'ir': 0.0,
            'rank_ic': 0.0,
            'coverage': 0.0,
            'periods_calculated': 0,
            'calculation_date': datetime.now().isoformat()
        }

    def get_performance_summary(self, factor_id: str) -> Dict:
        """获取因子性能摘要

        Args:
            factor_id: 因子ID

        Returns:
            Dict: 性能摘要
        """
        try:
            metadata = self.registry.get_metadata(factor_id)
            performance = metadata.get_performance()

            summary = {
                'factor_id': factor_id,
                'factor_name': metadata.name,
                'category': metadata.category,
                'last_calc_time': metadata.last_calc_time,
                'performance_by_frequency': performance
            }

            return summary

        except Exception as e:
            return {'error': str(e)}