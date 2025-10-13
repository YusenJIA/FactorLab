"""
因子组合管理模块

用于策略相关因子分组、同类因子批量管理和因子库分层组织。
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from factor_framework.config import FrequencyConfig

from .registry import FactorRegistry
from .monitor import FactorMonitor
from .utils import save_json, load_json

class FactorGroup:
    """因子组合管理 - 策略相关因子分组、同类因子批量管理"""

    def __init__(self, registry: FactorRegistry, groups_path: str = './data/factor_registry/groups.json'):
        """初始化因子组合管理器

        Args:
            registry: 因子注册中心
            groups_path: 组合配置文件路径
        """
        self.registry = registry
        self.groups_path = groups_path
        self._groups_data = {}
        self._load_groups()

    def _load_groups(self):
        """加载组合数据"""
        try:
            self._groups_data = load_json(self.groups_path)
        except Exception as e:
            print(f"Warning: Failed to load groups from {self.groups_path}: {e}")
            self._groups_data = {}

    def _save_groups(self):
        """保存组合数据"""
        try:
            save_json(self._groups_data, self.groups_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save groups to {self.groups_path}: {e}")

    def create_group(self,
                    group_name: str,
                    factor_ids: List[str],
                    description: str = None,
                    tags: List[str] = None) -> str:
        """创建因子组合

        Args:
            group_name: 组合名称
            factor_ids: 因子ID列表
            description: 组合描述
            tags: 组合标签

        Returns:
            str: 组合ID

        Raises:
            ValueError: 如果组合名称已存在或因子不存在
        """
        # 生成组合ID
        group_id = self._generate_group_id(group_name)

        # 检查组合是否已存在
        if group_id in self._groups_data:
            raise ValueError(f"Group '{group_name}' already exists with ID '{group_id}'")

        # 验证所有因子是否存在
        missing_factors = []
        for factor_id in factor_ids:
            if factor_id not in self.registry:
                missing_factors.append(factor_id)

        if missing_factors:
            raise ValueError(f"Factors not found: {missing_factors}")

        # 创建组合记录
        group_record = {
            'group_id': group_id,
            'name': group_name,
            'description': description or '',
            'factor_ids': factor_ids,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'performance_cache': {}
        }

        # 保存组合
        self._groups_data[group_id] = group_record
        self._save_groups()

        print(f"Created group '{group_name}' with ID '{group_id}' containing {len(factor_ids)} factors")
        return group_id

    def add_factors(self, group_id: str, factor_ids: List[str]):
        """向组合添加因子

        Args:
            group_id: 组合ID
            factor_ids: 要添加的因子ID列表

        Raises:
            ValueError: 如果组合不存在或因子不存在
        """
        if group_id not in self._groups_data:
            raise ValueError(f"Group '{group_id}' not found")

        # 验证因子存在性
        missing_factors = []
        for factor_id in factor_ids:
            if factor_id not in self.registry:
                missing_factors.append(factor_id)

        if missing_factors:
            raise ValueError(f"Factors not found: {missing_factors}")

        # 添加因子（避免重复）
        group_data = self._groups_data[group_id]
        existing_factors = set(group_data['factor_ids'])

        new_factors = []
        for factor_id in factor_ids:
            if factor_id not in existing_factors:
                group_data['factor_ids'].append(factor_id)
                new_factors.append(factor_id)

        if new_factors:
            group_data['updated_at'] = datetime.now().isoformat()
            self._save_groups()
            print(f"Added {len(new_factors)} factors to group '{group_id}'")
        else:
            print(f"No new factors added to group '{group_id}' (all already exist)")

    def remove_factors(self, group_id: str, factor_ids: List[str]):
        """从组合移除因子

        Args:
            group_id: 组合ID
            factor_ids: 要移除的因子ID列表

        Raises:
            ValueError: 如果组合不存在
        """
        if group_id not in self._groups_data:
            raise ValueError(f"Group '{group_id}' not found")

        group_data = self._groups_data[group_id]
        original_count = len(group_data['factor_ids'])

        # 移除因子
        group_data['factor_ids'] = [fid for fid in group_data['factor_ids'] if fid not in factor_ids]

        removed_count = original_count - len(group_data['factor_ids'])
        if removed_count > 0:
            group_data['updated_at'] = datetime.now().isoformat()
            self._save_groups()
            print(f"Removed {removed_count} factors from group '{group_id}'")
        else:
            print(f"No factors removed from group '{group_id}'")

    def evaluate_group(self,
                      group_id: str,
                      config: FrequencyConfig,
                      assets: List[str],
                      start_date: str,
                      end_date: str,
                      use_cache: bool = True) -> pd.DataFrame:
        """评估组合中所有因子的性能

        Args:
            group_id: 组合ID
            config: 频率配置
            assets: 资产列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            DataFrame: 所有因子的性能对比

        Raises:
            ValueError: 如果组合不存在
        """
        if group_id not in self._groups_data:
            raise ValueError(f"Group '{group_id}' not found")

        group_data = self._groups_data[group_id]
        factor_ids = group_data['factor_ids']

        if not factor_ids:
            print(f"Warning: Group '{group_id}' is empty")
            return pd.DataFrame()

        # 检查缓存
        cache_key = f"{config.input_freq}_{start_date}_{end_date}"
        if use_cache and cache_key in group_data.get('performance_cache', {}):
            print(f"Using cached performance data for group '{group_id}'")
            cached_data = group_data['performance_cache'][cache_key]
            return pd.DataFrame(cached_data)

        # 创建监控器进行批量评估
        monitor = FactorMonitor(self.registry)
        results_df = monitor.batch_evaluate(factor_ids, config, assets, start_date, end_date)

        # 添加组合信息
        if not results_df.empty:
            results_df['group_id'] = group_id
            results_df['group_name'] = group_data['name']

            # 缓存结果
            if use_cache:
                group_data['performance_cache'][cache_key] = results_df.to_dict('records')
                self._save_groups()

        return results_df

    def get_best_factors(self,
                        group_id: str,
                        metric: str = 'ir',
                        top_n: int = 5,
                        config: FrequencyConfig = None,
                        assets: List[str] = None,
                        start_date: str = None,
                        end_date: str = None) -> List[str]:
        """获取组合中表现最好的N个因子

        Args:
            group_id: 组合ID
            metric: 评估指标 ('ir', 'ic_mean', 'rank_ic')
            top_n: 返回前N个因子
            config: 频率配置（如果需要重新评估）
            assets: 资产列表（如果需要重新评估）
            start_date: 开始日期（如果需要重新评估）
            end_date: 结束日期（如果需要重新评估）

        Returns:
            List[str]: 表现最好的因子ID列表

        Raises:
            ValueError: 如果组合不存在或指标无效
        """
        if group_id not in self._groups_data:
            raise ValueError(f"Group '{group_id}' not found")

        valid_metrics = ['ir', 'ic_mean', 'rank_ic', 'coverage']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Valid options: {valid_metrics}")

        # 如果提供了评估参数，进行新的评估
        if all([config, assets, start_date, end_date]):
            performance_df = self.evaluate_group(group_id, config, assets, start_date, end_date)
        else:
            # 尝试使用最近的缓存数据
            group_data = self._groups_data[group_id]
            cache = group_data.get('performance_cache', {})

            if not cache:
                print(f"No performance data available for group '{group_id}'")
                return []

            # 使用最新的缓存数据
            latest_cache_key = max(cache.keys())
            performance_df = pd.DataFrame(cache[latest_cache_key])

        if performance_df.empty:
            return []

        # 排序并获取top N
        if metric in performance_df.columns:
            sorted_df = performance_df.sort_values(metric, ascending=False, na_position='last')
            top_factors = sorted_df.head(top_n)['factor_id'].tolist()
            return top_factors
        else:
            print(f"Warning: Metric '{metric}' not found in performance data")
            return []

    def list_groups(self, tag: str = None) -> List[Dict]:
        """列出所有组合

        Args:
            tag: 按标签筛选

        Returns:
            List[Dict]: 组合信息列表
        """
        results = []

        for group_id, group_data in self._groups_data.items():
            # 应用标签筛选
            if tag and tag not in group_data.get('tags', []):
                continue

            group_info = {
                'group_id': group_id,
                'name': group_data['name'],
                'description': group_data['description'],
                'factor_count': len(group_data['factor_ids']),
                'tags': group_data.get('tags', []),
                'created_at': group_data['created_at'],
                'updated_at': group_data['updated_at']
            }
            results.append(group_info)

        # 按创建时间倒序
        results.sort(key=lambda x: x['created_at'], reverse=True)
        return results

    def get_group_details(self, group_id: str) -> Dict:
        """获取组合详细信息

        Args:
            group_id: 组合ID

        Returns:
            Dict: 组合详细信息

        Raises:
            ValueError: 如果组合不存在
        """
        if group_id not in self._groups_data:
            raise ValueError(f"Group '{group_id}' not found")

        group_data = self._groups_data[group_id].copy()

        # 添加因子详细信息
        factor_details = []
        for factor_id in group_data['factor_ids']:
            try:
                metadata = self.registry.get_metadata(factor_id)
                factor_details.append({
                    'factor_id': factor_id,
                    'name': metadata.name,
                    'category': metadata.category,
                    'status': metadata.status
                })
            except Exception as e:
                factor_details.append({
                    'factor_id': factor_id,
                    'name': 'Unknown',
                    'category': 'Unknown',
                    'status': 'Error',
                    'error': str(e)
                })

        group_data['factor_details'] = factor_details
        return group_data

    def delete_group(self, group_id: str):
        """删除组合

        Args:
            group_id: 组合ID

        Raises:
            ValueError: 如果组合不存在
        """
        if group_id not in self._groups_data:
            raise ValueError(f"Group '{group_id}' not found")

        group_name = self._groups_data[group_id]['name']
        del self._groups_data[group_id]
        self._save_groups()

        print(f"Deleted group '{group_name}' (ID: {group_id})")

    def compare_groups(self,
                      group_ids: List[str],
                      config: FrequencyConfig,
                      assets: List[str],
                      start_date: str,
                      end_date: str) -> pd.DataFrame:
        """比较多个组合的性能

        Args:
            group_ids: 组合ID列表
            config: 频率配置
            assets: 资产列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 组合性能对比
        """
        group_summaries = []

        for group_id in group_ids:
            try:
                # 评估组合
                group_performance = self.evaluate_group(group_id, config, assets, start_date, end_date)

                if not group_performance.empty:
                    # 计算组合汇总指标
                    summary = {
                        'group_id': group_id,
                        'group_name': self._groups_data[group_id]['name'],
                        'factor_count': len(group_performance),
                        'avg_ir': group_performance['ir'].mean(),
                        'best_ir': group_performance['ir'].max(),
                        'avg_ic': group_performance['ic_mean'].mean(),
                        'avg_coverage': group_performance['coverage'].mean()
                    }
                    group_summaries.append(summary)

            except Exception as e:
                print(f"Error evaluating group {group_id}: {e}")

        # 转换为DataFrame
        comparison_df = pd.DataFrame(group_summaries)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('avg_ir', ascending=False)

        return comparison_df

    def _generate_group_id(self, group_name: str) -> str:
        """生成组合ID

        Args:
            group_name: 组合名称

        Returns:
            str: 组合ID
        """
        import hashlib
        import re

        # 清理名称
        clean_name = re.sub(r'[^\w\s-]', '', group_name).strip()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)

        # 生成短哈希
        hash_object = hashlib.md5(f"{group_name}_{datetime.now().isoformat()}".encode())
        hash_hex = hash_object.hexdigest()[:6]

        return f"{clean_name.lower()}_{hash_hex}"

    def get_group_statistics(self) -> Dict:
        """获取组合统计信息

        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_groups': len(self._groups_data),
            'total_factors_in_groups': 0,
            'avg_factors_per_group': 0,
            'groups_by_tag': {}
        }

        if not self._groups_data:
            return stats

        factor_counts = []
        all_tags = []

        for group_data in self._groups_data.values():
            factor_count = len(group_data['factor_ids'])
            factor_counts.append(factor_count)
            stats['total_factors_in_groups'] += factor_count

            # 统计标签
            for tag in group_data.get('tags', []):
                all_tags.append(tag)

        # 计算平均值
        if factor_counts:
            stats['avg_factors_per_group'] = sum(factor_counts) / len(factor_counts)

        # 统计标签分布
        from collections import Counter
        tag_counts = Counter(all_tags)
        stats['groups_by_tag'] = dict(tag_counts)

        return stats

    def __len__(self) -> int:
        """返回组合数量"""
        return len(self._groups_data)

    def __contains__(self, group_id: str) -> bool:
        """检查组合是否存在"""
        return group_id in self._groups_data

    def __iter__(self):
        """迭代所有组合ID"""
        return iter(self._groups_data.keys())