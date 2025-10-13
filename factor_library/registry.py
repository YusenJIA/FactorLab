"""
因子注册中心

负责因子的注册、存储、检索和管理。
"""

import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from factor_framework.nodes.base import FactorNode
from factor_framework.config import FrequencyConfig

from .metadata import FactorMetadata
from .utils import (
    serialize_factor_node, deserialize_factor_node,
    serialize_frequency_configs, deserialize_frequency_configs,
    generate_factor_id, extract_factor_expression,
    extract_dependencies, validate_metadata,
    save_json, load_json
)

class FactorRegistry:
    """因子注册中心 - 管理所有 FactorNode 的定义"""

    def __init__(self, registry_path: str = './data/factor_registry/factors.json'):
        """初始化注册中心

        Args:
            registry_path: 注册表存储路径
        """
        self.registry_path = registry_path
        self._registry_data = {}
        self._load_registry()

    def _load_registry(self):
        """加载注册表数据"""
        try:
            self._registry_data = load_json(self.registry_path)
        except Exception as e:
            print(f"Warning: Failed to load registry from {self.registry_path}: {e}")
            self._registry_data = {}

    def _save_registry(self):
        """保存注册表数据"""
        try:
            save_json(self._registry_data, self.registry_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save registry to {self.registry_path}: {e}")

    def register_factor(self,
                       factor_node: FactorNode,
                       factor_configs: List[FrequencyConfig],
                       metadata: Dict) -> str:
        """注册新因子

        Args:
            factor_node: 因子计算节点（来自factor_framework）
            factor_configs: 该因子支持的频率配置列表
            metadata: 因子元数据

        Returns:
            factor_id: 因子唯一标识

        Example:
            >>> from factor_framework import Close, Mean, Volume, FrequencyConfig
            >>>
            >>> price_to_vol = Close() / Mean(Volume(), window_length=20)
            >>> configs = [
            >>>     FrequencyConfig(input_freq='daily', window_length=20, calc_freq='daily'),
            >>>     FrequencyConfig(input_freq='minute', window_length=30, calc_freq='5min')
            >>> ]
            >>>
            >>> factor_id = registry.register_factor(
            >>>     factor_node=price_to_vol,
            >>>     factor_configs=configs,
            >>>     metadata={
            >>>         'name': '价格成交量比',
            >>>         'category': 'liquidity',
            >>>         'description': '收盘价除以平均成交量'
            >>>     }
            >>> )
        """
        # 验证元数据
        validated_metadata = validate_metadata(metadata)

        # 生成因子ID
        factor_id = generate_factor_id(factor_node, validated_metadata)

        # 检查是否已存在
        if factor_id in self._registry_data:
            raise ValueError(f"Factor with ID '{factor_id}' already exists")

        # 提取因子信息
        factor_expression = extract_factor_expression(factor_node)
        dependencies = extract_dependencies(factor_node)
        supported_frequencies = [config.input_freq for config in factor_configs]

        # 创建元数据对象
        factor_metadata = FactorMetadata(
            factor_id=factor_id,
            name=validated_metadata['name'],
            description=validated_metadata['description'],
            category=validated_metadata['category'],
            tags=validated_metadata.get('tags', []),
            author=validated_metadata.get('author', 'unknown'),
            factor_expression=factor_expression,
            supported_frequencies=supported_frequencies,
            dependencies=dependencies
        )

        # 序列化因子节点和配置
        serialized_node = serialize_factor_node(factor_node)
        serialized_configs = serialize_frequency_configs(factor_configs)

        # 存储到注册表
        self._registry_data[factor_id] = {
            'metadata': factor_metadata.to_dict(),
            'versions': {
                'v1.0.0': {
                    'factor_node': serialized_node,
                    'factor_configs': serialized_configs,
                    'created_at': datetime.now().isoformat(),
                    'is_active': True
                }
            },
            'current_version': 'v1.0.0'
        }

        # 保存注册表
        self._save_registry()

        print(f"Successfully registered factor: {factor_id}")
        return factor_id

    def get_factor(self, factor_id: str, version: str = None) -> Tuple[FactorNode, List[FrequencyConfig]]:
        """获取因子定义

        Args:
            factor_id: 因子ID
            version: 版本号，如果不指定则返回当前版本

        Returns:
            Tuple[FactorNode, List[FrequencyConfig]]: 因子节点和配置列表

        Raises:
            ValueError: 如果因子不存在
        """
        if factor_id not in self._registry_data:
            raise ValueError(f"Factor '{factor_id}' not found")

        factor_data = self._registry_data[factor_id]

        # 确定版本
        if version is None:
            version = factor_data['current_version']

        if version not in factor_data['versions']:
            raise ValueError(f"Version '{version}' not found for factor '{factor_id}'")

        version_data = factor_data['versions'][version]

        # 反序列化
        factor_node = deserialize_factor_node(version_data['factor_node'])
        factor_configs = deserialize_frequency_configs(version_data['factor_configs'])

        return factor_node, factor_configs

    def get_metadata(self, factor_id: str) -> FactorMetadata:
        """获取因子元数据

        Args:
            factor_id: 因子ID

        Returns:
            FactorMetadata: 因子元数据

        Raises:
            ValueError: 如果因子不存在
        """
        if factor_id not in self._registry_data:
            raise ValueError(f"Factor '{factor_id}' not found")

        metadata_dict = self._registry_data[factor_id]['metadata']
        return FactorMetadata.from_dict(metadata_dict)

    def update_metadata(self, factor_id: str, updates: Dict):
        """更新因子元数据

        Args:
            factor_id: 因子ID
            updates: 要更新的字段

        Raises:
            ValueError: 如果因子不存在
        """
        if factor_id not in self._registry_data:
            raise ValueError(f"Factor '{factor_id}' not found")

        metadata = self.get_metadata(factor_id)

        # 更新字段
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        metadata.updated_time = datetime.now()

        # 保存更新
        self._registry_data[factor_id]['metadata'] = metadata.to_dict()
        self._save_registry()

    def list_factors(self,
                    category: str = None,
                    status: str = None,
                    frequency: str = None,
                    tags: List[str] = None) -> List[Dict]:
        """列出所有因子

        Args:
            category: 筛选因子类别
            status: 筛选因子状态
            frequency: 筛选支持特定频率的因子（'daily', 'minute'等）
            tags: 筛选包含指定标签的因子

        Returns:
            List[Dict]: 因子信息列表
        """
        results = []

        for factor_id, factor_data in self._registry_data.items():
            metadata = FactorMetadata.from_dict(factor_data['metadata'])

            # 应用筛选条件
            if category and metadata.category != category:
                continue

            if status and metadata.status != status:
                continue

            if frequency and frequency not in metadata.supported_frequencies:
                continue

            if tags:
                if not any(tag in metadata.tags for tag in tags):
                    continue

            # 构造结果
            factor_info = {
                'factor_id': factor_id,
                'name': metadata.name,
                'category': metadata.category,
                'status': metadata.status,
                'version': factor_data['current_version'],
                'supported_frequencies': metadata.supported_frequencies,
                'tags': metadata.tags,
                'description': metadata.description,
                'created_time': metadata.created_time,
                'updated_time': metadata.updated_time,
                'dependencies': metadata.dependencies
            }

            results.append(factor_info)

        # 按创建时间倒序排列
        results.sort(key=lambda x: x['created_time'], reverse=True)
        return results

    def search_factors(self,
                      keyword: str = None,
                      tags: List[str] = None) -> List[Dict]:
        """搜索因子

        Args:
            keyword: 关键词（在名称和描述中搜索）
            tags: 标签列表

        Returns:
            List[Dict]: 匹配的因子列表
        """
        all_factors = self.list_factors()
        results = []

        for factor in all_factors:
            match = True

            # 关键词搜索
            if keyword:
                keyword_lower = keyword.lower()
                name_match = keyword_lower in factor['name'].lower()
                desc_match = keyword_lower in factor['description'].lower()
                if not (name_match or desc_match):
                    match = False

            # 标签搜索
            if tags and match:
                if not any(tag in factor['tags'] for tag in tags):
                    match = False

            if match:
                results.append(factor)

        return results

    def delete_factor(self, factor_id: str):
        """删除因子

        Args:
            factor_id: 因子ID

        Raises:
            ValueError: 如果因子不存在
        """
        if factor_id not in self._registry_data:
            raise ValueError(f"Factor '{factor_id}' not found")

        del self._registry_data[factor_id]
        self._save_registry()
        print(f"Successfully deleted factor: {factor_id}")

    def get_factor_count(self) -> Dict[str, int]:
        """获取因子统计信息

        Returns:
            Dict[str, int]: 统计信息
        """
        stats = {
            'total': len(self._registry_data),
            'active': 0,
            'testing': 0,
            'deprecated': 0
        }

        for factor_data in self._registry_data.values():
            metadata = FactorMetadata.from_dict(factor_data['metadata'])
            if metadata.status in stats:
                stats[metadata.status] += 1

        return stats

    def export_registry(self, export_path: str):
        """导出注册表

        Args:
            export_path: 导出路径
        """
        save_json(self._registry_data, export_path)
        print(f"Registry exported to: {export_path}")

    def import_registry(self, import_path: str, merge: bool = True):
        """导入注册表

        Args:
            import_path: 导入路径
            merge: 是否与现有数据合并

        Raises:
            ValueError: 如果导入文件无效
        """
        try:
            imported_data = load_json(import_path)
        except Exception as e:
            raise ValueError(f"Failed to import registry from {import_path}: {e}")

        if merge:
            # 合并数据
            for factor_id, factor_data in imported_data.items():
                if factor_id in self._registry_data:
                    print(f"Warning: Factor '{factor_id}' already exists, skipping")
                else:
                    self._registry_data[factor_id] = factor_data
        else:
            # 替换数据
            self._registry_data = imported_data

        self._save_registry()
        print(f"Registry imported from: {import_path}")

    def __len__(self) -> int:
        """返回注册的因子数量"""
        return len(self._registry_data)

    def __contains__(self, factor_id: str) -> bool:
        """检查因子是否存在"""
        return factor_id in self._registry_data

    def __iter__(self):
        """迭代所有因子ID"""
        return iter(self._registry_data.keys())