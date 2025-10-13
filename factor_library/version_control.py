"""
因子版本控制模块

跟踪 FactorNode 定义的演进历史，支持版本比较和回滚。
"""

import re
from typing import Dict, List, Optional
from datetime import datetime

from factor_framework.nodes.base import FactorNode
from factor_framework.config import FrequencyConfig

from .registry import FactorRegistry
from .metadata import FactorMetadata
from .utils import (
    serialize_factor_node, deserialize_factor_node,
    serialize_frequency_configs, deserialize_frequency_configs
)

class FactorVersionControl:
    """因子版本控制 - 跟踪 FactorNode 定义的演进历史"""

    def __init__(self, registry: FactorRegistry):
        """初始化版本控制

        Args:
            registry: 因子注册中心
        """
        self.registry = registry

    def create_version(self,
                      factor_id: str,
                      factor_node: FactorNode,
                      factor_configs: List[FrequencyConfig],
                      changes: str,
                      version_type: str = 'minor') -> str:
        """创建新版本

        Args:
            factor_id: 因子ID
            factor_node: 新的因子节点
            factor_configs: 新的频率配置
            changes: 变更说明
            version_type: 版本类型 ('major', 'minor', 'patch')

        Returns:
            str: 新版本号

        Raises:
            ValueError: 如果因子不存在或版本类型无效
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        if version_type not in ['major', 'minor', 'patch']:
            raise ValueError(f"Invalid version type: {version_type}")

        # 获取因子数据
        factor_data = self.registry._registry_data[factor_id]
        current_version = factor_data['current_version']

        # 生成新版本号
        new_version = self._increment_version(current_version, version_type)

        # 序列化新的因子定义
        serialized_node = serialize_factor_node(factor_node)
        serialized_configs = serialize_frequency_configs(factor_configs)

        # 创建新版本记录
        version_record = {
            'factor_node': serialized_node,
            'factor_configs': serialized_configs,
            'created_at': datetime.now().isoformat(),
            'changes': changes,
            'version_type': version_type,
            'is_active': True
        }

        # 添加版本记录
        factor_data['versions'][new_version] = version_record

        # 将之前的版本标记为非活跃
        if current_version in factor_data['versions']:
            factor_data['versions'][current_version]['is_active'] = False

        # 更新当前版本
        factor_data['current_version'] = new_version

        # 更新元数据
        metadata = self.registry.get_metadata(factor_id)
        metadata.version = new_version
        metadata.updated_time = datetime.now()
        metadata.changelog = changes

        # 更新因子表达式和依赖信息
        from .utils import extract_factor_expression, extract_dependencies
        metadata.factor_expression = extract_factor_expression(factor_node)
        metadata.dependencies = extract_dependencies(factor_node)
        metadata.supported_frequencies = [config.input_freq for config in factor_configs]

        factor_data['metadata'] = metadata.to_dict()

        # 保存更改
        self.registry._save_registry()

        print(f"Created version {new_version} for factor {factor_id}")
        return new_version

    def get_version_history(self, factor_id: str) -> List[Dict]:
        """获取版本历史

        Args:
            factor_id: 因子ID

        Returns:
            List[Dict]: 版本历史列表

        Raises:
            ValueError: 如果因子不存在
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        factor_data = self.registry._registry_data[factor_id]
        versions = factor_data['versions']

        history = []
        for version, version_data in versions.items():
            history.append({
                'version': version,
                'created_at': version_data['created_at'],
                'changes': version_data.get('changes', ''),
                'version_type': version_data.get('version_type', 'unknown'),
                'is_active': version_data.get('is_active', False)
            })

        # 按版本号排序（最新的在前）
        history.sort(key=lambda x: self._parse_version(x['version']), reverse=True)
        return history

    def compare_versions(self, factor_id: str, v1: str, v2: str) -> Dict:
        """比较两个版本的差异

        Args:
            factor_id: 因子ID
            v1: 版本1
            v2: 版本2

        Returns:
            Dict: 比较结果
            {
                'node_changed': bool,
                'configs_changed': bool,
                'metadata_changed': Dict,
                'performance_diff': Dict
            }

        Raises:
            ValueError: 如果因子或版本不存在
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        factor_data = self.registry._registry_data[factor_id]
        versions = factor_data['versions']

        if v1 not in versions:
            raise ValueError(f"Version '{v1}' not found")
        if v2 not in versions:
            raise ValueError(f"Version '{v2}' not found")

        version1_data = versions[v1]
        version2_data = versions[v2]

        # 比较因子节点
        node_changed = version1_data['factor_node'] != version2_data['factor_node']

        # 比较配置
        configs_changed = version1_data['factor_configs'] != version2_data['factor_configs']

        # 比较元数据（基本信息）
        metadata_diff = {
            'v1_changes': version1_data.get('changes', ''),
            'v2_changes': version2_data.get('changes', ''),
            'v1_created': version1_data['created_at'],
            'v2_created': version2_data['created_at']
        }

        # 性能差异（如果有缓存的性能数据）
        performance_diff = self._compare_performance(factor_id, v1, v2)

        return {
            'node_changed': node_changed,
            'configs_changed': configs_changed,
            'metadata_changed': metadata_diff,
            'performance_diff': performance_diff,
            'summary': self._generate_diff_summary(node_changed, configs_changed, v1, v2)
        }

    def rollback(self, factor_id: str, target_version: str):
        """回滚到指定版本

        Args:
            factor_id: 因子ID
            target_version: 目标版本

        Raises:
            ValueError: 如果因子或版本不存在
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        factor_data = self.registry._registry_data[factor_id]
        versions = factor_data['versions']

        if target_version not in versions:
            raise ValueError(f"Version '{target_version}' not found")

        current_version = factor_data['current_version']

        # 如果已经是目标版本，无需回滚
        if current_version == target_version:
            print(f"Factor {factor_id} is already at version {target_version}")
            return

        # 将当前版本标记为非活跃
        if current_version in versions:
            versions[current_version]['is_active'] = False

        # 激活目标版本
        versions[target_version]['is_active'] = True
        factor_data['current_version'] = target_version

        # 更新元数据
        metadata = self.registry.get_metadata(factor_id)
        metadata.version = target_version
        metadata.updated_time = datetime.now()
        metadata.changelog = f"Rolled back to version {target_version}"

        factor_data['metadata'] = metadata.to_dict()

        # 保存更改
        self.registry._save_registry()

        print(f"Rolled back factor {factor_id} to version {target_version}")

    def get_current_version(self, factor_id: str) -> str:
        """获取当前版本号

        Args:
            factor_id: 因子ID

        Returns:
            str: 当前版本号

        Raises:
            ValueError: 如果因子不存在
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        return self.registry._registry_data[factor_id]['current_version']

    def list_versions(self, factor_id: str) -> List[str]:
        """列出所有版本

        Args:
            factor_id: 因子ID

        Returns:
            List[str]: 版本号列表

        Raises:
            ValueError: 如果因子不存在
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        versions = list(self.registry._registry_data[factor_id]['versions'].keys())
        # 按版本号排序
        versions.sort(key=self._parse_version, reverse=True)
        return versions

    def delete_version(self, factor_id: str, version: str):
        """删除指定版本

        Args:
            factor_id: 因子ID
            version: 版本号

        Raises:
            ValueError: 如果因子不存在、版本不存在或尝试删除当前版本
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        factor_data = self.registry._registry_data[factor_id]
        versions = factor_data['versions']

        if version not in versions:
            raise ValueError(f"Version '{version}' not found")

        if version == factor_data['current_version']:
            raise ValueError(f"Cannot delete current version '{version}'")

        # 删除版本
        del versions[version]

        # 保存更改
        self.registry._save_registry()

        print(f"Deleted version {version} of factor {factor_id}")

    def _increment_version(self, current_version: str, version_type: str) -> str:
        """递增版本号

        Args:
            current_version: 当前版本号 (e.g., "v1.2.3")
            version_type: 版本类型 ('major', 'minor', 'patch')

        Returns:
            str: 新版本号
        """
        # 解析版本号
        major, minor, patch = self._parse_version(current_version)

        # 根据类型递增
        if version_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif version_type == 'minor':
            minor += 1
            patch = 0
        elif version_type == 'patch':
            patch += 1

        return f"v{major}.{minor}.{patch}"

    def _parse_version(self, version: str) -> tuple:
        """解析版本号

        Args:
            version: 版本号字符串 (e.g., "v1.2.3")

        Returns:
            tuple: (major, minor, patch)
        """
        # 移除 'v' 前缀
        version_clean = version.lstrip('v')

        # 解析数字
        parts = version_clean.split('.')
        if len(parts) != 3:
            # 如果格式不正确，返回默认值
            return (1, 0, 0)

        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
            return (major, minor, patch)
        except ValueError:
            return (1, 0, 0)

    def _compare_performance(self, factor_id: str, v1: str, v2: str) -> Dict:
        """比较两个版本的性能差异

        Args:
            factor_id: 因子ID
            v1: 版本1
            v2: 版本2

        Returns:
            Dict: 性能差异
        """
        # 这里是简化实现，实际可以从缓存中读取历史性能数据
        return {
            'performance_comparison': 'Not implemented',
            'note': 'Performance comparison requires historical performance data'
        }

    def _generate_diff_summary(self, node_changed: bool, configs_changed: bool, v1: str, v2: str) -> str:
        """生成差异摘要

        Args:
            node_changed: 因子节点是否变化
            configs_changed: 配置是否变化
            v1: 版本1
            v2: 版本2

        Returns:
            str: 差异摘要
        """
        changes = []

        if node_changed:
            changes.append("factor definition")

        if configs_changed:
            changes.append("frequency configurations")

        if not changes:
            return f"No significant changes between {v1} and {v2}"

        return f"Changes between {v1} and {v2}: {', '.join(changes)}"

    def get_version_info(self, factor_id: str, version: str = None) -> Dict:
        """获取版本详细信息

        Args:
            factor_id: 因子ID
            version: 版本号，如果不指定则返回当前版本

        Returns:
            Dict: 版本信息

        Raises:
            ValueError: 如果因子或版本不存在
        """
        if factor_id not in self.registry:
            raise ValueError(f"Factor '{factor_id}' not found")

        factor_data = self.registry._registry_data[factor_id]

        if version is None:
            version = factor_data['current_version']

        if version not in factor_data['versions']:
            raise ValueError(f"Version '{version}' not found")

        version_data = factor_data['versions'][version]

        return {
            'factor_id': factor_id,
            'version': version,
            'is_current': version == factor_data['current_version'],
            'is_active': version_data.get('is_active', False),
            'created_at': version_data['created_at'],
            'changes': version_data.get('changes', ''),
            'version_type': version_data.get('version_type', 'unknown'),
            'has_factor_node': 'factor_node' in version_data,
            'has_configs': 'factor_configs' in version_data
        }