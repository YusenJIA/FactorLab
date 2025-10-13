"""
因子库管理系统

基于统一因子计算框架的管理层扩展，提供：
- 因子注册和检索
- 版本控制
- 性能监控
- 组合管理

Example:
    >>> from factor_library import FactorLibrary, FactorMetadata
    >>> from factor_framework import Close, Mean, Volume, FrequencyConfig

    >>> # 初始化因子库
    >>> library = FactorLibrary()

    >>> # 定义因子
    >>> factor = Close() / Mean(Volume(), window_length=20)
    >>> configs = [FrequencyConfig(input_freq='daily', window_length=20)]

    >>> # 注册因子
    >>> factor_id = library.register_factor(
    ...     factor_node=factor,
    ...     factor_configs=configs,
    ...     metadata={'name': '价格成交量比', 'category': 'liquidity'}
    ... )
"""

from .metadata import FactorMetadata
from .registry import FactorRegistry
from .version_control import FactorVersionControl
from .monitor import FactorMonitor
from .groups import FactorGroup
from .utils import serialize_factor_node, deserialize_factor_node

class FactorLibrary:
    """因子库管理系统主入口"""

    def __init__(self, registry_path='./data/factor_registry/factors.json'):
        """初始化因子库

        Args:
            registry_path: 因子注册表存储路径
        """
        self.registry = FactorRegistry(registry_path)
        self.version_control = FactorVersionControl(self.registry)
        self.monitor = FactorMonitor(self.registry)
        self.groups = FactorGroup(self.registry)

    def register_factor(self, factor_node, factor_configs, metadata):
        """注册新因子"""
        return self.registry.register_factor(factor_node, factor_configs, metadata)

    def get_factor(self, factor_id, version=None):
        """获取因子定义"""
        return self.registry.get_factor(factor_id, version)

    def list_factors(self, **filters):
        """列出因子"""
        return self.registry.list_factors(**filters)

    def evaluate_factor(self, factor_id, assets, start_date, end_date, returns_data=None):
        """评估因子性能"""
        return self.monitor.evaluate_factor(factor_id, assets, start_date, end_date, returns_data)

__all__ = [
    'FactorLibrary',
    'FactorMetadata',
    'FactorRegistry',
    'FactorVersionControl',
    'FactorMonitor',
    'FactorGroup'
]