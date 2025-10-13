"""
因子元数据定义

定义因子的元数据结构，包括基本信息、版本信息、性能信息等。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

@dataclass
class FactorMetadata:
    """因子元数据定义"""

    # 基本信息
    factor_id: str                      # 因子唯一标识
    name: str                          # 因子名称
    description: str                   # 因子描述
    category: str                      # 因子类别 (liquidity, momentum, value, etc.)
    tags: List[str] = field(default_factory=list)  # 标签
    author: str = "unknown"            # 创建者

    # 版本信息
    version: str = "v1.0.0"           # 版本号 (v1.0.0)
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    status: str = "active"            # active/testing/deprecated
    changelog: str = ""               # 变更日志

    # 计算框架相关信息
    factor_expression: str = ""       # 因子表达式的字符串表示
    supported_frequencies: List[str] = field(default_factory=list)  # 支持的频率
    dependencies: List[str] = field(default_factory=list)  # 依赖的原子数据

    # 性能元数据
    last_calc_time: Optional[datetime] = None
    performance_by_freq: Optional[Dict[str, Dict]] = None  # 不同频率下的性能指标
    used_in_strategies: Optional[List[str]] = None
    performance_notes: str = ""

    def to_dict(self) -> Dict:
        """转换为字典格式，用于JSON序列化"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif value is None:
                result[key] = None
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'FactorMetadata':
        """从字典创建FactorMetadata实例"""
        # 处理datetime字段
        datetime_fields = ['created_time', 'updated_time', 'last_calc_time']
        for field_name in datetime_fields:
            if field_name in data and data[field_name] is not None:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])

        # 处理可能为None的字段
        if 'tags' not in data:
            data['tags'] = []
        if 'supported_frequencies' not in data:
            data['supported_frequencies'] = []
        if 'dependencies' not in data:
            data['dependencies'] = []

        return cls(**data)

    def update_performance(self, frequency: str, metrics: Dict):
        """更新特定频率下的性能指标"""
        if self.performance_by_freq is None:
            self.performance_by_freq = {}

        self.performance_by_freq[frequency] = {
            **metrics,
            'last_update': datetime.now().isoformat()
        }
        self.last_calc_time = datetime.now()

    def get_performance(self, frequency: str = None) -> Dict:
        """获取性能指标"""
        if self.performance_by_freq is None:
            return {}

        if frequency:
            return self.performance_by_freq.get(frequency, {})
        else:
            return self.performance_by_freq

    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str):
        """移除标签"""
        if tag in self.tags:
            self.tags.remove(tag)

    def is_active(self) -> bool:
        """检查因子是否处于活跃状态"""
        return self.status == "active"

    def __str__(self) -> str:
        return f"FactorMetadata(id={self.factor_id}, name={self.name}, version={self.version})"

    def __repr__(self) -> str:
        return self.__str__()