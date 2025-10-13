"""
频率配置层实现

FrequencyConfig类为抽象因子表达式赋予具体的时间含义，包括：
- 输入数据的频率（分钟线还是日线）
- 计算窗口的长度和单位
- 计算的调度频率（多久计算一次）

这对应设计文档中的"中间层：频率配置层"
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class FrequencyConfig:
    """
    频率配置类

    按照设计文档中的规范，为抽象因子表达式赋予具体的时间含义。
    例如："20期平均"可以是"20日平均"或"20分钟平均"，取决于配置。

    属性:
        input_freq: 输入数据频率 ('daily', 'minute', '5min', '30min', 'hour' 等)
        window_length: 计算窗口长度（数值）
        calc_freq: 计算调度频率，决定多久计算一次因子值
        timezone: 时区设置，默认为中国时区
        market_hours: 交易时间设置
    """

    input_freq: str = 'daily'  # 输入数据频率
    window_length: int = 20    # 计算窗口长度
    calc_freq: str = 'daily'   # 计算调度频率
    timezone: str = 'Asia/Shanghai'  # 时区
    market_hours: Optional[Dict[str, Any]] = None  # 交易时间配置

    def __post_init__(self):
        """初始化后验证配置的合理性"""
        self._validate_config()
        self._set_default_market_hours()

    def _validate_config(self):
        """验证配置参数的合理性"""
        # 验证频率格式
        valid_freqs = ['daily', 'minute', '5min', '30min', 'hour', 'week', 'month']

        if self.input_freq not in valid_freqs:
            # 检查是否是pandas频率字符串格式
            try:
                pd.Timedelta(f"1{self.input_freq}")
            except ValueError:
                raise ValueError(f"不支持的输入频率: {self.input_freq}")

        if self.calc_freq not in valid_freqs:
            try:
                pd.Timedelta(f"1{self.calc_freq}")
            except ValueError:
                raise ValueError(f"不支持的计算频率: {self.calc_freq}")

        # 验证窗口长度
        if self.window_length <= 0:
            raise ValueError("窗口长度必须大于0")

        # 验证计算频率不能比输入频率更高频
        if not self._is_calc_freq_valid():
            raise ValueError(f"计算频率 {self.calc_freq} 不能比输入频率 {self.input_freq} 更高频")

    def _is_calc_freq_valid(self) -> bool:
        """检查计算频率是否合理（不能比输入频率更高频）"""
        freq_order = {
            'minute': 1,
            '5min': 5,
            '30min': 30,
            'hour': 60,
            'daily': 1440,  # 1440分钟 = 1天
            'week': 10080,  # 7*24*60分钟
            'month': 43200  # 30*24*60分钟（近似）
        }

        input_minutes = freq_order.get(self.input_freq, 1)
        calc_minutes = freq_order.get(self.calc_freq, 1)

        return calc_minutes >= input_minutes

    def _set_default_market_hours(self):
        """设置默认的交易时间"""
        if self.market_hours is None:
            # 中国A股交易时间
            self.market_hours = {
                'morning_start': '09:30:00',
                'morning_end': '11:30:00',
                'afternoon_start': '13:00:00',
                'afternoon_end': '15:00:00',
                'trading_days': 'weekdays'  # 工作日
            }

    def get_pandas_freq(self, freq_type: str) -> str:
        """
        将自定义频率转换为pandas频率字符串

        Args:
            freq_type: 'input' 或 'calc'，指定要转换的频率类型

        Returns:
            pandas频率字符串
        """
        freq = self.input_freq if freq_type == 'input' else self.calc_freq

        freq_mapping = {
            'minute': '1min',
            '5min': '5min',
            '30min': '30min',
            'hour': '1H',
            'daily': '1D',
            'week': '1W',
            'month': '1M'
        }

        return freq_mapping.get(freq, freq)

    def get_window_timedelta(self) -> pd.Timedelta:
        """
        获取窗口长度对应的时间差

        Returns:
            窗口长度对应的pd.Timedelta对象
        """
        pandas_freq = self.get_pandas_freq('input')
        return pd.Timedelta(f"{self.window_length}{pandas_freq}")

    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """
        判断给定时间戳是否在交易时间内

        Args:
            timestamp: 要检查的时间戳

        Returns:
            是否在交易时间内
        """
        # 检查是否是交易日（简化版，实际应该查交易日历）
        if timestamp.weekday() >= 5:  # 周六、周日
            return False

        time_str = timestamp.strftime('%H:%M:%S')

        # 检查是否在交易时间段内
        morning_start = self.market_hours['morning_start']
        morning_end = self.market_hours['morning_end']
        afternoon_start = self.market_hours['afternoon_start']
        afternoon_end = self.market_hours['afternoon_end']

        in_morning = morning_start <= time_str <= morning_end
        in_afternoon = afternoon_start <= time_str <= afternoon_end

        return in_morning or in_afternoon

    def __repr__(self) -> str:
        return (f"FrequencyConfig(input_freq='{self.input_freq}', "
                f"window_length={self.window_length}, "
                f"calc_freq='{self.calc_freq}')")


# 预定义的常用配置
class CommonConfigs:
    """常用的频率配置预设"""

    @staticmethod
    def daily_config(window_length: int = 20) -> FrequencyConfig:
        """日线配置"""
        return FrequencyConfig(
            input_freq='daily',
            window_length=window_length,
            calc_freq='daily'
        )

    @staticmethod
    def minute_config(window_length: int = 30, calc_freq: str = '5min') -> FrequencyConfig:
        """分钟线配置"""
        return FrequencyConfig(
            input_freq='minute',
            window_length=window_length,
            calc_freq=calc_freq
        )

    @staticmethod
    def high_freq_config(window_length: int = 60) -> FrequencyConfig:
        """高频配置（分钟级输入，每分钟计算）"""
        return FrequencyConfig(
            input_freq='minute',
            window_length=window_length,
            calc_freq='minute'
        )

    @staticmethod
    def intraday_config(window_length: int = 10) -> FrequencyConfig:
        """日内配置（30分钟线）"""
        return FrequencyConfig(
            input_freq='30min',
            window_length=window_length,
            calc_freq='30min'
        )