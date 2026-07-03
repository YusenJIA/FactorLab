# -*- coding: utf-8 -*-
"""
行为金融因子计算引擎 - 基类和配置
================================

参考 Qlib 的 DataHandler 和 Processor 设计模式
"""

import abc
import os
import logging
import pickle
from typing import Union, List, Optional, Dict, Any, Text, Set
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import polars as pl

# ============================================================================
# 配置
# ============================================================================

class Config:
    """全局配置"""
    PROJECT_ROOT = os.environ.get("ASHARE_STOCK_PROJECT_ROOT", "/home/yusen/ashare_stock_proj")
    LOG_DIR = os.environ.get("ASHARE_LOG_DIR", os.path.join(PROJECT_ROOT, "data/logs"))
    INPUT_DIR = os.environ.get("ASHARE_INPUT_DIR", os.path.join(PROJECT_ROOT, "price1m_data"))
    OUTPUT_DIR = os.environ.get("ASHARE_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "data/features"))
    
    # 确保目录存在
    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.INPUT_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


Config.ensure_dirs()

# ============================================================================
# 常量
# ============================================================================

EPS = 1e-3

# 时间周期常量
"""PERIODS_SHORT = [1, 4, 16, 48, 96]
PERIODS_MEDIUM = [12, 24, 48, 72, 96]
PERIODS_LONG = [24, 48, 72, 168, 336]
PERIODS_EXTENDED = [16, 32, 48, 96, 288]"""

PERIODS = [5,15,60,241]
TEMP_FEATURES = ['_close_ret', '_volume_ret', '_ret_rolling_mean', '_ret_rolling_std', '_ret_rolling_var', '_volume_rolling_mean',
                 '_volume_rolling_std', '_close_rolling_mean', '_close_rolling_std', '_close_rolling_max', '_close_rolling_min', '_high_rolling_max',
                 '_low_rolling_min', '_volatility', '_rsi_gain', '_rsi_loss', '_rsi', '_bollinger_pos'  
                ]

PRECOMPUTED_FEATURES = [f"{feature}_{period}" for feature in TEMP_FEATURES for period in PERIODS]
PRECOMPUTED_FEATURES += ['_close_ret_1'] 

MIXED_FEATURE_BASES = {
    'round_number': [
        "near_round_1_time",
        "round_1_bounce",
        "round_1_volume_cluster",
        "ath_5min_distance",
    ],
    'fomo_fud': [
        "return_kurtosis_proxy",
        "rally_volume",
        "v_reversal_5min",
        "chase_momentum_lag1",
        "sentiment_shift_5min",
        "volume_acceleration_5min",
        "volatility_regime_5min",
        "bollinger_position_5min",
        "fear_index_5min",
        "momentum_divergence_5min",
    ],
    'retail_pattern': [
        "retail_capitulation",
        "buy_high_pattern_5min",
        "buy_dip_pattern_5min",
        "stop_loss_2pct",
        "range_trading_5min",
        "breakout_trading_5min",
        "retail_panic_5min",
        "momentum_chasing_lag1",
        "price_anchoring_5min",
        "mean_reversion_5min",
        "news_driven_5min",
        "ta_dependency_5min",
        "retail_fatigue_5min",
        "money_flow_5min",
        "volatility_preference_241min",
    ],
    'herding': [
        "leader_effectiveness",
        "info_shock_absorption",
        "info_processing_speed",
        "info_asymmetry_proxy",
        "price_clustering_5min",
        "volatility_contagion_lag1",
        "social_learning_5min",
        "network_effect_5min",
        "social_contagion_lag1",
    ],
    'microstructure': [
        "kyle_lambda",
        "roll_effective_spread",
        "amihud_illiquidity",
        "price_efficiency_var",
        "volatility_ratio",
        "depth_adjusted_spread",
        "price_efficiency_5min",
        "price_impact_5min",
        "execution_risk_5min",
        "order_flow_toxicity_5min",
        "info_content_5min",
        "micro_volatility_5min",
        "order_flow_toxicity_15min",
    ],
    'sentiment_cycle': [
        "anchoring_bias",
        "bull_bear_position_5min",
        "sentiment_momentum_5min",
        "fear_greed_5min",
        "cumulative_sentiment_95",
        "sentiment_dispersion_5min",
        "sentiment_oscillation_15min",
        "sentiment_cyclical_5min",
        "sentiment_memory_5min",
        "risk_appetite_5min",
    ],
    'attention': [
        "attention_inflow",
        "net_attention_flow",
        "viral_attention",
        "price_memory_lag5h",
        "attention_shift_15min",
        "focal_moments_15min",
        "attention_focus_5min",
        "surprise_factor_5min",
        "attention_intensity_5min",
        "attention_momentum_5min",
        "attention_dispersion_5min",
        "anticipated_attention_5min",
        "attention_contagion_5min",
    ],
}

# ============================================================================
# 日志工具
# ============================================================================

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """获取配置好的 logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(
            os.path.join(Config.LOG_DIR, log_file),
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# 辅助函数
# ============================================================================

def calculate_rsi(price_col: str, period: int) -> pl.Expr:
    """计算 RSI 指标"""
    price_diff = pl.col(price_col).diff()
    gain = pl.when(price_diff > 0).then(price_diff).otherwise(0)
    loss = pl.when(price_diff < 0).then(-price_diff).otherwise(0)
    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)
    rs = avg_gain / (avg_loss + EPS)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_position(price_col: str, period: int, std_dev: float = 2) -> pl.Expr:
    """计算布林带位置"""
    sma = pl.col(price_col).rolling_mean(window_size=period)
    std = pl.col(price_col).rolling_std(window_size=period)
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    position = (pl.col(price_col) - lower_band) / (upper_band - lower_band + EPS)
    return position.clip(0, 1)


# ============================================================================
# Processor 基类
# ============================================================================

class Processor(abc.ABC):
    """
    因子处理器基类
    
    参考 Qlib 的 Processor 设计，支持：
    - fit: 学习数据处理参数
    - __call__: 执行数据处理
    - is_for_infer: 是否可用于推理
    - readonly: 是否只读处理
    """
    
    def __init__(self, fields_group: Optional[str] = None):
        self.fields_group = fields_group
        self.logger = get_logger(self.__class__.__name__)
        self._fitted = False
    
    def fit(self, df: pl.DataFrame = None) -> "Processor":
        """学习数据处理参数"""
        self._fitted = True
        return self
    
    @abc.abstractmethod
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """处理数据"""
        pass
    
    def is_for_infer(self) -> bool:
        """是否可用于推理"""
        return True
    
    def readonly(self) -> bool:
        """是否只读处理"""
        return False
    
    def get_feature_names(self) -> List[str]:
        """获取特征名列表"""
        return []
    
    def config(self, **kwargs):
        """配置处理器参数"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def save(self, path: str):
        """保存处理器状态"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "Processor":
        """加载处理器状态"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# 数据处理器（标准化等）
# ============================================================================

class DropnaProcessor(Processor):
    """删除缺失值处理器"""
    
    def __init__(self, fields_group: Optional[str] = None, subset: Optional[List[str]] = None):
        super().__init__(fields_group)
        self.subset = subset
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.subset:
            return df.drop_nulls(subset=self.subset)
        return df.drop_nulls()
    
    def readonly(self) -> bool:
        return True


class FillnaProcessor(Processor):
    """填充缺失值处理器"""
    
    def __init__(self, fill_value: float = 0.0, fields_group: Optional[str] = None):
        super().__init__(fields_group)
        self.fill_value = fill_value
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.fill_null(self.fill_value)


class CSZScoreNormProcessor(Processor):
    """截面 Z-Score 标准化处理器"""
    
    def __init__(
        self, 
        fields_group: Optional[str] = None,
        group_col: str = 'datetime',
        exclude_cols: Optional[List[str]] = None
    ):
        super().__init__(fields_group)
        self.group_col = group_col
        self.exclude_cols = exclude_cols or ['datetime', 'code']
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in df.columns 
                       if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                       and c not in self.exclude_cols]
        
        expressions = []
        for col in numeric_cols:
            expr = (
                (pl.col(col) - pl.col(col).mean().over(self.group_col)) / 
                (pl.col(col).std().over(self.group_col) + 1e-8)
            ).alias(col)
            expressions.append(expr)
        
        return df.with_columns(expressions)


class CSRankNormProcessor(Processor):
    """截面排名标准化处理器"""
    
    def __init__(
        self, 
        fields_group: Optional[str] = None,
        group_col: str = 'datetime',
        exclude_cols: Optional[List[str]] = None
    ):
        super().__init__(fields_group)
        self.group_col = group_col
        self.exclude_cols = exclude_cols or ['symbol', 'datetime']
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in df.columns 
                       if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                       and c not in self.exclude_cols]
        
        expressions = []
        for col in numeric_cols:
            # 使用 rank 并转换为 0-1 范围，然后标准化
            expr = (
                pl.col(col).rank().over(self.group_col) / 
                pl.col(col).count().over(self.group_col) - 0.5
            ) * 3.46  # 标准化到接近标准正态分布
            expressions.append(expr.alias(col))
        
        return df.with_columns(expressions)


class CSMinMaxNormProcessor(Processor):
    """截面 Min-Max 标准化处理器"""
    
    def __init__(
        self, 
        fields_group: Optional[str] = None,
        group_col: str = 'datetime',
        exclude_cols: Optional[List[str]] = None
    ):
        super().__init__(fields_group)
        self.group_col = group_col
        self.exclude_cols = exclude_cols or ['datetime', 'code', 'date']
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in df.columns 
                       if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                       and c not in self.exclude_cols]
        
        expressions = []
        for col in numeric_cols:
            col_min = pl.col(col).min().over(self.group_col)
            col_max = pl.col(col).max().over(self.group_col)
            expr = (
                (pl.col(col) - col_min) / (col_max - col_min + 1e-8)
            ).alias(col)
            expressions.append(expr)
        
        return df.with_columns(expressions)


class MixedFeatureSplitter(Processor):
    """
    混合类型特征拆分处理器
    将指定特征拆分为：
    - {col}_ctn: 连续部分（非零值保留，零值设为 null）
    - {col}_dummy: 哑变量部分（布尔类型，非零为True，零为False）
    
    支持模糊匹配：去除含数字的部分后进行匹配
    例如: "attention_intensity_5min" -> ["attention", "intensity"] 
         与 "attention_intensity" -> ["attention", "intensity"] 匹配
    """
    
    def __init__(
        self,
        processor_name: Optional[str] = None,
        fields_group: Optional[str] = 'datetime',
    ):
        super().__init__(fields_group)
        self.mixed_col_bases = MIXED_FEATURE_BASES[processor_name]
        self._fitted = True
        # 预处理：提取基础名称的非数字部分
        self._base_patterns = self._build_base_patterns()
    
    @staticmethod
    def _extract_non_numeric_parts(name: str) -> tuple:
        """提取名称中不含数字的部分，返回元组用于匹配"""
        parts = name.split('_')
        non_numeric = [p for p in parts if not re.search(r'\d', p)]
        return tuple(non_numeric)
    
    def _build_base_patterns(self) -> Set[tuple]:
        """构建基础特征的匹配模式"""
        patterns = set()
        for base in self.mixed_col_bases:
            pattern = self._extract_non_numeric_parts(base)
            patterns.add(pattern)
        return patterns
    
    def _match_mixed_cols(self, columns: List[str]) -> List[str]:
        """根据非数字部分匹配实际列名"""
        matched = []
        for col in columns:
            pattern = self._extract_non_numeric_parts(col)
            if pattern in self._base_patterns:
                matched.append(col)
        return matched
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        # 自动匹配混合特征列
        mixed_cols = self._match_mixed_cols(df.columns)
        
        if not mixed_cols:
            self.logger.info("No mixed features found to split")
            return df
        
        self.logger.info(f"Found {len(mixed_cols)} mixed features to split")
        
        expressions = []
        cols_to_drop = []
        
        for col in mixed_cols:
            # 连续部分：非零值保留，零值设为 null
            ctn_expr = (
                pl.when(pl.col(col) != 0)
                .then(pl.col(col))
                .otherwise(None)
                .alias(f"{col}_ctn")
            )
            
            # 哑变量部分：布尔类型
            dummy_expr = (
                (pl.col(col) != 0).alias(f"{col}_dummy")
            )
            
            expressions.extend([ctn_expr, dummy_expr])
            cols_to_drop.append(col)
        
        # 添加新列并删除原列
        df = df.with_columns(expressions)
        df = df.drop(cols_to_drop)
        
        self.logger.info(f"Split {len(cols_to_drop)} mixed features into _ctn and _dummy")
        return df


class NonlinearTransformer(Processor):
    """
    非线性变换处理器
    生成三种变换：
    - {col}_sqrt: (x - x.min) ^ 0.5
    - {col}_cubic: x ^ 3
    - {col}_exp: e ^ x (需先标准化避免溢出)
    """
    
    def __init__(
        self,
        fields_group: Optional[str] = None,
        group_col: str = 'datetime',
        exclude_cols: Optional[List[str]] = None,
        transforms: Optional[List[str]] = None,  # ['sqrt', 'cubic', 'exp']
    ):
        super().__init__(fields_group)
        self.group_col = group_col
        self.exclude_cols = exclude_cols or ['datetime', 'code']
        self.transforms = transforms or ['sqrt', 'cubic', 'exp']
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in df.columns 
                       if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                       and c not in self.exclude_cols
                       and not c.endswith('_dummy')]  # 不对哑变量做变换
        
        expressions = []
        
        for col in numeric_cols:
            if 'sqrt' in self.transforms:
                # sqrt 变换: (x - x.min) ^ 0.5
                col_min = pl.col(col).min().over(self.group_col)
                sqrt_expr = (
                    (pl.col(col) - col_min).sqrt().alias(f"{col}_sqrt")
                )
                expressions.append(sqrt_expr)
            
            if 'cubic' in self.transforms:
                # cubic 变换: x ^ 3 (先做截面标准化避免数值过大)
                col_mean = pl.col(col).mean().over(self.group_col)
                col_std = pl.col(col).std().over(self.group_col)
                normalized = (pl.col(col) - col_mean) / (col_std + 1e-8)
                cubic_expr = (
                    normalized.pow(3).alias(f"{col}_cubic")
                )
                expressions.append(cubic_expr)
            
            if 'exp' in self.transforms:
                # exp 变换: e ^ x (先做截面标准化并 clip 避免溢出)
                col_mean = pl.col(col).mean().over(self.group_col)
                col_std = pl.col(col).std().over(self.group_col)
                normalized = (pl.col(col) - col_mean) / (col_std + 1e-8)
                clipped = pl.when(normalized > 5).then(5).when(normalized < -5).then(-5).otherwise(normalized)
                exp_expr = (
                    clipped.exp().alias(f"{col}_exp")
                )
                expressions.append(exp_expr)
        
        self.logger.info(f"Generated {len(expressions)} nonlinear transforms for {len(numeric_cols)} columns")
        return df.with_columns(expressions)
