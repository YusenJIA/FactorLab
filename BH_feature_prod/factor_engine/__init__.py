# -*- coding: utf-8 -*-
"""
行为金融因子计算引擎
==================

一个参考 Qlib 设计的行为金融因子计算框架。

主要组件：
- Processor: 因子处理器基类
- FeatureDataHandler: 数据处理器（类似 Qlib DataHandlerLP）
- FeaturePipeline: 因子计算流水线

因子处理器：
- RoundNumberProcessor: 整数关口因子
- FOMOFUDProcessor: FOMO/FUD 情绪因子
- RetailPatternProcessor: 散户交易模式因子
- HerdingProcessor: 羊群效应因子
- MicrostructureProcessor: 市场微观结构因子
- SentimentCycleProcessor: 情绪周期因子
- AttentionProcessor: 市场注意力因子

标准化处理器：
- CSZScoreNormProcessor: 截面 Z-Score 标准化
- CSRankNormProcessor: 截面排名标准化
- CSMinMaxNormProcessor: 截面 Min-Max 标准化

Usage:
    from behavior_engine import FeaturePipeline, FeatureDataHandler
    
    # 方式1: 使用流水线
    pipeline = FeaturePipeline(
        input_dir="/data/raw",
        output_dir="/data/features",
    )
    df = pipeline.run(start_date="2024-01-01", end_date="2024-12-31")
    
    # 方式2: 使用数据处理器
    from behavior_engine import create_default_handler
    
    handler = create_default_handler(
        fit_start_time="2020-01-01",
        fit_end_time="2022-12-31",
        norm_method="zscore"
    )
    handler.fit(train_df)
    features = handler.fetch(test_df)
"""

__version__ = "1.0.0"
__author__ = "yusen"

# 配置
from .base import Config, EPS

# 基类
from .base import (
    Processor,
    get_logger,
    calculate_rsi,
    calculate_bollinger_position,
)

# 标准化处理器
from .base import (
    DropnaProcessor,
    FillnaProcessor,
    CSZScoreNormProcessor,
    CSMinMaxNormProcessor,
    CSRankNormProcessor
)

# 行为金融因子处理器
from .behavior_features import (
    RoundNumberProcessor, FOMOFUDProcessor,
    RetailPatternProcessor, HerdingProcessor,
    MicrostructureProcessor, SentimentCycleProcessor, AttentionProcessor
    )

# 流水线和数据处理器
from .pipeline import (
    FeatureDataHandler,
    FeaturePipeline,
    create_default_handler,
)

# 导出列表
__all__ = [
    # 配置
    "Config",
    "EPS",
    
    # 基类和工具
    "Processor",
    "get_logger",
    "calculate_rsi",
    "calculate_bollinger_position",
    
    # 标准化处理器
    "DropnaProcessor",
    "FillnaProcessor",
    "CSZScoreNormProcessor",
    "CSRankNormProcessor",
    "CSMinMaxNormProcessor",
    
    # 行为金融因子处理器
    "RoundNumberProcessor",
    "FOMOFUDProcessor",
    "RetailPatternProcessor",
    "HerdingProcessor",
    "MicrostructureProcessor",
    "SentimentCycleProcessor",
    "AttentionProcessor",
    
    # 流水线和数据处理器
    "FeatureDataHandler",
    "FeaturePipeline",
    "create_default_handler",
]
