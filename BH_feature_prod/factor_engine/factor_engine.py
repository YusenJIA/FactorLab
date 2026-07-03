# -*- coding: utf-8 -*-
"""
基础特征预计算处理器
====================

预计算所有高频复用的中间变量，供其他处理器直接引用。

主要预计算的8类高频复用模式：
1. 价格变化率: pl.col('close').pct_change() 和 pl.col('close').pct_change(period)
2. 成交量变化率: pl.col('volume').pct_change() 和 pl.col('volume').pct_change(period)
3. 价格变化率的滚动统计: rolling_mean, rolling_std, rolling_var, rolling_skew
4. 成交量的滚动统计: rolling_mean, rolling_std
5. 价格的滚动统计: rolling_mean, rolling_std, rolling_max, rolling_min
6. 波动率: pl.col('close').pct_change().rolling_std()
7. 成交量归一化: pl.col('volume') / pl.col('volume').rolling_mean()
8. 价格方向: (pl.col('close').pct_change() > 0).cast(pl.Float32)
"""

import numpy as np
import polars as pl
from typing import List, Optional
import gc

from .base import (
    Processor, EPS, get_logger,
    PERIODS
)

from .behavior_features import (
    RoundNumberProcessor, FOMOFUDProcessor,
    RetailPatternProcessor, HerdingProcessor,
    MicrostructureProcessor, SentimentCycleProcessor, AttentionProcessor
    )

class BaseFeatureProcessor(Processor):
    
    def __init__(self, processors: Optional[List[Processor]] = None):
        """
        初始化处理器
        
        Args:
            processors: 可选的处理器列表，将在预计算后依次调用
        """
        super().__init__()
        
        # 默认周期配置
        self.periods = PERIODS
        
        # 处理器链
        self.processors = processors if processors is not None else self._get_default_processors()
        
        """if self.processors:
            self.logger.info(f"Initialized with {len(self.processors)} additional processors")"""
    
    def _get_default_processors(self) -> List[Processor]:
        """
        获取默认处理器列表
        """
            
        return [
                RoundNumberProcessor(),
                FOMOFUDProcessor(),
                RetailPatternProcessor(),
                HerdingProcessor(),
                MicrostructureProcessor(),
                SentimentCycleProcessor(),
                AttentionProcessor(),
        ]
    
    def add_processor(self, processor: Processor):
        """
        添加处理器到链中
        
        Args:
            processor: 要添加的处理器
        """
        self.processors.append(processor)
        self.logger.info(f"Added {processor.__class__.__name__} to processor chain")
    
    def fit(self, df: pl.DataFrame = None) -> "BaseFeatureProcessor":
        """
        可选的 fit 方法，用于根据数据自适应调整周期
        
        如果不传入 df，则使用默认配置；
        如果传入 df，则根据数据统计信息自适应调整周期配置。
        同时会 fit 所有链式处理器。
        
        Args:
            df: 训练数据（可选）。如果为 None，使用默认配置
            
        Returns:
            self
            
        示例:
            # 方式1: 使用默认配置
            processor = BaseFeatureProcessor()
            processor.fit()  # 或 processor.fit(None)
            
            # 方式2: 根据数据自适应
            processor = BaseFeatureProcessor()
            processor.fit(train_df)
        """
        if df is not None:

            # Fit 所有链式处理器
            if self.processors:
                # self.logger.info(f"Fitting {len(self.processors)} processors in chain...")
                for i, proc in enumerate(self.processors, 1):
                    try:
                        if hasattr(proc, 'fit'):
                            proc.fit(df)
                            # self.logger.info(f"  [{i}/{len(self.processors)}] Fitted {proc.__class__.__name__}")
                    except Exception as e:
                        self.logger.error(f"  [{i}/{len(self.processors)}] Error fitting {proc.__class__.__name__}: {e}")
        else:
            # 使用默认配置
            self.logger.info("No data provided!")
            
        self._fitted = True
        self.logger.info("BaseFeatureProcessor fitted successfully")
        return self
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        print(f">>> [BaseFeatureProcessor] input: {len(df)} rows, {df['code'].n_unique()} codes")
        
        self.logger.info("=" * 80)
        self.logger.info("Starting BaseFeatureProcessor pipeline")
        self.logger.info("=" * 80)

        if "code" in df.columns and "datetime" in df.columns:
            df = df.sort(["code", "datetime"])

        base_df = self._precompute_features(df)
        print(f">>> [BaseFeatureProcessor] after precompute: {len(base_df)} rows")

        base_snapshot_cols = ["datetime", "open", "high", "low", "close", "volume", "money"]
        base_snapshot_exprs = [
            pl.col(c).last().alias(c) for c in base_snapshot_cols if c in base_df.columns
        ]
        result = (
            base_df
            .group_by("code", maintain_order=True)
            .agg(base_snapshot_exprs)
        )
        print(f">>> [BaseFeatureProcessor] base snapshot: {len(result)} rows")

        if not self.processors:
            return result

        for i, proc in enumerate(self.processors, 1):
            try:
                if hasattr(proc, "_fitted") and not proc._fitted:
                    proc.fit(base_df)

                snap = proc(base_df)
                
                # 🔑 关键：打印每个 processor 的输出行数
                print(f">>> [{i}] {proc.__class__.__name__}: {len(base_df)} -> {len(snap)} rows")

                if "code" not in snap.columns:
                    raise ValueError(f"{proc.__class__.__name__} output missing required column: 'code'")

                existing_cols = set(result.columns)
                keep_cols = ["code"] + [c for c in snap.columns if c != "code" and c not in existing_cols]

                if len(keep_cols) == 1:
                    continue

                snap = snap.select(keep_cols)
                result = result.join(snap, on="code", how="left")
                
                print(f">>> [{i}] After join, result: {len(result)} rows")

            except Exception as e:
                self.logger.error(f"  ✗ Error in {proc.__class__.__name__}: {e}")
                raise

        print(f">>> [BaseFeatureProcessor] final result: {len(result)} rows")
        return result

    def _precompute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        内部方法：执行基础特征预计算
        
        Args:
            df: 输入数据
            
        Returns:
            添加了预计算特征的 DataFrame
        """
        self.logger.info("Starting feature precomputation...")
        
        # Phase 1: 价格和成交量变化率
        #self.logger.info("Phase 1/8: Computing price and volume returns...")
        exprs = []
        
        # 1. 价格变化率
        exprs.append(pl.col('close').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).fill_null(0.0).alias(f'_close_ret_1'))

        for period in self.periods:
            exprs.append(
                pl.col('close').pct_change(period).over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).fill_null(0.0).alias(f'_close_ret_{period}')
            )
        
        # 2. 成交量变化率
        # 首先用log volume来替代volume
        df = df.with_columns(
            pl.col('volume').log1p().alias('volume')
        )

        for period in self.periods:
            exprs.append(
                pl.col('volume').pct_change(period).over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).fill_null(0.0).alias(f'_volume_ret_{period}')
            )
        
        df = df.with_columns(exprs)
        
        # Phase 2: 收益率的滚动统计
        # self.logger.info("Phase 2/8: Computing return rolling statistics...")
        exprs = []
        
        for period in self.periods:
            # 3. 收益率滚动均值
            exprs.append(
                pl.col('_close_ret_1').rolling_mean(window_size=period).over('code')
                .alias(f'_ret_rolling_mean_{period}')
            )
            
            # 收益率滚动标准差
            exprs.append(
                pl.col('_close_ret_1').rolling_std(window_size=period).over('code')
                .alias(f'_ret_rolling_std_{period}')
            )
            
            # 收益率滚动方差
            exprs.append(
                pl.col('_close_ret_1').rolling_var(window_size=period).over('code')
                .alias(f'_ret_rolling_var_{period}')
            )
        
        # 偏度（只对部分周期计算）
        """for period in self.periods:
            if period in self.rolling_periods:
                exprs.append(
                    pl.col('_close_ret_1').rolling_skew(window_size=period).over('code')
                    .alias(f'_ret_rolling_skew_{period}')
                )"""
        
        df = df.with_columns(exprs)
        
        # Phase 3: 成交量滚动统计
        # self.logger.info("Phase 3/8: Computing volume rolling statistics...")
        exprs = []
        
        for period in self.periods:
            # 4. 成交量滚动均值
            exprs.append(
                pl.col('volume').rolling_mean(window_size=period).over('code')
                .alias(f'_volume_rolling_mean_{period}')
            )
            
            # 成交量滚动标准差
            exprs.append(
                pl.col('volume').rolling_std(window_size=period).over('code')
                .alias(f'_volume_rolling_std_{period}')
            )
        
        df = df.with_columns(exprs)
        
        # Phase 4: 价格滚动统计
        # self.logger.info("Phase 4/8: Computing price rolling statistics...")
        exprs = []
        
        for period in self.periods:
            # 5. 收盘价滚动均值
            exprs.append(
                pl.col('close').rolling_mean(window_size=period).over('code')
                .alias(f'_close_rolling_mean_{period}')
            )
            
            # 收盘价滚动标准差
            exprs.append(
                pl.col('close').rolling_std(window_size=period).over('code')
                .alias(f'_close_rolling_std_{period}')
            )
            
            # 收盘价滚动最大值
            exprs.append(
                pl.col('close').rolling_max(window_size=period).over('code')
                .alias(f'_close_rolling_max_{period}')
            )
            
            # 收盘价滚动最小值
            exprs.append(
                pl.col('close').rolling_min(window_size=period).over('code')
                .alias(f'_close_rolling_min_{period}')
            )
        
        # High/Low 的极值
        for period in self.periods:
            exprs.append(
                pl.col('high').rolling_max(window_size=period).over('code')
                .alias(f'_high_rolling_max_{period}')
            )
            exprs.append(
                pl.col('low').rolling_min(window_size=period).over('code')
                .alias(f'_low_rolling_min_{period}')
            )
        
        df = df.with_columns(exprs)
        
        # Phase 5: 波动率
        # self.logger.info("Phase 5/8: Computing volatility...")
        exprs = []
        
        for period in self.periods:
            if f'_ret_rolling_std_{period}' in df.columns:
                # 6. 波动率
                exprs.append(
                    (pl.col(f'_ret_rolling_std_{period}') * np.sqrt(period))
                    .alias(f'_volatility_{period}')
                )
        
        df = df.with_columns(exprs)
        
        # Phase 8: RSI 和派生特征
        # self.logger.info("Phase 8/8: Computing RSI and derived features...")
        exprs = []
        
        # RSI 中间值
        for period in self.periods:
            alpha = 1.0 / period
            
            # 收益为正的部分
            exprs.append(
                pl.when(pl.col('_close_ret_1') > 0)
                .then(pl.col('_close_ret_1'))
                .otherwise(0)
                .ewm_mean(alpha=alpha)
                .over('code')
                .alias(f'_rsi_gain_{period}')
            )
            
            # 收益为负的部分
            exprs.append(
                pl.when(pl.col('_close_ret_1') < 0)
                .then(-pl.col('_close_ret_1'))
                .otherwise(0)
                .ewm_mean(alpha=alpha)
                .over('code')
                .alias(f'_rsi_loss_{period}')
            )
        
        df = df.with_columns(exprs)
        
        # RSI 最终值
        exprs = []
        for period in self.periods:
            exprs.append(
                (100 - 100 / (1 + pl.col(f'_rsi_gain_{period}') / (pl.col(f'_rsi_loss_{period}') + EPS)))
                .alias(f'_rsi_{period}')
            )
        
        # 布林带位置
        for period in self.periods:
            if (f'_close_rolling_mean_{period}' in df.columns and 
                f'_close_rolling_std_{period}' in df.columns):
                exprs.append(
                    ((pl.col('close') - pl.col(f'_close_rolling_mean_{period}')) / 
                     (2 * pl.col(f'_close_rolling_std_{period}') + EPS))
                    .alias(f'_bollinger_pos_{period}')
                )
        
        df = df.with_columns(exprs)
        
        self.logger.info("✓ Feature precomputation complete!")
        
        return df