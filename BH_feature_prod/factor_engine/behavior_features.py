import numpy as np
import polars as pl
from typing import List, Optional

from .base import (
    Processor, EPS, get_logger,
    PERIODS
)

"""
===========================

包含：
- RoundNumberProcessor: 整数关口因子
- FOMOFUDProcessor: FOMO/FUD 情绪因子
- RetailPatternProcessor: 散户交易模式因子
- HerdingProcessor: 羊群效应因子
- MicrostructureProcessor: 市场微观结构因子
- SentimentCycleProcessor: 情绪周期因子
- AttentionProcessor: 市场注意力因子

"""

class RoundNumberProcessor(Processor):
    """
    整数关口吸引力因子处理器（实盘流式计算版）
    
    计算价格对整数关口的吸引效应
    
    采用混合模式：
    - 大部分特征：group_by().agg() 高效计算
    - 少数复杂特征：保留 rolling().over() 方式
    """
    
    def __init__(self):
        super().__init__()
        self.base_levels = None
        self.psych_levels = None
        self.decimal_places = None
    
    def fit(self, df: pl.DataFrame = None) -> "RoundNumberProcessor":
        """根据价格水平确定整数关口级别"""
        if df is None:
            return self
        
        price_stats = df.select([
            pl.col('close').median().alias('median_price'),
        ]).row(0)
        median_price = price_stats[0]
        price_magnitude = 10 ** np.floor(np.log10(median_price + EPS))
        
        if median_price < 1:
            self.base_levels = [0.001, 0.005, 0.01, 0.05, 0.1]
            self.psych_levels = [0.005, 0.01, 0.025, 0.05, 0.1]
            self.decimal_places = max(0, int(-np.log10(price_magnitude)) + 2)
        elif median_price < 10:
            self.base_levels = [0.1, 0.5, 1, 5, 10]
            self.psych_levels = [0.25, 0.5, 1, 2.5, 5]
            self.decimal_places = 2
        elif median_price < 100:
            self.base_levels = [1, 5, 10, 50, 100]
            self.psych_levels = [10, 25, 50, 100, 250]
            self.decimal_places = 2
        else:
            self.base_levels = [int(price_magnitude * x) for x in [0.1, 0.5, 1, 5, 10]]
            self.psych_levels = [10, 25, 50, 100, 250]
            self.decimal_places = 0
        
        self._fitted = True
        return self
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算整数关口因子"""
        if not self._fitted:
            self.fit(df)
        
        # 确保按 code 和时间排序
        df = df.sort(['code', 'datetime'])
        
        # =====================================================================
        # Part 1: 复杂特征（保留 rolling，需要滑动窗口内每个点的计算）
        # =====================================================================
        complex_expressions = []
        
        # 整数关口附近停留时间（需要 rolling_mean）
        for i, level in enumerate(self.base_levels):
            threshold = level * 0.01 if level >= 1 else level * 0.05
            complex_expressions.append(
                (pl.col('close') % level < threshold)
                .cast(pl.Float32)
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'near_round_{i+1}_time')
            )
        
        # 整数关口反弹（需要 rolling_mean）
        for i, level in enumerate(self.base_levels):
            threshold = level * 0.01 if level >= 1 else level * 0.05
            complex_expressions.append(
                ((pl.col('close') % level < threshold) & (pl.col("close") > pl.col("close").shift(1).over("code")))
                .cast(pl.Float32)
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'round_{i+1}_bounce')
            )
        
        # 成交量聚集（需要条件 rolling_mean）
        for i, level in enumerate(self.base_levels):
            threshold = level * 0.02 if level >= 1 else level * 0.1
            complex_expressions.append(
                (pl.when(pl.col('close') % level < threshold)
                 .then(pl.col('volume'))
                 .otherwise(0)
                 .rolling_mean(window_size=60)
                 .over('code') / 
                 (pl.col('_volume_rolling_mean_15') + EPS))
                .alias(f'round_{i+1}_volume_cluster')
            )
        
        # 支撑反弹（需要 rolling_sum）
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('low') == pl.col(f'_close_rolling_min_{period}')) & (pl.col('close') > pl.col('low')))
                .cast(pl.Float32)
                .rolling_sum(window_size=period)
                .over('code')
                .alias(f'support_bounce_{period}min')
            )
        
        # 阻力测试（需要 rolling_sum）
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('high') == pl.col(f'_close_rolling_max_{period}')) & (pl.col('close') < pl.col('high')))
                .cast(pl.Float32)
                .rolling_sum(window_size=period)
                .over('code')
                .alias(f'resistance_test_{period}min')
            )
        
        # 计算复杂特征
        df_with_complex = df.with_columns(complex_expressions)
        complex_cols = [expr.meta.output_name() for expr in complex_expressions]
        
        # =====================================================================
        # Part 2: 简单特征 + 合并（一次性 group_by）
        # =====================================================================
        agg_expressions = []
        
        # -----------------------------------------------------------------
        # 简单特征：直接用 last() 或 tail() + 聚合
        # -----------------------------------------------------------------
        
        # 整数关口距离（瞬时特征，只需最新值）
        for i, level in enumerate(self.base_levels):
            agg_expressions.append(
                ((pl.col('close').last() % level) / level)
                .alias(f'round_{i+1}_distance')
            )
        
        # 整数关口突破：(close % level) 的 diff
        for i, level in enumerate(self.base_levels):
            agg_expressions.append(
                (((pl.col('close').last() % level) - (pl.col('close').tail(2).first() % level)) / level)
                .alias(f'round_{i+1}_breakout')
            )
        
        # 斐波那契回撤位（瞬时特征）
        for i, ratio in enumerate([0.236, 0.382, 0.5, 0.618, 0.786]):
            agg_expressions.append(
                (((pl.col('close').last() - pl.col('_close_rolling_min_60').last()) / 
                  (pl.col('_close_rolling_max_60').last() - pl.col('_close_rolling_min_60').last() + EPS) - ratio).abs())
                .alias(f'fib_{int(ratio*1000)}_distance')
            )
        
        # 百分比移动接近度（瞬时特征）
        for pct in [5, 10, 15, 20, 25]:
            agg_expressions.append(
                ((pl.col('_close_ret_15').last() * 100 % pct) / pct)
                .alias(f'pct_{pct}_move_proximity')
            )
        
        # 历史高点距离（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (1 - pl.col('close').last() / (pl.col(f'_close_rolling_max_{period}').last() + EPS))
                .alias(f'ath_{period}min_distance')
            )
        
        # 心理价位距离（瞬时特征）
        for i, level in enumerate(self.psych_levels):
            agg_expressions.append(
                ((pl.col('close').last() % level) / level)
                .alias(f'psych_{i+1}_distance')
            )
        
        # -----------------------------------------------------------------
    
        # Part 3: 执行聚合
        # =====================================================================
        """result_with_simple = df.group_by("code", maintain_order=True).agg(agg_expressions)
        result_with_complex = df_with_complex.group_by('code', maintain_order=True).last() 

        base_cols = set(df.columns) 
    
        simple_snap = result_with_simple

        complex_keep = ["code"] + [c for c in result_with_complex.columns if c != "code" and c not in base_cols]
        complex_snap = result_with_complex.select(complex_keep)

        result = complex_snap.join(simple_snap, on="code", how="left")"""

        # 动态构建 agg 表达式
        all_agg = agg_expressions.copy()  # simple 特征
        all_agg += [pl.col(c).last().alias(c) for c in complex_cols]  # complex 特征取 last
        result = df_with_complex.group_by("code", maintain_order=True).agg(all_agg)
        
        # 释放中间 DataFrame
        # del df_with_complex
        # gc.collect()
        
        return result
    

class FOMOFUDProcessor(Processor):
    """
    FOMO/FUD 情绪因子处理器（实盘流式计算版）
    
    计算市场恐惧和贪婪情绪指标
    
    采用混合模式：
    - 大部分特征：group_by().agg() 高效计算
    - 少数复杂特征：保留 rolling().over() 方式
    """
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 FOMO/FUD 因子"""
        
        # 确保按 code 和时间排序
        df = df.sort(['code', 'datetime'])
        
        # =====================================================================
        # Part 1: 复杂特征（保留 rolling，需要滑动窗口内每个点的计算）
        # =====================================================================
        complex_expressions = [
            # return_skewness: 需要完整的 rolling_skew
            pl.col('close').pct_change().over('code')
            .rolling_skew(window_size=241).over('code')
            .alias('return_skewness'),
            
            # consecutive_gains: 需要 rolling_sum
            (pl.col('_close_ret_1') > 0).cast(pl.Int32)
            .rolling_sum(window_size=241).over('code')
            .alias('consecutive_gains'),
            
            # rally_volume: 条件 rolling_mean
            pl.when(pl.col('_close_ret_1') > 0)
            .then(pl.col('volume'))
            .otherwise(0)
            .rolling_mean(window_size=60).over('code')
            .alias('rally_volume'),
            
            # uptrend_acceleration: 复杂条件
            pl.when(pl.col('_close_ret_1') > 0)
            .then(pl.col('close').pct_change().over('code').diff().over('code'))
            .otherwise(0)
            .rolling_mean(window_size=60).over('code')
            .alias('uptrend_acceleration'),
        ]
        
        # 连续上涨（条件 rolling_sum）
        for threshold in [0.005, 0.01, 0.015, 0.025]:
            complex_expressions.append(
                (pl.col('_close_ret_1') > threshold)
                .cast(pl.Int32)
                .rolling_sum(window_size=60)
                .over('code')
                .alias(f'consecutive_up_{int(threshold*10000)}bp')
            )
        
        # V型反转
        for period in [5, 15, 30, 60, 120]:
            mid = period // 2
            complex_expressions.append(
                (pl.col('close') / (pl.col('close').shift(2*mid).over('code') + EPS) * 
                 (pl.col('low').rolling_min(window_size=period).over('code') == pl.col('low'))
                 .shift(mid).over('code').cast(pl.Float32))
                .alias(f'v_reversal_{period}min')
            )
        
        # 极端反转
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('close') == pl.col(f'_close_rolling_max_{period}')) | 
                 (pl.col('close') == pl.col(f'_close_rolling_min_{period}')))
                .cast(pl.Float32)
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'extreme_reversal_{period}min')
            )
        
        # RSI 过热
        for period in PERIODS:
            complex_expressions.append(
                (pl.col(f'_rsi_{period}') > 70)
                .cast(pl.Float32)
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'sentiment_overheat_{period}min')
            )
        
        # RSI 过冷
        for period in PERIODS:
            complex_expressions.append(
                (pl.col(f'_rsi_{period}') < 30)
                .cast(pl.Float32)
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'sentiment_overcold_{period}min')
            )
        
        # 成交量价格背离
        for period in PERIODS:
            price_change = pl.col(f'_close_ret_{period}')
            volume_change = pl.col(f'_volume_ret_{period}')
            complex_expressions.append(
                (price_change.sign() != volume_change.sign())
                .cast(pl.Float32)
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'volume_price_divergence_{period}min')
            )
        
        # 恐惧指数
        for period in PERIODS:
            complex_expressions.append(
                (1 - (pl.col('low') / (pl.col('high') + EPS))
                 .rolling_mean(window_size=period).over('code'))
                .alias(f'fear_index_{period}min')
            )
        
        # 贪婪指数
        for period in PERIODS:
            price_momentum = pl.col(f'_close_ret_{period}')
            volume_momentum = pl.col(f'_volume_ret_{period}')
            complex_expressions.append(
                (price_momentum * volume_momentum)
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'greed_index_{period}min')
            )
        
        # 情绪振荡器
        for period in PERIODS:
            high_count = (pl.col('close') >= pl.col(f'_high_rolling_max_{period}')).cast(pl.Float32)
            low_count = (pl.col('close') <= pl.col(f'_low_rolling_min_{period}')).cast(pl.Float32)
            complex_expressions.append(
                (high_count - low_count)
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'sentiment_oscillator_{period}min')
            )
        
        # 情绪极端持续时间
        for period in PERIODS:
            extreme_condition = (pl.col(f'_rsi_{period}') > 80) | (pl.col(f'_rsi_{period}') < 20)
            complex_expressions.append(
                extreme_condition
                .cast(pl.Float32)
                .rolling_sum(window_size=241)
                .over('code')
                .alias(f'sentiment_extreme_duration_{period}min')
            )
        
        # 价格跳跃
        for period in PERIODS:
            price_change = pl.col(f'_close_ret_{period}')
            price_std = pl.col('_ret_rolling_std_241')
            complex_expressions.append(
                (price_change.abs() > price_std * 2)
                .cast(pl.Float32)
                .rolling_sum(window_size=241)
                .over('code')
                .alias(f'price_jump_{period}min')
            )
        
        # 流动性压力
        for period in PERIODS:
            spread = (pl.col('high') - pl.col('low')) / (pl.col('close') + EPS)
            volume_normalized = pl.col('volume') / (pl.col(f'_volume_rolling_mean_{period}') + EPS)
            complex_expressions.append(
                (spread / (volume_normalized + EPS))
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'liquidity_stress_{period}min')
            )
        
        # 情绪惯性
        for period in PERIODS:
            rsi_change = pl.col(f'_rsi_{period}').diff().over('code').abs()
            complex_expressions.append(
                (1 / (1 + rsi_change))
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'sentiment_inertia_{period}min')
            )
        
        # 计算复杂特征
        df_with_complex = df.with_columns(complex_expressions)
        complex_cols = [expr.meta.output_name() for expr in complex_expressions]

        # =====================================================================
        # Part 2: 简单特征 + 合并（一次性 group_by）
        # =====================================================================
        agg_expressions = [
            # 基础列
            pl.col('datetime').last().alias('datetime'),
            pl.col('close').last().alias('close'),
            pl.col('open').last().alias('open'),
            pl.col('high').last().alias('high'),
            pl.col('low').last().alias('low'),
            pl.col('volume').last().alias('volume'),
            pl.col('money').last().alias('money'),
            
            # -----------------------------------------------------------------
            # 简单特征：直接用 tail + 聚合
            # -----------------------------------------------------------------
            
            # volatility_index
            (pl.col('_ret_rolling_std_60').last() * np.sqrt(60) * 100)
            .alias('volatility_index'),
            
            # return_kurtosis_proxy
            pl.col('_ret_rolling_var_60').last()
            .alias('return_kurtosis_proxy'),
        ]
        
        # FOMO 浪涌: ret_N / std_241
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_close_ret_{period}').last() / 
                 (pl.col('_ret_rolling_std_241').last() + EPS))
                .alias(f'fomo_surge_{period}min')
            )
        
        # 成交量 FOMO: volume_ma_N / volume_ma_4N
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_volume_rolling_mean_{period}').last() / 
                 (pl.col('volume').tail(period * 4).mean() + EPS))
                .alias(f'volume_fomo_{period}min')
            )
        
        # 恐慌抛售
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col(f'_close_ret_{period}').last() < -0.005).cast(pl.Float32) * 
                 (pl.col('volume').last() / (pl.col('volume').tail(period * 4).mean() + EPS)))
                .alias(f'panic_sell_{period}min')
            )
        
        # 追涨动量: ret_1[-1] * ret_1[-1-lag]
        for lag in [1, 2, 3, 4, 5]:
            agg_expressions.append(
                (pl.col('_close_ret_1').last() * 
                 pl.col('_close_ret_1').tail(lag + 1).first())
                .alias(f'chase_momentum_lag{lag}')
            )
        
        # RSI 变化
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_rsi_{period}').last() - pl.col(f'_rsi_{period}').tail(2).first()).abs()
                .alias(f'sentiment_shift_{period}min')
            )
        
        # 价格加速度: ret_N 的 diff
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_close_ret_{period}').last() - 
                 pl.col(f'_close_ret_{period}').tail(2).first())
                .alias(f'price_acceleration_{period}min')
            )
        
        # 成交量加速度
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_volume_ret_{period}').last() - 
                 pl.col(f'_volume_ret_{period}').tail(2).first())
                .alias(f'volume_acceleration_{period}min')
            )
        
        # 波动率状态: std_short / std_long
        for period in PERIODS:
            agg_expressions.append(
                (pl.col('_close_ret_1').tail(period // 2).std() / 
                 (pl.col(f'_ret_rolling_std_{period}').last() + EPS))
                .alias(f'volatility_regime_{period}min')
            )
        
        # 布林带位置
        for period in PERIODS:
            agg_expressions.append(
                pl.col(f'_bollinger_pos_{period}').last()
                .alias(f'bollinger_position_{period}min')
            )
        
        # 动量背离
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_close_ret_{period}').last() - pl.col(f'_volume_ret_{period}').last()).abs()
                .alias(f'momentum_divergence_{period}min')
            )
        
        # 成交量异常
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col('volume').last() - pl.col(f'_volume_rolling_mean_{period}').last()) / 
                 (pl.col(f'_volume_rolling_std_{period}').last() + EPS)).abs()
                .alias(f'volume_anomaly_{period}min')
            )
        
        # =====================================================================
        # Part 3: 执行聚合
        # =====================================================================
        """result_with_simple = df.group_by("code", maintain_order=True).agg(agg_expressions)
        result_with_complex = df_with_complex.group_by('code', maintain_order=True).last() 

        base_cols = set(df.columns) 
    
        simple_snap = result_with_simple

        complex_keep = ["code"] + [c for c in result_with_complex.columns if c != "code" and c not in base_cols]
        complex_snap = result_with_complex.select(complex_keep)

        result = complex_snap.join(simple_snap, on="code", how="left")"""

        # 动态构建 agg 表达式
        all_agg = agg_expressions.copy()  # simple 特征
        all_agg += [pl.col(c).last().alias(c) for c in complex_cols]  # complex 特征取 last
        result = df_with_complex.group_by("code", maintain_order=True).agg(all_agg)
        
        # 释放中间 DataFrame
        # del df_with_complex
        # gc.collect()
        
        # self.logger.info(f"Generated {len(result.columns)} FOMO/FUD factors")
        return result


class RetailPatternProcessor(Processor):
    """
    散户交易模式因子处理器（实盘流式计算版）
    
    识别散户典型交易行为：追涨杀跌、止损止盈、周末效应等
    
    采用混合模式：
    - 大部分特征：group_by().agg() 高效计算
    - 少数复杂特征：保留 rolling().over() 方式
    """
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算散户交易模式因子"""
        
        # 确保按 code 和时间排序
        df = df.sort(['code', 'datetime'])
        
        # =====================================================================
        # Part 1: 复杂特征（保留 rolling，需要滑动窗口内每个点的计算）
        # =====================================================================
        complex_expressions = []
        
        # 追高买入（需要 rolling_sum）
        for period in [5, 15, 30, 60, 120, 241]:
            complex_expressions.append(
                (((pl.col('close') > pl.col('high').shift(1).over('code')).cast(pl.Float32) * pl.col('volume').cast(pl.Float32))
                 .rolling_sum(window_size=period)
                 .over('code') / 
                 (pl.col('volume').cast(pl.Float32).rolling_sum(window_size=period).over('code') + EPS))
                .cast(pl.Float32)
                .alias(f'buy_high_pattern_{period}min')
            )
        
        # 抄底买入（需要 rolling_sum）
        for period in [5, 15, 30, 60, 120, 241]:
            complex_expressions.append(
                (((pl.col('close') < pl.col('low').shift(1).over('code')).cast(pl.Float32) * pl.col('volume').cast(pl.Float32))
                 .rolling_sum(window_size=period)
                 .over('code') / 
                 (pl.col('volume').cast(pl.Float32).rolling_sum(window_size=period).over('code') + EPS))
                .cast(pl.Float32)
                .alias(f'buy_dip_pattern_{period}min')
            )
        
        # 散户恐慌（需要 rolling_mean）
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('_close_ret_1') < -0.02).cast(pl.Float32) * 
                 (pl.col('volume').cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32))
                .rolling_mean(window_size=241)
                .over('code')
                .cast(pl.Float32)
                .alias(f'retail_panic_{period}min')
            )
        
        # 成交量聚集（需要 rolling_sum）
        for period in PERIODS:
            vol_zscore = ((pl.col('volume').cast(pl.Float32) - pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32)) / 
                          (pl.col(f'_volume_rolling_std_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32)
            complex_expressions.append(
                (vol_zscore > 2).cast(pl.Float32)
                .rolling_sum(window_size=241)
                .over('code')
                .cast(pl.Float32)
                .alias(f'volume_clustering_{period}min')
            )
        
        # 逆势模式（需要 rolling_mean）
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('_close_ret_1') > 0).cast(pl.Float32) * 
                 (pl.col('volume') < pl.col(f'_volume_rolling_mean_{period}')).cast(pl.Float32))
                .rolling_mean(window_size=241)
                .over('code')
                .cast(pl.Float32)
                .alias(f'contrarian_pattern_{period}min')
            )
        
        # 盘整突破（需要 rolling_mean）
        for period in PERIODS:
            price_range = (pl.col(f'_high_rolling_max_{period}').cast(pl.Float32) - 
                           pl.col(f'_low_rolling_min_{period}').cast(pl.Float32))
            daily_range = pl.col('high').cast(pl.Float32) - pl.col('low').cast(pl.Float32)
            complex_expressions.append(
                (daily_range / (price_range + EPS)).cast(pl.Float32)
                .rolling_mean(window_size=241)
                .over('code')
                .cast(pl.Float32)
                .alias(f'consolidation_breakout_{period}min')
            )
        
        # 情绪反转（需要 rolling_mean）
        for period in PERIODS:
            price_momentum = pl.col(f'_close_ret_{period}')
            volume_momentum = pl.col(f'_volume_ret_{period}')
            complex_expressions.append(
                ((price_momentum > 0).cast(pl.Float32) * (volume_momentum < 0).cast(pl.Float32))
                .rolling_mean(window_size=241)
                .over('code')
                .cast(pl.Float32)
                .alias(f'sentiment_reversal_{period}min')
            )
        
        # 散户疲劳（需要 rolling_sum）
        for period in PERIODS:
            consecutive_days = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_sum(window_size=period).over('code')
            complex_expressions.append(
                ((consecutive_days / period).cast(pl.Float32) * 
                 (pl.col('volume').cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32))
                .cast(pl.Float32)
                .alias(f'retail_fatigue_{period}min')
            )
        
        # 羊群效应（需要 rolling_mean）
        for period in PERIODS:
            price_direction = pl.col('_close_ret_1').sign().cast(pl.Float32)
            volume_spike = (pl.col('volume').cast(pl.Float32) / 
                            (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32)
            complex_expressions.append(
                (price_direction * volume_spike)
                .rolling_mean(window_size=241)
                .over('code')
                .cast(pl.Float32)
                .alias(f'herding_effect_{period}min')
            )
        
        # 价格发现滞后（需要 rolling_corr）
        for lag in [1, 5, 15]:
            vol_pct_change = (
                pl.col('volume')
                .pct_change()
                .over('code')
                .fill_nan(0.0)
                .replace([float('inf'), float('-inf')], 0.0)
                .cast(pl.Float32)
                .shift(lag)
                .over('code')
            )
            complex_expressions.append(
                pl.rolling_corr(
                    pl.col('_close_ret_1').cast(pl.Float32), 
                    vol_pct_change, 
                    window_size=241
                )
                .over('code')
                .fill_nan(0.0)
                .fill_null(0.0)
                .clip(-1.0, 1.0)
                .cast(pl.Float32)
                .alias(f'price_discovery_lag{lag}')
            )
        
        # 资金流向（需要 rolling_sum + diff）
        for period in PERIODS:
            typical_price = ((pl.col('high').cast(pl.Float32) + pl.col('low').cast(pl.Float32) + 
                              pl.col('close').cast(pl.Float32)) / 3.0).cast(pl.Float32)
            money_flow = typical_price * pl.col('volume').cast(pl.Float32)
            complex_expressions.append(
                (money_flow.diff().over('code').rolling_sum(window_size=period).over('code') / 
                 (money_flow.rolling_sum(window_size=period).over('code') + EPS))
                .cast(pl.Float32)
                .alias(f'money_flow_{period}min')
            )
        
        # 计算复杂特征
        df_with_complex = df.with_columns(complex_expressions)
        complex_cols = [expr.meta.output_name() for expr in complex_expressions]

        # =====================================================================
        # Part 2: 简单特征 + 合并（一次性 group_by）
        # =====================================================================
        agg_expressions = [
            # 基础列
            pl.col('datetime').last().alias('datetime'),
            pl.col('close').last().alias('close'),
            pl.col('open').last().alias('open'),
            pl.col('high').last().alias('high'),
            pl.col('low').last().alias('low'),
            pl.col('volume').last().alias('volume'),
            pl.col('money').last().alias('money'),
        ]
        
        # -----------------------------------------------------------------
        # 简单特征：直接用 last() 或 tail() + 聚合
        # -----------------------------------------------------------------
        
        # 散户投降（瞬时特征）
        agg_expressions.append(
            ((pl.col('_close_ret_241').last() < -0.04) & 
             (pl.col('volume').last() > pl.col('_volume_rolling_mean_241').last() * 2) & 
             (pl.col('_close_ret_1').last() < -0.01))
            .cast(pl.Float32)
            .alias('retail_capitulation')
        )
        
        # 止损触发（瞬时特征）
        for threshold in [0.02, 0.04]:
            agg_expressions.append(
                ((pl.col('_close_ret_60').last() < -threshold).cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col('_volume_rolling_mean_60').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'stop_loss_{int(threshold*100)}pct')
            )
        
        # 止盈触发（瞬时特征）
        for threshold in [0.02, 0.04]:
            agg_expressions.append(
                ((pl.col('_close_ret_60').last() > threshold).cast(pl.Float32) * 
                 (pl.col('_close_ret_1').last() < 0).cast(pl.Float32))
                .cast(pl.Float32)
                .alias(f'take_profit_{int(threshold*100)}pct')
            )
        
        # 区间交易（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col('volume').last().cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS)) * 
                 (1.0 / (1.0 + pl.col(f'_ret_rolling_std_{period}').last().cast(pl.Float32))))
                .cast(pl.Float32)
                .alias(f'range_trading_{period}min')
            )
        
        # 突破交易（瞬时特征，需要 shift）
        for period in PERIODS:
            agg_expressions.append(
                (((pl.col('close').last() > pl.col(f'_high_rolling_max_{period}').tail(2).first()) | 
                  (pl.col('close').last() < pl.col(f'_low_rolling_min_{period}').tail(2).first()))
                 .cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'breakout_trading_{period}min')
            )
        
        # 动量追逐（瞬时特征）
        for lag in [1, 3, 5]:
            agg_expressions.append(
                (pl.col('_close_ret_1').last().cast(pl.Float32) * 
                 pl.col('_close_ret_1').tail(lag + 1).first().cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col('_volume_rolling_mean_60').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'momentum_chasing_lag{lag}')
            )
        
        # 散户拥挤度（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_volume_rolling_std_{period}').last().cast(pl.Float32) / 
                 (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS))
                .cast(pl.Float32)
                .alias(f'retail_crowding_{period}min')
            )
        
        # 价格锚定（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col('close').last().cast(pl.Float32) - pl.col(f'_close_rolling_mean_{period}').last().cast(pl.Float32)) / 
                 (pl.col(f'_close_rolling_std_{period}').last().cast(pl.Float32) + EPS))
                .abs()
                .cast(pl.Float32)
                .alias(f'price_anchoring_{period}min')
            )
        
        # 均值回归（瞬时特征，需要比较当前和前一期）
        for period in PERIODS:
            agg_expressions.append(
                (((pl.col('close').last() < pl.col(f'_close_rolling_mean_{period}').last()) & 
                  (pl.col('close').tail(2).first() > pl.col(f'_close_rolling_mean_{period}').tail(2).first()))
                 .cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'mean_reversion_{period}min')
            )
        
        # 新闻驱动（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (((pl.col('open').last().cast(pl.Float32) - pl.col('close').tail(2).first().cast(pl.Float32)) / 
                  (pl.col('close').tail(2).first().cast(pl.Float32) + EPS)).abs() * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'news_driven_{period}min')
            )
        
        # 技术分析依赖（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (((pl.col('close').last() > pl.col(f'_close_rolling_mean_{period}').last()) & 
                  (pl.col('close').tail(2).first() <= pl.col(f'_close_rolling_mean_{period}').tail(2).first()))
                 .cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col('_volume_rolling_mean_60').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'ta_dependency_{period}min')
            )
        
        # 支撑阻力测试强度（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (((pl.col('low').last().cast(pl.Float32) <= pl.col(f'_low_rolling_min_{period}').last().cast(pl.Float32) * 1.01) | 
                  (pl.col('high').last().cast(pl.Float32) >= pl.col(f'_high_rolling_max_{period}').last().cast(pl.Float32) * 0.99))
                 .cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'sr_test_intensity_{period}min')
            )
        
        # 波动率偏好（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_ret_rolling_std_{period}').last().cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'volatility_preference_{period}min')
            )
        
        
        # =====================================================================
        # Part 3: 执行聚合
        # =====================================================================
        """result_with_simple = df.group_by("code", maintain_order=True).agg(agg_expressions)
        result_with_complex = df_with_complex.group_by('code', maintain_order=True).last() 

        base_cols = set(df.columns) 
    
        simple_snap = result_with_simple

        complex_keep = ["code"] + [c for c in result_with_complex.columns if c != "code" and c not in base_cols]
        complex_snap = result_with_complex.select(complex_keep)

        result = complex_snap.join(simple_snap, on="code", how="left")"""

        # 动态构建 agg 表达式
        all_agg = agg_expressions.copy()  # simple 特征
        all_agg += [pl.col(c).last().alias(c) for c in complex_cols]  # complex 特征取 last
        result = df_with_complex.group_by("code", maintain_order=True).agg(all_agg)
        
        # 释放中间 DataFrame
        # del df_with_complex
        # gc.collect()
        
        return result 


class HerdingProcessor(Processor):
    """
    羊群效应因子处理器（实盘流式计算版）
    
    识别市场中的从众行为：方向共识度、信息级联、社会传染等
    
    采用混合模式：
    - 大部分特征：group_by().agg() 高效计算
    - 少数复杂特征：保留 rolling().over() 方式
    """
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算羊群效应因子"""
        
        # 确保按 code 和时间排序
        df = df.sort(['code', 'datetime'])
        
        # =====================================================================
        # Part 1: 复杂特征（保留 rolling，需要滑动窗口内每个点的计算）
        # =====================================================================
        complex_expressions = [
            # 强羊群效应（牛市）
            ((pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=120).over('code') > 0.7)
            .alias('strong_herding_bull'),
            
            # 强羊群效应（熊市）
            ((pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=120).over('code') < 0.3)
            .alias('strong_herding_bear'),
            
            # 市场分化
            (((pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=120).over('code')).is_between(0.45, 0.55))
            .alias('fragmented_market'),
            
            # 状态转换
            ((pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=120).over('code').diff(60).abs() > 0.1)
            .alias('regime_transition'),
            
            # 领导者行为
            (pl.col('_close_ret_15').abs() > pl.col('_close_ret_15').abs().rolling_quantile(0.9, window_size=60).over('code'))
            .cast(pl.Float32)
            .alias('leader_moves'),
            
            # 跟随倾向
            ((pl.col('_close_ret_1') > 0) == (pl.col('_close_ret_1').shift(1).over('code') > 0))
            .cast(pl.Float32)
            .rolling_mean(window_size=60)
            .over('code')
            .alias('follower_tendency'),
            
            # 领导效果
            (pl.col('_close_ret_15').shift().over('code') * pl.col('_close_ret_15'))
            .rolling_mean(window_size=120)
            .over('code')
            .alias('leader_effectiveness'),
            
            # 信息冲击吸收
            (pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).abs() / 
             (pl.col('_close_ret_1').abs() + EPS))
            .fill_nan(0.0)
            .fill_null(0.0)
            .rolling_mean(window_size=60)
            .over('code')
            .cast(pl.Float32)
            .alias('info_shock_absorption'),
            
            # 信息处理速度
            (pl.col('_close_ret_1').abs() / (pl.col('_close_ret_1').abs().shift(1).over('code') + EPS))
            .rolling_mean(window_size=60)
            .over('code')
            .alias('info_processing_speed'),
        ]
        
        # 方向共识（需要 rolling_mean）
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code') * 2 - 1)
                .abs()
                .alias(f'direction_consensus_{period}min')
            )
        
        # 极端同步（需要 rolling_mean + rolling_max）
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('high') == pl.col(f'_high_rolling_max_{period}')) &
                 (pl.col('volume') == pl.col('volume').rolling_max(window_size=period).over('code')))
                .cast(pl.Float32)
                .rolling_mean(window_size=period)
                .over('code')
                .alias(f'extreme_sync_{period}min')
            )
        
        # 反转同步（需要 rolling_mean + shift）
        for period in PERIODS:
            complex_expressions.append(
                (pl.col('_close_ret_1') * pl.col('_close_ret_1').shift(period).over('code'))
                .rolling_mean(window_size=period)
                .over('code')
                .alias(f'reversal_sync_{period}min')
            )
        
        # 波动率传染（需要 rolling_mean + shift）
        for lag in [1, 5, 15, 60, 241]:
            vol = pl.col('_close_ret_1').abs()
            complex_expressions.append(
                (vol * vol.shift(lag).over('code'))
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'volatility_contagion_lag{lag}')
            )
        
        # 成交量传染（需要 rolling_mean + shift）
        for lag in [1, 5, 15, 60, 241]:
            vol_ratio = pl.col('volume') / (pl.col('_volume_rolling_mean_60') + EPS)
            complex_expressions.append(
                (vol_ratio * vol_ratio.shift(lag).over('code'))
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'volume_contagion_lag{lag}')
            )
        
        # 趋势跟随（需要 rolling_mean + shift）
        for period in PERIODS:
            ma = pl.col(f'_close_rolling_mean_{period}')
            complex_expressions.append(
                ((pl.col('close') > ma) & (pl.col('close').shift(1).over('code') > ma.shift(1).over('code')))
                .cast(pl.Float32)
                .rolling_mean(window_size=period)
                .over('code')
                .alias(f'trend_following_{period}min')
            )
        
        # 共识打破（需要 rolling_mean + diff）
        for period in PERIODS:
            consensus = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            complex_expressions.append(
                ((consensus > 0.75).cast(pl.Float32).diff().over('code') * -1)
                .alias(f'consensus_break_{period}min')
            )
        
        # 价格磁吸（需要 rolling_mean）
        for period in PERIODS:
            ma = pl.col(f'_close_rolling_mean_{period}')
            complex_expressions.append(
                (pl.col('close') / (ma + EPS) - 1).abs()
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'price_magnetic_{period}min')
            )
        
        # 决策滞后（需要 rolling_corr）
        for period in PERIODS:
            price_signal = pl.col(f'_close_ret_{period}')
            volume_response = pl.col(f'_volume_ret_{period}')
            complex_expressions.append(
                pl.rolling_corr(price_signal.shift(1).over('code') * volume_response, price_signal, window_size=period)
                .over('code')
                .alias(f'decision_lag_{period}min')
            )
        
        # 羊群偏离（需要 rolling_mean）
        for period in PERIODS:
            avg_return = pl.col(f'_ret_rolling_mean_{period}')
            complex_expressions.append(
                (pl.col('_close_ret_1') - avg_return).abs()
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'herding_deviation_{period}min')
            )
        
        # 信息级联（需要 rolling_corr）
        for period in PERIODS:
            complex_expressions.append(
                pl.rolling_corr(
                    pl.col('volume'), 
                    pl.col('volume').shift(1).over('code'), 
                    window_size=period
                )
                .over('code')
                .fill_nan(0.0)
                .fill_null(0.0)
                .clip(-1.0, 1.0)
                .cast(pl.Float32)
                .alias(f'info_cascade_{period}min')
            )
        
        # 模仿交易（需要 rolling_corr）
        for period in PERIODS:
            volume_autocorr = (
                pl.rolling_corr(
                    pl.col('volume'), 
                    pl.col('volume').shift(1).over('code'), 
                    window_size=period
                )
                .over('code')
                .fill_nan(0.0)
                .fill_null(0.0)
                .clip(-1.0, 1.0)
            )
            price_autocorr = (
                pl.rolling_corr(
                    pl.col('_close_ret_1'), 
                    pl.col('_close_ret_1').shift(1).over('code'), 
                    window_size=period
                )
                .over('code')
                .fill_nan(0.0)
                .fill_null(0.0)
                .clip(-1.0, 1.0)
            )
            complex_expressions.append(
                (volume_autocorr * price_autocorr)
                .cast(pl.Float32)
                .alias(f'mimetic_trading_{period}min')
            )
        
        # 群体极化（需要 rolling_quantile + rolling_mean）
        for period in PERIODS:
            extreme_moves = pl.col('_close_ret_1').abs() > pl.col('_close_ret_1').abs().rolling_quantile(0.8, window_size=period*4).over('code')
            complex_expressions.append(
                extreme_moves.cast(pl.Float32)
                .rolling_mean(window_size=period)
                .over('code')
                .alias(f'group_polarization_{period}min')
            )
        
        # 从众压力（需要 rolling_mean）
        for period in PERIODS:
            majority_direction = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            complex_expressions.append(
                ((majority_direction - 0.5).abs() * 2)
                .alias(f'conformity_pressure_{period}min')
            )
        
        # 逆势敏感度（需要 rolling_mean + when/then）
        for period in PERIODS:
            crowd_sentiment = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            contrarian_signal = pl.when(crowd_sentiment > 0.7).then(-1).when(crowd_sentiment < 0.3).then(1).otherwise(0)
            complex_expressions.append(
                (contrarian_signal.shift().over('code') * pl.col('_close_ret_1'))
                .rolling_mean(window_size=60)
                .over('code')
                .alias(f'contrarian_sensitivity_{period}min')
            )
        
        # 网络效应（需要 rolling_mean）
        for period in PERIODS:
            vol_cluster = pl.col(f'_volume_rolling_std_{period}') / (pl.col(f'_volume_rolling_mean_{period}') + EPS)
            price_momentum = pl.col(f'_close_ret_{period}').abs()
            complex_expressions.append(
                (vol_cluster * price_momentum)
                .rolling_mean(window_size=120)
                .over('code')
                .alias(f'network_effect_{period}min')
            )
        
        # 信息不对称（需要 rolling_quantile + rolling_mean）
        for period in PERIODS:
            large_moves = pl.col('_close_ret_1').abs() > pl.col('_close_ret_1').abs().rolling_quantile(0.9, window_size=period*2).over('code')
            reversal = (pl.col('_close_ret_1').shift().over('code') * pl.col('_close_ret_1')) < 0
            complex_expressions.append(
                (large_moves.shift().over('code') & reversal)
                .cast(pl.Float32)
                .rolling_mean(window_size=period)
                .over('code')
                .alias(f'info_asymmetry_{period}min')
            )
        
        # 羊群衰减（需要 ewm_mean + rolling_mean）
        for period in PERIODS:
            herding_strength = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            complex_expressions.append(
                herding_strength.ewm_mean(span=period//2)
                .over('code')
                .alias(f'herding_decay_{period}min')
            )
        
        # 共识动量（需要 rolling_mean + diff）
        for period in PERIODS:
            consensus = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            complex_expressions.append(
                consensus.diff()
                .over('code')
                .rolling_mean(window_size=120)
                .over('code')
                .alias(f'consensus_momentum_{period}min')
            )
        
        # 异质性（需要 rolling_mean）
        for period in PERIODS:
            price_impact = pl.col('_close_ret_1').abs() / ((pl.col('volume') + EPS) / (pl.col(f'_volume_rolling_mean_{period}') + EPS) + EPS)
            complex_expressions.append(
                (1 / (1 + price_impact))
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'heterogeneity_{period}min')
            )
        
        # 羊群饱和（需要 rolling_mean）
        for period in PERIODS:
            direction_consistency = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            volume_surge = pl.col('volume') / (pl.col(f'_volume_rolling_mean_{period}') + EPS)
            complex_expressions.append(
                (direction_consistency * (1 - 1/(volume_surge + EPS)))
                .alias(f'herding_saturation_{period}min')
            )
        
        # 羊群反转（需要 rolling_mean + diff）
        for period in [60, 241]:
            herding_strength = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            complex_expressions.append(
                ((herding_strength > 0.8).cast(pl.Float32).diff().over('code') * -1 + 
                 (herding_strength < 0.2).cast(pl.Float32).diff().over('code'))
                .alias(f'herding_reversal_{period}min')
            )
        
        # 集体智慧（需要 rolling_mean + shift）
        for period in PERIODS:
            crowd_direction = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            future_performance = pl.col(f'_close_ret_{period}')
            complex_expressions.append(
                (crowd_direction.shift(period).over('code') * future_performance)
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'collective_wisdom_{period}min')
            )
        
        # 社会传染（需要 rolling_mean + shift）
        for lag in [1, 5, 30]:
            volume_contagion = (pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0) * 
                               pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).shift(lag).over('code'))
            price_contagion = pl.col('_close_ret_1') * pl.col('_close_ret_1').shift(lag).over('code')
            complex_expressions.append(
                (volume_contagion + price_contagion)
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'social_contagion_lag{lag}')
            )
        
        # 羊群免疫（需要 rolling_mean）
        for period in PERIODS:
            expected_herding = (pl.col('_close_ret_1') > 0).cast(pl.Float32).rolling_mean(window_size=period).over('code')
            actual_behavior = (pl.col('_close_ret_1') > 0).cast(pl.Float32)
            complex_expressions.append(
                (expected_herding - actual_behavior).abs()
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'herding_immunity_{period}min')
            )
        
        # 学习曲线（需要 rolling_mean + diff）
        for period in [5, 15, 60, 241]:
            pattern_volatility = pl.col('_close_ret_1').abs().rolling_mean(window_size=period).over('code')
            complex_expressions.append(
                pattern_volatility.diff()
                .over('code')
                .alias(f'learning_curve_{period}min')
            )
        
        # 计算复杂特征
        df_with_complex = df.with_columns(complex_expressions)
        complex_cols = [expr.meta.output_name() for expr in complex_expressions]

        # =====================================================================
        # Part 2: 简单特征 + 合并（一次性 group_by）
        # =====================================================================
        agg_expressions = [
            # 基础列
            pl.col('datetime').last().alias('datetime'),
            pl.col('close').last().alias('close'),
            pl.col('open').last().alias('open'),
            pl.col('high').last().alias('high'),
            pl.col('low').last().alias('low'),
            pl.col('volume').last().alias('volume'),
            pl.col('money').last().alias('money'),
        ]
        
        # -----------------------------------------------------------------
        # 简单特征：直接用 last() 或 tail() + 聚合
        # -----------------------------------------------------------------
        
        # 信息不对称代理（瞬时特征）
        agg_expressions.append(
            (pl.col('volume').last() / (pl.col('_volume_rolling_mean_60').last() + EPS) - 1)
            .alias('info_asymmetry_proxy')
        )
        
        # 价格聚集（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (1 / (1 + pl.col(f'_close_rolling_std_{period}').last() / (pl.col(f'_close_rolling_mean_{period}').last() + EPS)))
                .alias(f'price_clustering_{period}min')
            )
        
        # 成交量聚集（瞬时特征）
        """for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_volume_rolling_std_{period}').last() / (pl.col(f'_volume_rolling_mean_{period}').last() + EPS))
                .alias(f'volume_clustering_{period}min')
            )"""
        
        # 羊群加速度（瞬时特征，diff 用 tail 实现）
        for period in PERIODS:
            # herd_strength = volume / volume_ma * |ret|
            # 用 last 和 tail(2).first() 计算 diff
            herd_now = (pl.col('volume').last() / (pl.col(f'_volume_rolling_mean_{period}').last() + EPS) * 
                       pl.col('_close_ret_1').last().abs())
            herd_prev = (pl.col('volume').tail(2).first() / (pl.col(f'_volume_rolling_mean_{period}').tail(2).first() + EPS) * 
                        pl.col('_close_ret_1').tail(2).first().abs())
            agg_expressions.append(
                (herd_now - herd_prev)
                .alias(f'herding_acceleration_{period}min')
            )
        
        # 社会学习（瞬时特征，diff 用 tail 实现）
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_ret_rolling_std_{period}').last() - pl.col(f'_ret_rolling_std_{period}').tail(2).first()).abs()
                .alias(f'social_learning_{period}min')
            )
        
        # =====================================================================
        # Part 3: 执行聚合
        # =====================================================================
        """result_with_simple = df.group_by("code", maintain_order=True).agg(agg_expressions)
        result_with_complex = df_with_complex.group_by('code', maintain_order=True).last() 

        base_cols = set(df.columns) 
    
        simple_snap = result_with_simple

        complex_keep = ["code"] + [c for c in result_with_complex.columns if c != "code" and c not in base_cols]
        complex_snap = result_with_complex.select(complex_keep)

        result = complex_snap.join(simple_snap, on="code", how="left")"""

        # 动态构建 agg 表达式
        all_agg = agg_expressions.copy()  # simple 特征
        all_agg += [pl.col(c).last().alias(c) for c in complex_cols]  # complex 特征取 last
        result = df_with_complex.group_by("code", maintain_order=True).agg(all_agg)

        
        # 释放中间 DataFrame
        # del df_with_complex
        # gc.collect()
        
        return result


class MicrostructureProcessor(Processor):
    """
    市场微观结构因子处理器（实盘流式计算版）
    
    计算市场微观结构相关因子
    
    采用混合模式：
    - 大部分特征：group_by().agg() 高效计算
    - 少数复杂特征：保留 rolling().over() 方式
    """
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算市场微观结构因子"""
        
        # 确保按 code 和时间排序
        df = df.sort(['code', 'datetime'])
        
        # =====================================================================
        # Part 1: 复杂特征（保留 rolling，需要滑动窗口内每个点的计算）
        # =====================================================================
        complex_expressions = [
            # kyle_lambda（需要 rolling_mean）
            (pl.col('_close_ret_1').abs() / (pl.col('volume') + 1))
            .rolling_mean(window_size=60)
            .over('code')
            .alias('kyle_lambda'),
            
            # roll_effective_spread（需要 rolling_mean + shift）
            (-2 * pl.col('_close_ret_1') * pl.col('_close_ret_1').shift(1).over('code'))
            .rolling_mean(window_size=60)
            .over('code')
            .sqrt()
            .alias('roll_effective_spread'),
            
            # amihud_illiquidity（需要 rolling_mean）
            (pl.col('_close_ret_1').abs() / (pl.col('volume') * pl.col('close') + 1))
            .rolling_mean(window_size=60)
            .over('code')
            .alias('amihud_illiquidity'),
        ]
        
        # 价格效率（需要 rolling_mean）
        for period in PERIODS:
            complex_expressions.append(
                ((pl.col('close') - pl.col('open')).abs() / (pl.col('high') - pl.col('low') + EPS))
                .rolling_mean(window_size=period)
                .over('code')
                .alias(f'price_efficiency_{period}min')
            )
        
        # 价格冲击（需要 rolling_sum）
        for period in PERIODS:
            complex_expressions.append(
                (pl.col(f'_close_ret_{period}').abs() / 
                 (pl.col('volume').rolling_sum(window_size=period).over('code') / (pl.col('_volume_rolling_mean_241') + EPS) + EPS))
                .alias(f'price_impact_{period}min')
            )
        
        # 流动性消耗（需要 rolling_sum + rolling_mean）
        for period in [5, 15, 60]:
            complex_expressions.append(
                (pl.col('volume').rolling_sum(window_size=period).over('code') / 
                 (pl.col('volume').rolling_mean(window_size=period * 4).over('code').rolling_sum(window_size=period).over('code') + EPS))
                .alias(f'liquidity_consumption_{period}min')
            )
        
        # 订单流毒性（需要 rolling_std）
        for period in [5, 15, 60, 241]:
            price_change = pl.col('_close_ret_1')
            volume_weight = pl.col('volume') / (pl.col('_volume_rolling_mean_60') + EPS)
            complex_expressions.append(
                (price_change * volume_weight)
                .rolling_std(window_size=period)
                .over('code')
                .alias(f'order_flow_toxicity_{period}min')
            )
        
        # info_content（需要 rolling_mean）
        for period in [5, 15, 60, 241]:
            complex_expressions.append(
                (pl.col('_close_ret_1').abs() / (pl.col('volume') + 1))
                .rolling_mean(window_size=period)
                .over('code')
                .alias(f'info_content_{period}min')
            )
        
        # micro_volatility（需要 rolling_std）
        for period in [5, 15, 60, 241]:
            hilo_vol = (pl.col('high') - pl.col('low')) / (pl.col('open') + EPS)
            complex_expressions.append(
                hilo_vol
                .rolling_std(window_size=period)
                .over('code')
                .alias(f'micro_volatility_{period}min')
            )
        
        # 市场韧性（需要 rolling_quantile + rolling_mean + shift）
        for period in [5, 15, 60, 241]:
            large_move = pl.col('_close_ret_1').abs() > pl.col('_close_ret_1').abs().rolling_quantile(0.9, window_size=241).over('code')
            price_recovery = pl.col('_close_ret_1') * pl.col('_close_ret_1').shift(period).over('code')
            complex_expressions.append(
                pl.when(large_move.shift(period).over('code'))
                .then(price_recovery)
                .otherwise(0)
                .rolling_mean(window_size=241)
                .over('code')
                .alias(f'market_resilience_{period}min')
            )
        
        # 流动性风险（需要 rolling_std + rolling_mean）
        for period in PERIODS:
            depth_proxy = pl.col('volume') / ((pl.col('high') - pl.col('low')) + EPS)
            complex_expressions.append(
                (depth_proxy.rolling_std(window_size=period).over('code') / 
                 (depth_proxy.rolling_mean(window_size=period).over('code') + EPS))
                .alias(f'liquidity_risk_{period}min')
            )
        
        # 计算复杂特征
        df_with_complex = df.with_columns(complex_expressions)
        complex_cols = [expr.meta.output_name() for expr in complex_expressions]
        
        # =====================================================================
        # Part 2: 简单特征 + 合并（一次性 group_by）
        # =====================================================================
        agg_expressions = [
            # 基础列
            pl.col('datetime').last().alias('datetime'),
            pl.col('close').last().alias('close'),
            pl.col('open').last().alias('open'),
            pl.col('high').last().alias('high'),
            pl.col('low').last().alias('low'),
            pl.col('volume').last().alias('volume'),
            pl.col('money').last().alias('money'),
        ]
        
        # -----------------------------------------------------------------
        # 简单特征：直接用 last() 或 tail() + 聚合
        # -----------------------------------------------------------------
        
        # price_efficiency_var（瞬时特征）
        agg_expressions.append(
            pl.col('_ret_rolling_var_60').last()
            .alias('price_efficiency_var')
        )
        
        # volatility_ratio（瞬时特征）
        agg_expressions.append(
            (pl.col('_ret_rolling_var_60').last() / (pl.col('_ret_rolling_var_15').last() + EPS))
            .alias('volatility_ratio')
        )
        
        # depth_adjusted_spread（瞬时特征）
        agg_expressions.append(
            ((pl.col('high').last() - pl.col('low').last()) / (pl.col('close').last() + EPS) / 
             (pl.col('volume').last() / (pl.col('_volume_rolling_mean_60').last() + EPS) + EPS))
            .alias('depth_adjusted_spread')
        )
        
        # 执行风险（瞬时特征）
        for period in [5, 15, 60, 241]:
            agg_expressions.append(
                (pl.col(f'_ret_rolling_std_{period}').last() * 
                 (1 / (pl.col('volume').last() / (pl.col(f'_volume_rolling_mean_{period}').last() + EPS) + EPS)))
                .alias(f'execution_risk_{period}min')
            )
        
        
        # =====================================================================
        # Part 3: 执行聚合
        # =====================================================================
        """result_with_simple = df.group_by("code", maintain_order=True).agg(agg_expressions)
        result_with_complex = df_with_complex.group_by('code', maintain_order=True).last() 

        base_cols = set(df.columns) 
    
        simple_snap = result_with_simple

        complex_keep = ["code"] + [c for c in result_with_complex.columns if c != "code" and c not in base_cols]
        complex_snap = result_with_complex.select(complex_keep)

        result = complex_snap.join(simple_snap, on="code", how="left")"""

        # 动态构建 agg 表达式
        all_agg = agg_expressions.copy()  # simple 特征
        all_agg += [pl.col(c).last().alias(c) for c in complex_cols]  # complex 特征取 last
        result = df_with_complex.group_by("code", maintain_order=True).agg(all_agg)
        
        # 释放中间 DataFrame
        # del df_with_complex
        # gc.collect()
        
        return result


class SentimentCycleProcessor(Processor):
    """
    情绪周期因子处理器（实盘流式计算版）
    
    计算市场情绪周期相关因子
    
    采用混合模式：
    - 大部分特征：group_by().agg() 高效计算
    - 少数复杂特征：保留 rolling().over() 方式
    """
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算情绪周期因子"""
        
        # 确保按 code 和时间排序
        df = df.sort(['code', 'datetime'])
        
        # =====================================================================
        # Part 1: 复杂特征（保留 rolling，需要滑动窗口内每个点的计算）
        # =====================================================================
        complex_expressions = [
            # recency_bias（需要 ewm_mean）
            (pl.col('_close_ret_1').ewm_mean(alpha=0.1).over('code') / (pl.col('_ret_rolling_mean_60') + EPS))
            .alias('recency_bias'),
            
            # herd_behavior（需要 rolling_quantile + rolling_mean）
            ((pl.col('volume') > pl.col('volume').rolling_quantile(0.8, window_size=241).over('code')) & 
             (pl.col('_close_ret_1').abs() > pl.col('_close_ret_1').rolling_quantile(0.8, window_size=241).over('code')))
            .cast(pl.Float32)
            .rolling_mean(window_size=30)
            .over('code')
            .alias('herd_behavior'),
        ]
        
        # 情绪动量（需要 rolling_sum）
        for period in PERIODS:
            sentiment_proxy = pl.col('volume') * pl.col('_close_ret_1')
            complex_expressions.append(
                (sentiment_proxy.rolling_sum(window_size=period).over('code') / period)
                .alias(f'sentiment_momentum_{period}min')
            )
        
        # 累积情绪（需要 ewm_mean）
        for decay in [0.95, 0.98, 0.995]:
            daily_sentiment = pl.col('_close_ret_15') * pl.col('volume')
            complex_expressions.append(
                daily_sentiment.ewm_mean(alpha=1-decay)
                .over('code')
                .alias(f'cumulative_sentiment_{int(decay*100)}')
            )
        
        # 情绪离散度（需要 rolling_std）
        for period in PERIODS:
            high_low_ratio = (pl.col('high') - pl.col('low')) / (pl.col('close') + EPS)
            complex_expressions.append(
                high_low_ratio.rolling_std(window_size=period)
                .over('code')
                .alias(f'sentiment_dispersion_{period}min')
            )
        
        # 情绪持续性（需要 rolling_sum）
        for period in PERIODS:
            returns = pl.col('_close_ret_1')
            complex_expressions.append(
                ((returns > 0).cast(pl.Float32).rolling_sum(window_size=period).over('code') / period)
                .alias(f'sentiment_persistence_{period}min')
            )
        
        # 情绪振荡（需要 rolling_std + rolling_mean）
        for period in PERIODS:
            sentiment_proxy = pl.col('close').pct_change(1).over('code') * pl.col('volume')
            complex_expressions.append(
                (sentiment_proxy.rolling_std(window_size=period).over('code') / 
                 (sentiment_proxy.abs().rolling_mean(window_size=period).over('code') + EPS))
                .alias(f'sentiment_oscillation_{period}min')
            )
        
        # 情绪周期性（需要 rolling_mean + shift）
        for period in PERIODS:
            sentiment = pl.col('_close_ret_1') * pl.col('volume')
            autocorr_proxy = (sentiment * sentiment.shift(period//2).over('code')).rolling_mean(window_size=period).over('code')
            complex_expressions.append(
                autocorr_proxy
                .alias(f'sentiment_cyclical_{period}min')
            )
        
        # 情绪记忆（需要 ewm_mean）
        for half_life in PERIODS:
            decay_factor = np.exp(-np.log(2) / half_life)
            sentiment = pl.col('_close_ret_1') * pl.col('volume')
            complex_expressions.append(
                sentiment.ewm_mean(alpha=1-decay_factor)
                .over('code')
                .alias(f'sentiment_memory_{half_life}min')
            )
        
        # 风险偏好（需要 rolling_sum + when/then）
        for period in PERIODS:
            decline_volume = pl.when(pl.col('_close_ret_1') < 0).then(pl.col('volume')).otherwise(0)
            advance_volume = pl.when(pl.col('_close_ret_1') > 0).then(pl.col('volume')).otherwise(0)
            complex_expressions.append(
                (advance_volume.rolling_sum(window_size=period).over('code') / 
                 (decline_volume.rolling_sum(window_size=period).over('code') + EPS))
                .alias(f'risk_appetite_{period}min')
            )
        
        # 计算复杂特征
        df_with_complex = df.with_columns(complex_expressions)
        complex_cols = [expr.meta.output_name() for expr in complex_expressions]

        # =====================================================================
        # Part 2: 简单特征 + 合并（一次性 group_by）
        # =====================================================================
        agg_expressions = [
            # 基础列
            pl.col('datetime').last().alias('datetime'),
            pl.col('close').last().alias('close'),
            pl.col('open').last().alias('open'),
            pl.col('high').last().alias('high'),
            pl.col('low').last().alias('low'),
            pl.col('volume').last().alias('volume'),
            pl.col('money').last().alias('money'),
        ]
        
        # -----------------------------------------------------------------
        # 简单特征：直接用 last() 或 tail() + 聚合
        # -----------------------------------------------------------------
        
        # bull_regime（瞬时特征）
        agg_expressions.append(
            ((pl.col('close').last() > pl.col('_close_rolling_mean_241').last()) & 
             (pl.col('_close_rolling_mean_60').last() > pl.col('_close_rolling_mean_241').last()))
            .cast(pl.Float32)
            .alias('bull_regime')
        )
        
        # bear_regime（瞬时特征）
        agg_expressions.append(
            ((pl.col('close').last() < pl.col('_close_rolling_mean_241').last()) & 
             (pl.col('_close_rolling_mean_60').last() < pl.col('_close_rolling_mean_241').last()))
            .cast(pl.Float32)
            .alias('bear_regime')
        )
        
        # consolidation_regime（瞬时特征）
        agg_expressions.append(
            (pl.col('_close_rolling_mean_60').last() / (pl.col('_close_rolling_mean_241').last() + EPS) > 0.95)
            .cast(pl.Float32)
            .alias('consolidation_regime')
        )
        
        # anchoring_bias（瞬时特征）
        agg_expressions.append(
            (1 - pl.col('close').last() / (pl.col('_high_rolling_max_241').last() + EPS))
            .alias('anchoring_bias')
        )
        
        # 牛熊位置（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col('close').last() - pl.col(f'_close_rolling_min_{period}').last()) / 
                 (pl.col(f'_close_rolling_max_{period}').last() - pl.col(f'_close_rolling_min_{period}').last() + EPS))
                .alias(f'bull_bear_position_{period}min')
            )
        
        # fear_greed（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (pl.col(f'_close_ret_{period}').last() / (pl.col(f'_ret_rolling_std_{period}').last() + EPS))
                .alias(f'fear_greed_{period}min')
            )
        
        # 情绪反转（瞬时特征，用 tail 实现 diff）
        for period in PERIODS:
            # sentiment.diff() * sentiment < 0
            # diff = last - prev
            sentiment_last = pl.col(f'_close_ret_{period}').last()
            sentiment_prev = pl.col(f'_close_ret_{period}').tail(2).first()
            agg_expressions.append(
                (((sentiment_last - sentiment_prev) * sentiment_last) < 0)
                .cast(pl.Float32)
                .alias(f'sc_sentiment_reversal_{period}min')
            )
        
        # 投降（瞬时特征）
        for threshold in [-0.02, -0.04]:
            agg_expressions.append(
                ((pl.col('_close_ret_60').last() < threshold).cast(pl.Float32) * 
                 (pl.col('volume').last() > pl.col('_volume_rolling_mean_241').last() * 2).cast(pl.Float32))
                .alias(f'capitulation_{int(abs(threshold)*100)}pct')
            )
        
        # 狂喜（瞬时特征）
        for threshold in [-0.02, -0.04]:
            agg_expressions.append(
                ((pl.col('_close_ret_60').last() > threshold).cast(pl.Float32) * 
                 (pl.col('volume').last() > pl.col('_volume_rolling_mean_241').last() * 2).cast(pl.Float32))
                .alias(f'euphoria_{int(threshold*100)}pct')
            )
        
        # =====================================================================
        # Part 3: 执行聚合
        # =====================================================================
        """result_with_simple = df.group_by("code", maintain_order=True).agg(agg_expressions)
        result_with_complex = df_with_complex.group_by('code', maintain_order=True).last() 

        base_cols = set(df.columns) 
    
        simple_snap = result_with_simple

        complex_keep = ["code"] + [c for c in result_with_complex.columns if c != "code" and c not in base_cols]
        complex_snap = result_with_complex.select(complex_keep)

        result = complex_snap.join(simple_snap, on="code", how="left")"""

        # 动态构建 agg 表达式
        all_agg = agg_expressions.copy()  # simple 特征
        all_agg += [pl.col(c).last().alias(c) for c in complex_cols]  # complex 特征取 last
        result = df_with_complex.group_by("code", maintain_order=True).agg(all_agg)
        
        # 释放中间 DataFrame
        # del df_with_complex
        # gc.collect()
        
        # self.logger.info(f"Generated {len(result.columns)} sentiment cycle factors")
        return result


class AttentionProcessor(Processor):
    """
    市场注意力因子处理器（实盘流式计算版）
    
    计算市场注意力相关因子
    
    采用混合模式：
    - 大部分特征：group_by().agg() 高效计算
    - 少数复杂特征：保留 rolling().over() 方式
    """
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算市场注意力因子"""
        
        # 确保按 code 和时间排序
        df = df.sort(['code', 'datetime'])
        
        # =====================================================================
        # Part 1: 复杂特征（保留 rolling，需要滑动窗口内每个点的计算）
        # =====================================================================
        complex_expressions = [
            # low_attention_regime（需要 rolling_mean）
            (pl.col('volume') < pl.col('_volume_rolling_mean_15') * 0.8)
            .cast(pl.Float32)
            .rolling_mean(window_size=60)
            .over('code')
            .cast(pl.Float32)
            .alias('low_attention_regime'),
            
            # high_attention_regime（需要 rolling_mean）
            (pl.col('volume') > pl.col('_volume_rolling_mean_15') * 1.5)
            .cast(pl.Float32)
            .rolling_mean(window_size=60)
            .over('code')
            .cast(pl.Float32)
            .alias('high_attention_regime'),
            
            # attention_inflow（需要 rolling_sum）
            pl.max_horizontal([
                pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32), 
                pl.lit(0.0).cast(pl.Float32)
            ])
            .rolling_sum(window_size=60)
            .over('code')
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias('attention_inflow'),
            
            # attention_outflow（需要 rolling_sum）
            pl.max_horizontal([
                -pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32), 
                pl.lit(0.0).cast(pl.Float32)
            ])
            .rolling_sum(window_size=60)
            .over('code')
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias('attention_outflow'),
            
            # net_attention_flow（需要 rolling_sum）
            pl.col('volume')
            .pct_change()
            .over('code')
            .fill_nan(0.0)
            .replace([float('inf'), float('-inf')], 0.0)
            .cast(pl.Float32)
            .rolling_sum(window_size=60)
            .over('code')
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias('net_attention_flow'),
            
            # viral_attention（需要 rolling_sum + when/then）
            pl.when(pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32) > 0)
            .then(pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32) ** 2)
            .otherwise(0.0)
            .cast(pl.Float32)
            .rolling_sum(window_size=120)
            .over('code')
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias('viral_attention'),
            
            # cascade_attention（需要 rolling_sum）
            (pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32) > 0.1)
            .cast(pl.Float32)
            .rolling_sum(window_size=60)
            .over('code')
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias('cascade_attention'),
            
            # herd_attention（需要 rolling_mean + shift）
            (pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32) * 
             pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32).shift(1).over('code') > 0)
            .cast(pl.Float32)
            .rolling_mean(window_size=60)
            .over('code')
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias('herd_attention'),
        ]
        
        # 交易活跃度（需要 rolling_sum）
        for period in PERIODS:
            complex_expressions.append(
                (pl.col('volume').cast(pl.Float32).rolling_sum(window_size=period).over('code') / 
                 (pl.col('volume').cast(pl.Float32).rolling_sum(window_size=period * 3).over('code') + EPS))
                .cast(pl.Float32)
                .alias(f'trading_activity_{period}min')
            )
        
        # 注意力衰减（需要 ewm_mean + rolling_mean）
        for halflife in PERIODS:
            alpha = 1 - np.exp(-np.log(2) / halflife)
            complex_expressions.append(
                (pl.col('volume').cast(pl.Float32).ewm_mean(alpha=alpha).over('code') / 
                 (pl.col('volume').cast(pl.Float32).rolling_mean(window_size=halflife * 4).over('code') + EPS))
                .cast(pl.Float32)
                .alias(f'attention_decay_hl{halflife}min')
            )
        
        # 注意力聚焦（需要 rolling_std）
        for period in PERIODS:
            price_range = ((pl.col('high').cast(pl.Float32) - pl.col('low').cast(pl.Float32)) / 
                           (pl.col('close').cast(pl.Float32) + EPS)).cast(pl.Float32)
            complex_expressions.append(
                (1.0 / (1.0 + price_range.rolling_std(window_size=period).over('code')))
                .cast(pl.Float32)
                .alias(f'attention_focus_{period}min')
            )
        
        # 注意力强度（需要 rolling_mean）
        for period in PERIODS:
            price_change = pl.col('_close_ret_1').cast(pl.Float32).abs()
            volume_weight = (pl.col('volume').cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32)
            complex_expressions.append(
                (price_change * volume_weight)
                .rolling_mean(window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_intensity_{period}min')
            )
        
        # 信息冲击（需要 rolling_sum + rolling_mean）
        for period in PERIODS:
            vol_shock = (pl.col('volume').cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32) > 2
            price_shock = pl.col('_close_ret_1').cast(pl.Float32).abs() > pl.col('_close_ret_1').cast(pl.Float32).abs().rolling_mean(window_size=period).over('code') * 2
            complex_expressions.append(
                (vol_shock & price_shock)
                .cast(pl.Float32)
                .rolling_sum(window_size=60)
                .over('code')
                .cast(pl.Float32)
                .alias(f'information_shock_{period}min')
            )
        
        # 注意力持续性（需要 rolling_sum）
        for period in PERIODS:
            high_attention = pl.col('volume').cast(pl.Float32) > pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) * 1.5
            complex_expressions.append(
                high_attention
                .cast(pl.Float32)
                .rolling_sum(window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_persistence_{period}min')
            )
        
        # 注意力饱和（需要 rolling_quantile + rolling_mean）
        for period in PERIODS:
            vol_percentile = pl.col('volume').cast(pl.Float32).rolling_quantile(0.9, window_size=period).over('code')
            complex_expressions.append(
                (pl.col('volume').cast(pl.Float32) >= vol_percentile)
                .cast(pl.Float32)
                .rolling_mean(window_size=60)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_saturation_{period}min')
            )
        
        # 注意力动量（需要 rolling_mean）
        for period in PERIODS:
            vol_change = pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32)
            complex_expressions.append(
                vol_change
                .rolling_mean(window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_momentum_{period}min')
            )
        
        # 注意力波动率（需要 rolling_std）
        for period in PERIODS:
            vol_normalized = (pl.col('volume').cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32)
            complex_expressions.append(
                vol_normalized
                .rolling_std(window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_volatility_{period}min')
            )
        
        # 社交注意力（需要 rolling_corr）
        for period in PERIODS:
            price_change = pl.col('_close_ret_1').cast(pl.Float32)
            volume_change = pl.col('volume').pct_change().over('code').replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32)
            complex_expressions.append(
                pl.rolling_corr(price_change, volume_change, window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'social_attention_{period}min')
            )
        
        # 注意力回流（需要 rolling_sum + shift）
        for period in PERIODS:
            quiet_period = pl.col('volume').cast(pl.Float32) < pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) * 0.7
            vol_surge_after_quiet = quiet_period.shift(1).over('code') & (pl.col('volume').cast(pl.Float32) > pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) * 1.3)
            complex_expressions.append(
                vol_surge_after_quiet
                .cast(pl.Float32)
                .rolling_sum(window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_reflow_{period}min')
            )
        
        # 注意力离散度（需要 rolling_std）
        for period in PERIODS:
            price_range = ((pl.col('high').cast(pl.Float32) - pl.col('low').cast(pl.Float32)) / 
                           (pl.col('close').cast(pl.Float32) + EPS)).cast(pl.Float32)
            volume_concentration = (pl.col('volume').cast(pl.Float32) / (price_range + EPS)).cast(pl.Float32)
            complex_expressions.append(
                volume_concentration
                .rolling_std(window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_dispersion_{period}min')
            )
        
        # 预期注意力（需要 rolling_mean + diff）
        for period in PERIODS:
            vol_acceleration = pl.col('volume').pct_change().over('code').fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32).diff().over('code').fill_null(0.0)
            complex_expressions.append(
                vol_acceleration
                .rolling_mean(window_size=period)
                .over('code')
                .cast(pl.Float32)
                .alias(f'anticipated_attention_{period}min')
            )
        
        # 注意力回声（需要 rolling_sum + shift）
        for lag in PERIODS:
            vol_spike = pl.col('volume').cast(pl.Float32) > pl.col('_volume_rolling_mean_15').cast(pl.Float32) * 2
            delayed_reaction = vol_spike.shift(lag).over('code') & (pl.col('volume').cast(pl.Float32) > pl.col('_volume_rolling_mean_15').cast(pl.Float32) * 1.5)
            complex_expressions.append(
                delayed_reaction
                .cast(pl.Float32)
                .rolling_sum(window_size=120)
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_echo_lag{lag}min')
            )
        
        # 注意力疲劳（需要 rolling_mean + diff）
        for period in PERIODS:
            price_stimulus = pl.col('_close_ret_1').cast(pl.Float32).abs()
            volume_response = (pl.col('volume').cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32)
            fatigue_ratio = (volume_response / (price_stimulus + EPS)).cast(pl.Float32)
            complex_expressions.append(
                fatigue_ratio
                .rolling_mean(window_size=period)
                .over('code')
                .diff()
                .over('code')
                .cast(pl.Float32)
                .alias(f'attention_fatigue_{period}min')
            )
        
        # 注意力传染（需要 rolling_mean + shift）
        for period in PERIODS:
            vol_pct_change = (
                pl.col('volume')
                .pct_change()
                .over('code')
                .fill_nan(0.0)
                .replace([float('inf'), float('-inf')], 0.0)
                .cast(pl.Float32)
            )
            vol_change_corr = vol_pct_change * vol_pct_change.shift(1).over('code')
            complex_expressions.append(
                vol_change_corr
                .rolling_mean(window_size=period)
                .over('code')
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias(f'attention_contagion_{period}min')
            )
        
        # 注意力质量（需要 rolling_std）
        for period in PERIODS:
            price_efficiency = (1.0 / (1.0 + (pl.col('high').cast(pl.Float32) - pl.col('low').cast(pl.Float32)).rolling_std(window_size=period).over('code'))).cast(pl.Float32)
            volume_consistency = (1.0 / (1.0 + (pl.col(f'_volume_rolling_std_{period}').cast(pl.Float32) / 
                                                 (pl.col(f'_volume_rolling_mean_{period}').cast(pl.Float32) + EPS)).cast(pl.Float32))).cast(pl.Float32)
            complex_expressions.append(
                (price_efficiency * volume_consistency)
                .cast(pl.Float32)
                .alias(f'attention_quality_{period}min')
            )
        
        # 计算复杂特征
        df_with_complex = df.with_columns(complex_expressions)
        complex_cols = [expr.meta.output_name() for expr in complex_expressions]

        # =====================================================================
        # Part 2: 简单特征 + 合并（一次性 group_by）
        # =====================================================================
        agg_expressions = [
            # 基础列
            pl.col('datetime').last().alias('datetime'),
            pl.col('close').last().alias('close'),
            pl.col('open').last().alias('open'),
            pl.col('high').last().alias('high'),
            pl.col('low').last().alias('low'),
            pl.col('volume').last().alias('volume'),
            pl.col('money').last().alias('money'),
        ]
        
        # -----------------------------------------------------------------
        # 简单特征：直接用 last() 或 tail() + 聚合
        # -----------------------------------------------------------------
        
        # volatile_attention_regime（瞬时特征）
        agg_expressions.append(
            (pl.col('_volume_rolling_std_15').last().cast(pl.Float32) / 
             (pl.col('_volume_rolling_mean_15').last().cast(pl.Float32) + EPS))
            .cast(pl.Float32)
            .alias('volatile_attention_regime')
        )
        
        # 价格记忆（瞬时特征）
        for lag in PERIODS:
            agg_expressions.append(
                (pl.col('_close_ret_1').last().cast(pl.Float32) * pl.col(f'_close_ret_{lag}').last().cast(pl.Float32))
                .cast(pl.Float32)
                .alias(f'price_memory_lag{lag}min')
            )
        
        # 异常注意力（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col('volume').last().cast(pl.Float32) - pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32)) / 
                 (pl.col(f'_volume_rolling_std_{period}').last().cast(pl.Float32) + EPS))
                .cast(pl.Float32)
                .alias(f'abnormal_attention_{period}min')
            )
        
        # 价格新闻价值（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col('close').last() == pl.col(f'_close_rolling_max_{period}').last()).cast(pl.Float32) +
                 (pl.col('close').last() == pl.col(f'_close_rolling_min_{period}').last()).cast(pl.Float32))
                .cast(pl.Float32)
                .alias(f'price_newsworthiness_{period}min')
            )
        
        # 事件驱动（瞬时特征）
        for threshold in [2, 3, 4]:
            agg_expressions.append(
                ((pl.col('volume').last().cast(pl.Float32) / (pl.col('_volume_rolling_mean_15').last().cast(pl.Float32) + EPS)) > threshold)
                .cast(pl.Float32)
                .alias(f'event_driven_{int(threshold*10)}x')
            )
        
        # 注意力转移（瞬时特征）
        for period in PERIODS:
            vol_change = (pl.col('volume').pct_change(period).last().fill_nan(0.0).replace([float('inf'), float('-inf')], 0.0).cast(pl.Float32))
            price_change = pl.col(f'_close_ret_{period}').last().cast(pl.Float32).abs()
            agg_expressions.append(
                (vol_change / (price_change + EPS))
                .cast(pl.Float32)
                .alias(f'attention_shift_{period}min')
            )
        
        # 焦点时刻（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (((pl.col('close').last() == pl.col(f'_close_rolling_max_{period}').last()) | 
                  (pl.col('close').last() == pl.col(f'_close_rolling_min_{period}').last())).cast(pl.Float32) * 
                 (pl.col('volume').last().cast(pl.Float32) / (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + EPS)))
                .cast(pl.Float32)
                .alias(f'focal_moments_{period}min')
            )
        
        # 惊喜因子（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                ((pl.col('close').last().cast(pl.Float32) - pl.col(f'_close_rolling_mean_{period}').last().cast(pl.Float32)).abs() / 
                 (pl.col(f'_close_rolling_mean_{period}').last().cast(pl.Float32) + EPS))
                .cast(pl.Float32)
                .alias(f'surprise_factor_{period}min')
            )
        
        # 突破注意力（瞬时特征）
        for period in PERIODS:
            agg_expressions.append(
                (pl.col('volume').last().cast(pl.Float32) > 
                 (pl.col(f'_volume_rolling_mean_{period}').last().cast(pl.Float32) + 
                  pl.col(f'_volume_rolling_std_{period}').last().cast(pl.Float32) * 3))
                .cast(pl.Float32)
                .alias(f'breaking_attention_{period}min')
            )
        
        # =====================================================================
        # Part 3: 执行聚合
        # =====================================================================
        """result_with_simple = df.group_by("code", maintain_order=True).agg(agg_expressions)
        result_with_complex = df_with_complex.group_by('code', maintain_order=True).last() 

        base_cols = set(df.columns) 
    
        simple_snap = result_with_simple

        complex_keep = ["code"] + [c for c in result_with_complex.columns if c != "code" and c not in base_cols]
        complex_snap = result_with_complex.select(complex_keep)

        result = complex_snap.join(simple_snap, on="code", how="left")"""

        # 动态构建 agg 表达式
        all_agg = agg_expressions.copy()  # simple 特征
        all_agg += [pl.col(c).last().alias(c) for c in complex_cols]  # complex 特征取 last
        result = df_with_complex.group_by("code", maintain_order=True).agg(all_agg)
        
        # 释放中间 DataFrame
        # del df_with_complex
        # gc.collect()
        
        return result