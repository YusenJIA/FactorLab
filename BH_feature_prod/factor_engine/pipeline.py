# -*- coding: utf-8 -*-
"""
因子计算流水线和数据处理器
========================

提供完整的因子计算工作流，包括：
- FeatureDataHandler: 数据处理器
- FeaturePipeline: 因子计算流水线
"""

import os
import logging
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from pathlib import Path

import polars as pl

from .base import (
    Processor, Config, get_logger,
    DropnaProcessor, FillnaProcessor, CSZScoreNormProcessor,
    MixedFeatureSplitter, NonlinearTransformer,
    PRECOMPUTED_FEATURES, MIXED_FEATURE_BASES
)
from .behavior_features import (
    RoundNumberProcessor, FOMOFUDProcessor,
    RetailPatternProcessor, HerdingProcessor,
    MicrostructureProcessor, SentimentCycleProcessor, AttentionProcessor
    )
from .factor_engine import BaseFeatureProcessor


# ============================================================================
# FeatureDataHandler 
# ============================================================================

class FeatureDataHandler:
    """
    特征数据处理器
    
    参考 Qlib 的 DataHandlerLP 设计，支持：
    - infer_processors: 推理时使用的处理器
    - learn_processors: 训练时使用的处理器
    - shared_processors: 共享处理器
    
    Usage:
        handler = FeatureDataHandler(
            infer_processors=[
                RoundNumberProcessor(),
                FOMOFUDProcessor(),
                CSZScoreNormProcessor(group_col="datetime"),
                FillnaProcessor(fill_value=0),
            ],
            learn_processors=[
                DropnaProcessor(),
            ]
        )
        handler.fit(train_df)
        infer_df = handler.fetch(test_df, col_set="infer")
        learn_df = handler.fetch(train_df, col_set="learn")
    """
    
    # 数据类型常量
    DK_R = "raw"      # 原始数据
    DK_I = "infer"    # 推理数据
    DK_L = "learn"    # 学习数据
    
    def __init__(
        self,
        infer_processors: List[Processor] = None,
        learn_processors: List[Processor] = None,
        shared_processors: List[Processor] = None,
        fit_start_time: Optional[str] = None,
        fit_end_time: Optional[str] = None,
    ):
        """
        初始化数据处理器
        
        Parameters
        ----------
        infer_processors : List[Processor]
            推理时使用的处理器列表
        learn_processors : List[Processor]
            训练时使用的处理器列表
        shared_processors : List[Processor]
            共享处理器列表（先于 infer/learn 处理器执行）
        fit_start_time : str
            fit 时的起始时间
        fit_end_time : str
            fit 时的结束时间
        """
        self.infer_processors = infer_processors or []
        self.learn_processors = learn_processors or []
        self.shared_processors = shared_processors or []
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        
        self.logger = get_logger("FeatureDataHandler", "handler.log")
        self._fitted = False
        
        # 配置处理器
        self._config_processors()
    
    def _config_processors(self):
        """配置处理器参数"""
        all_processors = self.shared_processors + self.infer_processors + self.learn_processors
        for proc in all_processors:
            proc.config(
                fit_start_time=self.fit_start_time,
                fit_end_time=self.fit_end_time
            )
    
    def fit(self, df: pl.DataFrame) -> "FeatureDataHandler":
        """
        在训练数据上 fit 所有处理器
        
        Parameters
        ----------
        df : pl.DataFrame
            训练数据
        """
        self.logger.info(f"Fitting handlers on data with {len(df)} rows")
        
        # Fit 共享处理器
        current_df = df
        for proc in self.shared_processors:
            proc.fit(current_df)
            current_df = proc(current_df)
        
        # Fit 推理处理器
        infer_df = current_df
        for proc in self.infer_processors:
            proc.fit(infer_df)
            infer_df = proc(infer_df)
        
        # Fit 学习处理器
        learn_df = infer_df
        for proc in self.learn_processors:
            proc.fit(learn_df)
        
        self._fitted = True
        self.logger.info("Fitting completed")
        return self
    
    def fetch(
        self,
        df: pl.DataFrame,
        col_set: str = "infer",
    ) -> pl.DataFrame:
        """
        获取处理后的数据
        
        Parameters
        ----------
        df : pl.DataFrame
            输入数据
        col_set : str
            数据类型: "raw", "infer", "learn"
        """
        if not self._fitted:
            self.logger.warning("Handler not fitted, fitting on input data")
            self.fit(df)
        
        # 应用共享处理器
        current_df = df
        for proc in self.shared_processors:
            current_df = proc(current_df)
        
        if col_set == self.DK_R:
            return current_df
        
        # 应用推理处理器
        for proc in self.infer_processors:
            current_df = proc(current_df)
        
        if col_set == self.DK_I:
            return current_df
        
        # 应用学习处理器
        for proc in self.learn_processors:
            current_df = proc(current_df)
        
        return current_df
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名"""
        names = []
        for proc in self.shared_processors + self.infer_processors:
            names.extend(proc.get_feature_names())
        return names


# ============================================================================
# FeaturePipeline
# ============================================================================

class FeaturePipeline:
    """
    因子计算流水线
    
    提供完整的因子计算工作流：
    1. 数据加载
    2. 因子计算
    3. 标准化处理
    4. 数据保存
    
    Usage:
        pipeline = FeaturePipeline(
            input_dir="/data/raw",
            output_dir="/data/features",
            processors=[
                RoundNumberProcessor(),
                FOMOFUDProcessor(),
                RetailPatternProcessor(),
            ],
        )
        pipeline.run(start_date="2024-01-01", end_date="2024-12-31")
    """
    
    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        feature_type: str = "round_number",
        processors: List[Processor] = None,
        fillna_value: float = 0.0,
        log_file: str = "pipeline.log",
    ):
        """
        初始化流水线
        
        Parameters
        ----------
        input_dir : str
            输入数据目录
        output_dir : str
            输出数据目录
        processors : List[Processor]
            因子处理器列表
        fillna_value : float
            缺失值填充值
        log_file : str
            日志文件名
        """
        self.input_dir = input_dir or Config.INPUT_DIR
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.processors = processors or self._get_default_processors()
        self.fillna_value = fillna_value
        self.feature_type = feature_type
        
        self.logger = get_logger("FeaturePipeline", log_file)
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_default_processors(self) -> List[Processor]:
        """获取默认处理器列表"""
        return [
            RoundNumberProcessor(),
            FOMOFUDProcessor(),
            RetailPatternProcessor(),
            HerdingProcessor(),
            MicrostructureProcessor(),
            SentimentCycleProcessor(),
            AttentionProcessor(),
        ]
    
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        执行因子计算
        
        Parameters
        ----------
        df : pl.DataFrame
            输入数据
        """
        self.logger.info(f"Starting feature processing with {len(self.processors)} processors")
        
        # 新代码 使用BaseFeatureProcessor一次性运行
        feature_processor = BaseFeatureProcessor(processors=self.processors)
        feature_processor.fit(df)
        result_df = feature_processor(df)

        self.logger.info(f"Feature processing completed. Total columns: {len(result_df.columns)}")
        
        return result_df
    
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        标准化处理
        
        Parameters
        ----------
        df : pl.DataFrame
            输入数据
        """
        
        if self.feature_type not in MIXED_FEATURE_BASES:
            raise ValueError(f"Unsupported feature_type for normalization: {self.feature_type}")

        self.logger.info(f"Normalizing with {self.feature_type}")
        
        # Step 1: 混合特征拆分

        splitter = MixedFeatureSplitter(processor_name=self.feature_type)
        df = splitter(df)
        
        # Step 2: 非线性变换
        transformer = NonlinearTransformer(
            group_col='datetime',
            exclude_cols=['datetime', 'code'],
            transforms=['sqrt', 'cubic', 'exp']
        )
        df = transformer(df)
        
        return df
    
    def fillna(self, df: pl.DataFrame) -> pl.DataFrame:
        """填充缺失值"""
        # self.logger.info(f"Filling NA with {self.fillna_value}")
        return df.fill_null(self.fillna_value).fill_nan(self.fillna_value)
    
    def save(
        self,
        df: pl.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        file_format: str = "parquet",
    ) -> list[str]:
        """
        保存数据（按日期分区）
        
        Parameters
        ----------
        df : pl.DataFrame
            要保存的数据（必须包含 'datetime' 列）
        start_date : str, optional
            起始日期 (格式: YYYY-MM-DD)，过滤掉此日期之前的数据
        end_date : str, optional
            结束日期 (格式: YYYY-MM-DD)，过滤掉此日期之后的数据
        file_format : str
            文件格式: "parquet" or "csv"
        
        Returns
        -------
        list[str]
            保存的文件路径列表
        """
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must contain 'datetime' column for partitioning")
        
        # 确保输出目录存在
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取日期
        df = df.with_columns(
            pl.col('datetime').dt.date().alias('_date')
        )
        
        # 过滤日期范围，排除 warmup 数据
        if start_date:
            df = df.filter(pl.col('_date') >= pl.lit(start_date).str.to_date())
        if end_date:
            df = df.filter(pl.col('_date') <= pl.lit(end_date).str.to_date())
        
        if df.is_empty():
            self.logger.warning(f"No data to save between {start_date} and {end_date}")
            return []
        
        
        saved_files = []
        for date_val in df['_date'].unique().sort():
            # 过滤当天数据
            daily_df = df.filter(pl.col('_date') == date_val).drop('_date')
            
            # 生成文件名：YYYY-MM-DD.parquet
            file_name = f"{date_val}.{file_format}"
            file_path = output_dir / file_name
            
            # 保存
            if file_format == "parquet":
                daily_df.write_parquet(file_path)
            elif file_format == "csv":
                daily_df.write_csv(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            saved_files.append(str(file_path))
        
        self.logger.info(f"Saved {len(saved_files)} files to {output_dir} (range: {start_date} to {end_date})")
        self.logger.info(f"Total {len(df)} rows with {len(df.columns)} columns")

        return saved_files
            
    def run(
        self,
        df,
        normalize: bool = False,
        fillna: bool = True,
    ) -> pl.DataFrame:
        """
        运行完整的流水线
        
        Parameters
        ----------
        normalize : bool
            是否标准化
        fillna : bool
            是否填充缺失值
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Feature Pipeline")
        self.logger.info("=" * 60)
        start_time = datetime.now()
        
        # 1. 计算因子
        df_feat = self.process(df)
        
        # 2. 标准化
        if normalize:
            df_feat = self.normalize(df_feat)
        
        # 3. 填充缺失值
        if fillna:
            df_feat = self.fillna(df_feat)
        
        # 4. 删除预计算列
        # df_feat = df_feat.drop(PRECOMPUTED_FEATURES)
        
        elapsed = datetime.now() - start_time
        self.logger.info(f"Pipeline completed in {elapsed}")
        self.logger.info(f"Final dataset: {len(df_feat)} rows, {len(df_feat.columns)} columns")
        
        return df_feat


# ============================================================================
# 预定义的 Handler 配置
# ============================================================================

def create_default_handler(
    fit_start_time: str = None,
    fit_end_time: str = None,
    norm_method: str = "zscore",
) -> FeatureDataHandler:
    """
    创建默认的数据处理器
    
    Parameters
    ----------
    fit_start_time : str
        fit 起始时间
    fit_end_time : str
        fit 结束时间
    norm_method : str
        标准化方法: "zscore", "robust", "minmax", "rank"
    """
    # 选择标准化处理器
    norm_processor = None
    
    # 创建处理器
    infer_processors = [
        RoundNumberProcessor(),
        FOMOFUDProcessor(),
        RetailPatternProcessor(),
        HerdingProcessor(),
        MicrostructureProcessor(),
        SentimentCycleProcessor(),
        AttentionProcessor(),
    ]
    
    if norm_processor:
        infer_processors.append(norm_processor)
    
    infer_processors.append(FillnaProcessor(fill_value=0))
    
    learn_processors = [
        DropnaProcessor(),
    ]
    
    return FeatureDataHandler(
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
    )
