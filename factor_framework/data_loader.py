"""
统一数据获取接口

这是设计文档中"技术挑战与解决方案"部分提到的统一数据查询接口。
引擎通过这个接口自动从不同的数据库获取数据，根据频率配置路由到相应的数据源。

支持的数据源：
- 日频数据：从 data/processed_data/日频数据/ 目录加载
- 30分钟频数据：从 data/processed_data/三十分钟频数据/ 目录加载
- 原始分钟数据：从 data/raw_data/wind_etf_data/ 目录加载

这个接口完全对应设计文档中的要求，实现了多频率数据的统一访问。
"""

import os
import pandas as pd
import numpy as np
from typing import List, Union, Optional
from datetime import datetime, timedelta
import glob
from .config import FrequencyConfig


def get_data(assets: Union[List[str], str],
             start_date: str,
             end_date: str,
             frequency: str,
             data_root: str = "data") -> pd.DataFrame:
    """
    统一数据获取接口

    按照设计文档的要求，这个函数提供统一接口自动从不同数据库获取数据。
    根据频率参数自动路由到相应的数据源。

    Args:
        assets: 资产代码列表或单个资产代码
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        frequency: 数据频率 ('daily', 'minute', '5min', '30min')
        data_root: 数据根目录路径

    Returns:
        包含OHLCV数据的DataFrame，格式统一为：
        - datetime: 时间戳
        - code: 资产代码
        - name: 资产名称
        - open, high, low, close: OHLC价格
        - volume: 成交量
        - amount: 成交额

    Raises:
        ValueError: 不支持的频率类型
        FileNotFoundError: 找不到对应的数据文件
    """
    # 标准化资产代码列表
    if isinstance(assets, str):
        assets = [assets]

    # 根据频率路由到不同的数据获取函数
    if frequency == 'daily':
        return _fetch_from_daily_db(assets, start_date, end_date, data_root)
    elif frequency in ['30min', '30T']:
        return _fetch_from_30min_db(assets, start_date, end_date, data_root)
    elif frequency in ['minute', '1min', '1T']:
        return _fetch_from_minute_db(assets, start_date, end_date, data_root)
    elif frequency in ['5min', '5T']:
        # 5分钟数据可以从分钟数据重采样得到
        minute_data = _fetch_from_minute_db(assets, start_date, end_date, data_root)
        return _resample_to_frequency(minute_data, '5min')
    else:
        raise ValueError(f"不支持的频率: {frequency}")


def _fetch_from_daily_db(assets: List[str],
                        start_date: str,
                        end_date: str,
                        data_root: str) -> pd.DataFrame:
    """
    从日频数据库获取数据

    数据路径：data/processed_data/日频数据/{year}/{etf_name}_daily.csv
    """
    daily_data_dir = os.path.join(data_root, "processed_data", "日频数据")

    if not os.path.exists(daily_data_dir):
        raise FileNotFoundError(f"日频数据目录不存在: {daily_data_dir}")

    # 解析日期范围
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 获取涉及的年份
    years = list(range(start_dt.year, end_dt.year + 1))

    all_data = []

    for year in years:
        year_dir = os.path.join(daily_data_dir, str(year))
        if not os.path.exists(year_dir):
            continue

        # 查找匹配的文件
        for asset in assets:
            # 尝试不同的文件名模式
            patterns = [
                f"*{asset}*_daily.csv",
                f"{asset}_daily.csv",
                f"*{asset.replace('.', '_')}*_daily.csv"
            ]

            file_found = False
            for pattern in patterns:
                files = glob.glob(os.path.join(year_dir, pattern))
                if files:
                    file_path = files[0]  # 取第一个匹配的文件
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                        df['datetime'] = pd.to_datetime(df['date'])
                        df['code'] = asset

                        # 筛选日期范围
                        mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
                        df_filtered = df.loc[mask]

                        if not df_filtered.empty:
                            all_data.append(df_filtered)

                        file_found = True
                        break
                    except Exception as e:
                        print(f"读取文件失败 {file_path}: {e}")

            if not file_found:
                print(f"警告：未找到资产 {asset} 在 {year} 年的日频数据")

    if not all_data:
        return pd.DataFrame()

    # 合并所有数据
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['code', 'datetime'])

    # 标准化列名
    result = _standardize_columns(result)

    return result


def _fetch_from_30min_db(assets: List[str],
                        start_date: str,
                        end_date: str,
                        data_root: str) -> pd.DataFrame:
    """
    从30分钟频数据库获取数据

    数据路径：data/processed_data/三十分钟频数据/{year}/{etf_name}_30min.csv
    """
    min30_data_dir = os.path.join(data_root, "processed_data", "三十分钟频数据")

    if not os.path.exists(min30_data_dir):
        raise FileNotFoundError(f"30分钟频数据目录不存在: {min30_data_dir}")

    # 解析日期范围
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 获取涉及的年份
    years = list(range(start_dt.year, end_dt.year + 1))

    all_data = []

    for year in years:
        year_dir = os.path.join(min30_data_dir, str(year))
        if not os.path.exists(year_dir):
            continue

        # 查找匹配的文件
        for asset in assets:
            patterns = [
                f"*{asset}*_30min.csv",
                f"{asset}_30min.csv",
                f"*{asset.replace('.', '_')}*_30min.csv"
            ]

            file_found = False
            for pattern in patterns:
                files = glob.glob(os.path.join(year_dir, pattern))
                if files:
                    file_path = files[0]
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df['code'] = asset

                        # 筛选日期范围
                        mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
                        df_filtered = df.loc[mask]

                        if not df_filtered.empty:
                            all_data.append(df_filtered)

                        file_found = True
                        break
                    except Exception as e:
                        print(f"读取文件失败 {file_path}: {e}")

            if not file_found:
                print(f"警告：未找到资产 {asset} 在 {year} 年的30分钟频数据")

    if not all_data:
        return pd.DataFrame()

    # 合并所有数据
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['code', 'datetime'])

    # 标准化列名
    result = _standardize_columns(result)

    return result


def _fetch_from_minute_db(assets: List[str],
                         start_date: str,
                         end_date: str,
                         data_root: str) -> pd.DataFrame:
    """
    从分钟级原始数据库获取数据

    数据路径：data/raw_data/wind_etf_data/{year}/{code}_{exchange}_{year}.csv
    """
    raw_data_dir = os.path.join(data_root, "raw_data", "wind_etf_data")

    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"原始数据目录不存在: {raw_data_dir}")

    # 解析日期范围
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 获取涉及的年份
    years = list(range(start_dt.year, end_dt.year + 1))

    all_data = []

    for year in years:
        year_dir = os.path.join(raw_data_dir, str(year))
        if not os.path.exists(year_dir):
            continue

        # 查找匹配的文件
        for asset in assets:
            # 尝试不同的文件名模式
            patterns = [
                f"{asset}_{year}.csv",
                f"*{asset}*_{year}.csv",
                f"{asset.replace('.', '_')}*_{year}.csv"
            ]

            file_found = False
            for pattern in patterns:
                files = glob.glob(os.path.join(year_dir, pattern))
                if files:
                    file_path = files[0]
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                        df['datetime'] = pd.to_datetime(df['datetime'])

                        # 筛选日期范围
                        mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
                        df_filtered = df.loc[mask]

                        if not df_filtered.empty:
                            all_data.append(df_filtered)

                        file_found = True
                        break
                    except Exception as e:
                        print(f"读取文件失败 {file_path}: {e}")

            if not file_found:
                print(f"警告：未找到资产 {asset} 在 {year} 年的分钟级数据")

    if not all_data:
        return pd.DataFrame()

    # 合并所有数据
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['code', 'datetime'])

    # 标准化列名
    result = _standardize_columns(result)

    return result


def _resample_to_frequency(data: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """
    将数据重采样到目标频率

    Args:
        data: 原始数据
        target_freq: 目标频率 ('5min', '10min', 'H' 等)

    Returns:
        重采样后的数据
    """
    if data.empty:
        return data

    resampled_data = []

    for code in data['code'].unique():
        asset_data = data[data['code'] == code].copy()
        asset_data = asset_data.set_index('datetime')

        # 重采样OHLCV数据
        resampled = asset_data.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum',
            'code': 'first',
            'name': 'first'
        }).reset_index()

        # 删除没有数据的时间段
        resampled = resampled.dropna(subset=['close'])
        resampled_data.append(resampled)

    if resampled_data:
        result = pd.concat(resampled_data, ignore_index=True)
        return result.sort_values(['code', 'datetime'])
    else:
        return pd.DataFrame()


def _standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    标准化数据列名和格式

    确保返回的数据具有统一的列名和数据类型
    """
    # 必需的列
    required_columns = ['datetime', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']

    # 检查并添加缺失的列
    for col in required_columns:
        if col not in data.columns:
            if col in ['open', 'high', 'low', 'close']:
                data[col] = np.nan
            elif col in ['volume', 'amount']:
                data[col] = 0.0
            elif col == 'code':
                data[col] = 'UNKNOWN'

    # 添加name列（如果不存在）
    if 'name' not in data.columns:
        data['name'] = data['code']

    # 确保数据类型正确
    try:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['open'] = pd.to_numeric(data['open'], errors='coerce')
        data['high'] = pd.to_numeric(data['high'], errors='coerce')
        data['low'] = pd.to_numeric(data['low'], errors='coerce')
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0)
        data['amount'] = pd.to_numeric(data['amount'], errors='coerce').fillna(0)
    except Exception as e:
        print(f"数据类型转换警告: {e}")

    return data


def get_available_assets(frequency: str = 'daily',
                        year: Optional[int] = None,
                        data_root: str = "data") -> List[str]:
    """
    获取可用的资产列表

    Args:
        frequency: 数据频率
        year: 年份（可选）
        data_root: 数据根目录

    Returns:
        可用资产代码列表
    """
    assets = set()

    if frequency == 'daily':
        data_dir = os.path.join(data_root, "processed_data", "日频数据")
        pattern = "*_daily.csv"
    elif frequency in ['30min', '30T']:
        data_dir = os.path.join(data_root, "processed_data", "三十分钟频数据")
        pattern = "*_30min.csv"
    elif frequency in ['minute', '1min']:
        data_dir = os.path.join(data_root, "raw_data", "wind_etf_data")
        pattern = "*.csv"
    else:
        return []

    if not os.path.exists(data_dir):
        return []

    # 搜索指定年份或所有年份
    if year is not None:
        years = [year]
    else:
        years = [d for d in os.listdir(data_dir) if d.isdigit()]

    for year_str in years:
        year_dir = os.path.join(data_dir, str(year_str))
        if os.path.exists(year_dir):
            files = glob.glob(os.path.join(year_dir, pattern))
            for file_path in files:
                filename = os.path.basename(file_path)
                # 从文件名提取资产代码
                if frequency == 'daily':
                    asset_code = filename.replace('_daily.csv', '')
                elif frequency in ['30min', '30T']:
                    asset_code = filename.replace('_30min.csv', '')
                else:
                    # 分钟数据：格式通常是 {code}_{exchange}_{year}.csv
                    parts = filename.replace('.csv', '').split('_')
                    if len(parts) >= 2:
                        asset_code = f"{parts[0]}.{parts[1]}"
                    else:
                        asset_code = parts[0]

                assets.add(asset_code)

    return sorted(list(assets))


def validate_data_availability(assets: List[str],
                              start_date: str,
                              end_date: str,
                              frequency: str,
                              data_root: str = "data") -> dict:
    """
    验证数据可用性

    Args:
        assets: 资产代码列表
        start_date: 开始日期
        end_date: 结束日期
        frequency: 数据频率
        data_root: 数据根目录

    Returns:
        数据可用性报告字典
    """
    report = {
        'available_assets': [],
        'missing_assets': [],
        'data_summary': {}
    }

    for asset in assets:
        try:
            data = get_data([asset], start_date, end_date, frequency, data_root)
            if not data.empty:
                report['available_assets'].append(asset)
                report['data_summary'][asset] = {
                    'records': len(data),
                    'start_date': data['datetime'].min().strftime('%Y-%m-%d'),
                    'end_date': data['datetime'].max().strftime('%Y-%m-%d')
                }
            else:
                report['missing_assets'].append(asset)
        except Exception as e:
            report['missing_assets'].append(asset)
            print(f"资产 {asset} 数据检查失败: {e}")

    return report