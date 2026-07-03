"""
Realtime A-share feature engine.

Current-day bars are read from kbar CSV files first. If a CSV is missing,
incomplete, or late beyond the configured tolerance, the engine falls back to
the JQData API. Both instant and backfill modes use the same fallback path.
"""

import time
import logging
import os
import glob
from datetime import datetime, timedelta
from typing import Optional

import polars as pl

import sys
PYTHONPATH_ROOT = os.environ.get("ASHARE_PYTHONPATH_ROOT", "/home/yusen")
if PYTHONPATH_ROOT and PYTHONPATH_ROOT not in sys.path:
    sys.path.insert(0, PYTHONPATH_ROOT)

from factor_engine import FeaturePipeline
from dataloader.jqdata import (
    get_price,
    get_open_price,
    get_all_securities,
    get_call_auction,
    get_trade_days,
)


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("realtime_updater")


# ============================================================
# 配置区
# ============================================================
class Config:
    # 交易日期
    TRADE_DATE = os.environ.get("TRADE_DATE", "2026-06-15")

    # Project paths. /data paths stay as production defaults; /home/yusen can be
    # overridden for local/dev deployments.
    FEATURE_HOME = os.environ.get("ASHARE_FEATURE_HOME", "/home/yusen/ashare_feature")

    # --- 主通道：CSV 文件落盘 ---
    KBAR_CSV_ROOT = os.environ.get("ASHARE_KBAR_CSV_ROOT", "/data/shenrun/dump_1m_kbar")
    KBAR_CSV_DIR = os.environ.get("ASHARE_KBAR_CSV_DIR", os.path.join(KBAR_CSV_ROOT, TRADE_DATE))

    # 输出文件路径
    OUTPUT_DIR = os.environ.get("ASHARE_OUTPUT_DIR", "/data/BH")

    # --- 主通道 tolerance ---
    # CSV 在 tolerance_time 内未就绪 → 切聚宽 API
    CSV_TOLERANCE_SEC = 6
    CSV_POLL_INTERVAL = 0.5    # CSV 轮询间隔

    # --- 备通道 (聚宽 API) 参数 ---
    JQ_MAX_RETRY = 30          # 聚宽 API 单 bar 最多重试次数
    JQ_RETRY_INTERVAL = 1      # 聚宽 API 重试间隔
    JQ_FETCH_WINDOW_MINUTES = 3  # 聚宽拉取窗口 (expected_time 前 N 分钟)

    # 主循环 polling offset: CSV 在每分钟第 1 秒落盘
    POLL_OFFSET_SEC = 1

    # 交易时段
    MORNING_START = "09:30"
    MORNING_END = "11:30"
    AFTERNOON_START = "13:01"
    AFTERNOON_END = "15:00"


# ============================================================
# 通用过滤：去除 '2'/'9' 开头的 symbol
# ============================================================
def _filter_symbol_prefix(df: pl.DataFrame, symbol_col: str = "symbol") -> pl.DataFrame:
    """统一过滤掉 symbol 以 '2' 或 '9' 开头的行 (B股/科创板等)。"""
    return df.filter(
        ~pl.col(symbol_col).str.starts_with("2") &
        ~pl.col(symbol_col).str.starts_with("9")
    )


# ============================================================
# 1. 集合竞价数据加载
# ============================================================
def load_auction_data_today(symbol_arr: list[str]) -> pl.DataFrame:
    """通过 get_open_price 读取当天集合竞价数据，带重试。"""
    N = len(symbol_arr)
    threshold = int(N * 0.95)

    df = None
    for attempt in range(1, 31):
        pdf = get_open_price(symbol_arr)
        df = pl.from_pandas(_safe_pandas_convert(pdf))
        count = len(df)
        if count > threshold:
            break
        logger.warning(
            f"get_open_price 返回 {count}/{N} 行 (attempt {attempt}/30)，不足 95%，重试..."
        )
        time.sleep(2)

    count = len(df)
    if count == N:
        logger.info(f"get_open_price 完美匹配: {count}/{N} 只标的")
    elif count > threshold:
        logger.warning(
            f"get_open_price 部分匹配: {count}/{N} 只标的 (>95%), 缺失将在启动流程中补全"
        )
    else:
        logger.error(
            f"get_open_price 严重不足: {count}/{N} 只标的 (<=95%), 仍返回现有数据"
        )

    df = df.with_columns([
        pl.col("time").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y-%m-%d")
            .dt.offset_by("9h30m").alias("time"),
        pl.col("code").cast(pl.Utf8).str.slice(0, 6).alias("symbol"),
    ]).select([
        "time", "symbol", "open", "close", "high", "low", "volume", "money"
    ])

    df = _filter_symbol_prefix(df)
    logger.info(f"当天集合竞价数据加载完成: {len(df)} 行, {df['symbol'].n_unique()} 只标的")
    return df


def load_auction_data_prev(symbol_arr: list[str], trade_date: str) -> pl.DataFrame:
    """通过 get_call_auction 读取历史交易日的集合竞价数据。"""
    pdf = get_call_auction(start_date=trade_date, end_date=trade_date, symbol=symbol_arr)
    df = pl.from_pandas(_safe_pandas_convert(pdf))

    df = df.with_columns(
        pl.col("symbol").cast(pl.Utf8).str.slice(0, 6).alias("symbol")
    )
    df = _filter_symbol_prefix(df)

    df = df.with_columns([
        pl.lit(datetime.strptime(f"{trade_date} 09:30:00", "%Y-%m-%d %H:%M:%S")).alias("time"),
        pl.col("current").cast(pl.Float64).alias("open"),
        pl.col("current").cast(pl.Float64).alias("close"),
        pl.col("current").cast(pl.Float64).alias("high"),
        pl.col("current").cast(pl.Float64).alias("low"),
        pl.col("volume").cast(pl.Float64),
        pl.col("money").cast(pl.Float64),
    ]).select([
        "time", "symbol", "open", "close", "high", "low", "volume", "money"
    ])

    logger.info(
        f"历史集合竞价数据加载完成 ({trade_date}): {len(df)} 行, {df['symbol'].n_unique()} 只标的"
    )
    return df


# ============================================================
# 2. 主通道：CSV 文件读取
# ============================================================
def _kbar_csv_path(bar_time: datetime) -> str:
    hhmm = bar_time.strftime("%H%M")
    return os.path.join(Config.KBAR_CSV_DIR, f"kbar_{hhmm}.csv")


def _read_kbar_csv(csv_path: str) -> pl.DataFrame:
    """读取单个 kbar CSV，标准化为统一 schema。"""
    df = pl.read_csv(csv_path)
    df = df.with_columns([
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("code").str.slice(0, 6).alias("symbol"),
        pl.col("open").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("money").cast(pl.Float64),
    ]).select([
        "time", "symbol", "open", "close", "high", "low", "volume", "money"
    ])
    df = _filter_symbol_prefix(df)
    return df


def fetch_bar_from_csv(
    bar_time: datetime,
    expected_min_symbols: int = 100,
    tolerance_sec: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """
    从 CSV 主通道读取单个 bar，按墙钟 tolerance_sec 判断超时。

    Returns
    -------
    pl.DataFrame 或 None
        None 表示 tolerance 内未成功（文件不存在 / 截面不完整 / 读取异常）
    """
    if tolerance_sec is None:
        tolerance_sec = Config.CSV_TOLERANCE_SEC

    csv_path = _kbar_csv_path(bar_time)
    deadline = time.time() + tolerance_sec
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        if not os.path.exists(csv_path):
            logger.debug(f"[CSV attempt {attempt}] 文件不存在: {csv_path}")
        else:
            try:
                df = _read_kbar_csv(csv_path)
                n_symbols = df["symbol"].n_unique()
                if n_symbols >= expected_min_symbols:
                    logger.info(
                        f"[CSV] 读取成功: {os.path.basename(csv_path)}, "
                        f"{len(df)} 行, {n_symbols} 只标的 (attempt {attempt})"
                    )
                    return df
                else:
                    logger.debug(
                        f"[CSV attempt {attempt}] 截面 {n_symbols} < {expected_min_symbols}"
                    )
            except Exception as e:
                logger.warning(f"[CSV attempt {attempt}] 读取异常: {csv_path}, {e}")

        # 若剩余时间不够下一次 sleep，直接退出
        if time.time() + Config.CSV_POLL_INTERVAL >= deadline:
            break
        time.sleep(Config.CSV_POLL_INTERVAL)

    logger.warning(
        f"[CSV] tolerance {tolerance_sec}s 内未就绪: bar={bar_time.strftime('%H:%M')}, "
        f"共尝试 {attempt} 次"
    )
    return None


def scan_existing_kbar_csvs(end_time: Optional[datetime] = None) -> pl.DataFrame:
    """扫描 KBAR_CSV_DIR 下所有 kbar_*.csv，合并返回。"""
    if not os.path.exists(Config.KBAR_CSV_DIR):
        logger.warning(f"CSV 目录不存在: {Config.KBAR_CSV_DIR}")
        return pl.DataFrame()

    csv_files = sorted(glob.glob(os.path.join(Config.KBAR_CSV_DIR, "kbar_*.csv")))
    if not csv_files:
        logger.info(f"{Config.KBAR_CSV_DIR} 下暂无 kbar CSV")
        return pl.DataFrame()

    dfs = []
    for f in csv_files:
        basename = os.path.basename(f)
        try:
            hhmm = basename.replace("kbar_", "").replace(".csv", "")
            file_time = datetime.strptime(
                f"{Config.TRADE_DATE} {hhmm[:2]}:{hhmm[2:4]}:00",
                "%Y-%m-%d %H:%M:%S"
            )
            if end_time is not None and file_time > end_time:
                continue
        except Exception:
            logger.warning(f"无法从文件名解析时间，跳过: {basename}")
            continue

        try:
            dfs.append(_read_kbar_csv(f))
        except Exception as e:
            logger.error(f"读取失败，跳过 {f}: {e}")

    if not dfs:
        return pl.DataFrame()

    df_all = pl.concat(dfs)
    df_all = df_all.unique(subset=["symbol", "time"], keep="last")
    df_all = df_all.sort(["symbol", "time"])
    logger.info(
        f"扫描 CSV 完成: {len(csv_files)} 个文件, "
        f"合计 {len(df_all)} 行, 时间范围 {df_all['time'].min()} ~ {df_all['time'].max()}"
    )
    return df_all


# ============================================================
# 3. 备通道：聚宽 API 读取
# ============================================================
def _safe_pandas_convert(pdf):
    """将 pandas DataFrame 中的 nullable 类型转为普通 numpy 类型。"""
    import pandas as pd
    for col in pdf.columns:
        dtype = pdf[col].dtype
        if isinstance(dtype, (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
                              pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
                              pd.Float32Dtype, pd.Float64Dtype)):
            pdf[col] = pdf[col].astype('float64')
        elif isinstance(dtype, pd.StringDtype):
            pdf[col] = pdf[col].astype('object')
        elif isinstance(dtype, pd.BooleanDtype):
            pdf[col] = pdf[col].astype('object')
    return pdf


def _pandas_to_polars(df) -> pl.DataFrame:
    """聚宽 get_price 返回的 pandas DataFrame → polars。"""
    df = _safe_pandas_convert(df)
    df_pl = pl.from_pandas(df)

    if not df_pl["time"].dtype.is_temporal():
        df_pl = df_pl.with_columns(
            pl.col("time").cast(pl.Utf8).str.to_datetime("%Y-%m-%d %H:%M:%S")
        )

    df_pl = df_pl.with_columns([
        pl.col("symbol").cast(pl.Utf8).str.slice(0, 6),
        pl.col("open").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("money").cast(pl.Float64),
    ])
    df_pl = _filter_symbol_prefix(df_pl)
    return df_pl


def fetch_bar_from_jqdata(
    symbols_arr: list,
    bar_time: datetime,
    expected_min_symbols: int = 100,
) -> Optional[pl.DataFrame]:
    """
    从聚宽 API 读取单个 bar (备通道)。

    策略：
      - 拉取窗口 [bar_time - JQ_FETCH_WINDOW_MINUTES, now]
      - 返回值 filter 到 time == bar_time 这一根
      - 校验截面标的数 >= expected_min_symbols
      - 最多 JQ_MAX_RETRY 次重试，间隔 JQ_RETRY_INTERVAL
    """
    start_date = bar_time - timedelta(minutes=Config.JQ_FETCH_WINDOW_MINUTES)

    for attempt in range(1, Config.JQ_MAX_RETRY + 1):
        try:
            logger.info(
                f"[JQ fallback attempt {attempt}/{Config.JQ_MAX_RETRY}] "
                f"拉取 bar {bar_time.strftime('%H:%M')}"
            )
            pdf = get_price(
                symbols_arr,
                start_date=start_date,
                end_date=datetime.now(),
                frequency='1m',
                count=2000000,
            )
            if not isinstance(pdf, pl.DataFrame):
                df = _pandas_to_polars(pdf)
            else:
                df = pdf

            # filter 到目标 bar
            df_bar = df.filter(pl.col("time") == bar_time)
            if len(df_bar) == 0:
                logger.warning(
                    f"[JQ attempt {attempt}] bar {bar_time} 不在返回结果中, "
                    f"返回最新时间 = {df['time'].max() if len(df) else 'empty'}"
                )
            else:
                n_symbols = df_bar["symbol"].n_unique()
                if n_symbols >= expected_min_symbols:
                    logger.info(
                        f"[JQ] 兜底成功: bar={bar_time.strftime('%H:%M')}, "
                        f"{len(df_bar)} 行, {n_symbols} 只标的"
                    )
                    return df_bar
                else:
                    logger.warning(
                        f"[JQ attempt {attempt}] 截面 {n_symbols} < {expected_min_symbols}"
                    )
        except Exception as e:
            logger.error(f"[JQ attempt {attempt}] 异常: {e}")

        if attempt < Config.JQ_MAX_RETRY:
            time.sleep(Config.JQ_RETRY_INTERVAL)

    logger.error(f"[JQ] 兜底仍失败: bar={bar_time.strftime('%H:%M')}")
    return None


# ============================================================
# 4. 统一入口：CSV 优先, 超时切聚宽
# ============================================================
def fetch_bar_with_fallback(
    bar_time: datetime,
    symbols_arr: list,
    expected_min_symbols: int = 100,
    csv_tolerance_sec: Optional[float] = None,
) -> tuple[Optional[pl.DataFrame], str]:
    """
    按 CSV → 聚宽 API 的顺序读取单个 bar。

    Returns
    -------
    (df, source)
        df: pl.DataFrame 或 None (两级都失败)
        source: "csv" | "jqdata" | "none"
    """
    # --- 一级: CSV 主通道 ---
    df = fetch_bar_from_csv(
        bar_time,
        expected_min_symbols=expected_min_symbols,
        tolerance_sec=csv_tolerance_sec,
    )
    if df is not None:
        return df, "csv"

    # --- 二级: 聚宽 API 兜底 ---
    logger.warning(
        f"bar {bar_time.strftime('%H:%M')} CSV 超时, 切聚宽 API 兜底"
    )
    df = fetch_bar_from_jqdata(
        symbols_arr=symbols_arr,
        bar_time=bar_time,
        expected_min_symbols=expected_min_symbols,
    )
    if df is not None:
        return df, "jqdata"

    return None, "none"


# ============================================================
# 5. 特征计算 + 输出
# ============================================================
def compute_features(pipeline: FeaturePipeline, df_all: pl.DataFrame) -> pl.DataFrame:
    df_input = df_all.rename({"symbol": "code", "time": "datetime"})
    df_feat = pipeline.run(df_input)
    df_feat = df_feat.with_columns(
        pl.col("datetime").dt.date().cast(pl.Utf8).alias("date")
    )
    logger.info(f"特征计算完成: {len(df_feat)} 行, {len(df_feat.columns)} 列")
    return df_feat


def append_to_output(df_features: pl.DataFrame, output_dir: str, trade_date: str):
    """{output_dir}/{trade_date}/{HHMMSS}.parquet"""
    date_dir = os.path.join(output_dir, trade_date)
    os.makedirs(date_dir, exist_ok=True)

    time_val = df_features["datetime"][0]
    if isinstance(time_val, str):
        time_str = time_val.replace(":", "").replace("-", "").split()[1] if " " in time_val else time_val.replace(":", "")
    else:
        time_str = time_val.strftime("%H%M%S")

    file_path = os.path.join(date_dir, f"{time_str}.parquet")
    df_features.write_parquet(file_path)
    logger.info(f"写入分片: {file_path}, 共 {len(df_features)} 行")


def append_timing_record(timing_path: str, bar_time: datetime, cost_seconds: float, source: str = ""):
    """流式写入单条耗时记录到 CSV，并记录数据源。"""
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)
    file_exists = os.path.exists(timing_path) and os.path.getsize(timing_path) > 0
    time_str = bar_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(bar_time, datetime) else str(bar_time)

    with open(timing_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("datetime,cost_seconds,source\n")
        f.write(f"{time_str},{cost_seconds:.6f},{source}\n")


# ============================================================
# 6. 交易时间判断
# ============================================================
def is_trading_time(now: datetime) -> bool:
    t = now.strftime("%H:%M")
    morning = Config.MORNING_START <= t <= Config.MORNING_END
    afternoon = Config.AFTERNOON_START <= t <= Config.AFTERNOON_END
    return morning or afternoon


_trade_days_cache = None

def get_prev_trade_date(date_str: str) -> str:
    global _trade_days_cache
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    if _trade_days_cache is None:
        trade_days = get_trade_days(
            start_date=(d - timedelta(days=60)).strftime("%Y-%m-%d"),
            end_date=d.strftime("%Y-%m-%d"),
        )
        _trade_days_cache = sorted(trade_days)
        logger.info(
            f"交易日历已缓存: {_trade_days_cache[0]} ~ {_trade_days_cache[-1]}, "
            f"共 {len(_trade_days_cache)} 天"
        )
    prev_days = [td for td in _trade_days_cache if td < d]
    if not prev_days:
        raise ValueError(f"在缓存范围内找不到 {date_str} 之前的交易日")
    return prev_days[-1].strftime("%Y-%m-%d")


# ============================================================
# 7. 历史数据加载
# ============================================================
def load_historical_data(symbols_arr: list[str], trade_date: str) -> tuple[pl.DataFrame, str]:
    prev_dates = []
    d = trade_date
    for i in range(5):
        d = get_prev_trade_date(d)
        prev_dates.append(d)
    prev_dates.reverse()
    prev_trade_date_1 = prev_dates[-1]
    logger.info(f"前五个交易日: {prev_dates}")

    dfs = []
    for prev_trade_date in prev_dates:
        df_prev_auction = load_auction_data_prev(symbols_arr, prev_trade_date)
        prev_start = datetime.strptime(f"{prev_trade_date} 09:00:00", "%Y-%m-%d %H:%M:%S")
        prev_end = datetime.strptime(f"{prev_trade_date} 15:05:00", "%Y-%m-%d %H:%M:%S")
        logger.info(f"拉取前交易日 1min 数据: {prev_start} ~ {prev_end}")

        df_prev_1m = get_price(
            symbols_arr,
            start_date=prev_start,
            end_date=prev_end,
            frequency='1m',
            count=2000000,
        )
        if not isinstance(df_prev_1m, pl.DataFrame):
            df_prev_1m = _pandas_to_polars(df_prev_1m)

        dfs.extend([df_prev_auction, df_prev_1m])

    df_history = pl.concat(dfs)
    df_history = df_history.unique(subset=["symbol", "time"], keep="last")
    df_history = df_history.sort(["symbol", "time"])
    logger.info(
        f"历史数据加载完成: {len(df_history)} 行, "
        f"时间范围 {df_history['time'].min()} ~ {df_history['time'].max()}"
    )
    return df_history, prev_trade_date_1


# ============================================================
# 8. Bar 时间序列工具
# ============================================================
def get_all_expected_bar_times(trade_date: str, end_time: Optional[datetime] = None) -> list[datetime]:
    """09:31 ~ 11:30 + 13:01 ~ 15:00 (09:30 由集合竞价提供)"""
    morning_start = datetime.strptime(f"{trade_date} 09:31:00", "%Y-%m-%d %H:%M:%S")
    morning_end = datetime.strptime(f"{trade_date} 11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start = datetime.strptime(f"{trade_date} 13:01:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end = datetime.strptime(f"{trade_date} 15:00:00", "%Y-%m-%d %H:%M:%S")

    if end_time is None:
        end_time = afternoon_end

    bars = []
    t = morning_start
    while t <= morning_end and t <= end_time:
        bars.append(t)
        t += timedelta(minutes=1)
    t = afternoon_start
    while t <= afternoon_end and t <= end_time:
        bars.append(t)
        t += timedelta(minutes=1)
    return bars


def get_existing_bar_times(output_dir: str, trade_date: str) -> set[datetime]:
    date_dir = os.path.join(output_dir, trade_date)
    if not os.path.exists(date_dir):
        return set()

    result = set()
    for f in glob.glob(os.path.join(date_dir, "*.parquet")):
        filename = os.path.basename(f).replace(".parquet", "")
        if len(filename) == 6 and filename.isdigit():
            t = datetime.strptime(
                f"{trade_date} {filename[:2]}:{filename[2:4]}:{filename[4:6]}",
                "%Y-%m-%d %H:%M:%S"
            )
            result.add(t)
    return result


# ============================================================
# 9. 集合竞价缺失填充 + Forward fill
# ============================================================
def _fill_missing_auction_symbols(
    df_auction: pl.DataFrame,
    df_history: pl.DataFrame,
    symbols_arr: list,
    prev_trade_date_1: str,
) -> pl.DataFrame:
    auction_symbols = set(df_auction["symbol"].to_list())
    expected_symbols = set(s[:6] for s in symbols_arr)
    expected_symbols = {s for s in expected_symbols if not (s.startswith("2") or s.startswith("9"))}
    missing = expected_symbols - auction_symbols

    if not missing:
        logger.info("当天集合竞价数据完整，无需填充")
        return df_auction

    logger.warning(f"当天集合竞价缺失 {len(missing)} 只标的，尝试用前一交易日收盘价填充")
    prev_close_time = datetime.strptime(f"{prev_trade_date_1} 15:00:00", "%Y-%m-%d %H:%M:%S")
    df_prev_close = df_history.filter(
        (pl.col("time") == prev_close_time) & pl.col("symbol").is_in(list(missing))
    )

    if len(df_prev_close) > 0:
        fill_time = datetime.strptime(f"{Config.TRADE_DATE} 09:30:00", "%Y-%m-%d %H:%M:%S")
        df_fill = df_prev_close.select([
            pl.lit(fill_time).alias("time"),
            pl.col("symbol"),
            pl.col("close").alias("open"),
            pl.col("close"),
            pl.col("close").alias("high"),
            pl.col("close").alias("low"),
            pl.lit(0.0).alias("volume"),
            pl.lit(0.0).alias("money"),
        ])
        df_auction = pl.concat([df_auction, df_fill])
        still_missing = missing - set(df_prev_close["symbol"].to_list())
        logger.info(f"填充完成: 补全 {len(df_prev_close)} 只, 仍缺失 {len(still_missing)} 只")
    else:
        logger.warning("前一交易日无收盘数据可用于填充")

    return df_auction


def _forward_fill_missing_symbols(df_all: pl.DataFrame, symbols_arr: list) -> pl.DataFrame:
    """缺失标的 forward fill (仅当缺失比例 <= 1% 时)"""
    latest_time = df_all["time"].max()
    all_syms = set(s[:6] for s in symbols_arr)
    all_syms = {s for s in all_syms if not (s.startswith("2") or s.startswith("9"))}
    current_syms = set(df_all.filter(pl.col("time") == latest_time)["symbol"].to_list())
    missing = all_syms - current_syms

    if 0 < len(missing) <= int(len(symbols_arr) * 0.01):
        prev_time = df_all.filter(pl.col("time") < latest_time)["time"].max()
        if prev_time is not None:
            df_fill = (
                df_all.filter(
                    (pl.col("time") == prev_time) & pl.col("symbol").is_in(list(missing))
                ).with_columns([
                    pl.lit(latest_time).alias("time"),
                    pl.lit(0.0).alias("volume"),
                    pl.lit(0.0).alias("money"),
                ])
            )
            if len(df_fill) > 0:
                df_all = pl.concat([df_all, df_fill])
                df_all = df_all.unique(subset=["symbol", "time"], keep="last")
                df_all = df_all.sort(["symbol", "time"])
                logger.info(
                    f"Forward fill: 填充 {len(df_fill)} 只标的 "
                    f"(价格继承上一 bar, volume/money 置 0, "
                    f"缺失 {len(missing)}/{len(symbols_arr)})"
                )
    return df_all


# ============================================================
# 10. Instant Mode
# ============================================================
def run_instant(symbols_arr: list, expected_min_symbols: int = 100):
    """Instant Mode: 跳到最新时刻立即跟踪，中间缺失 bar 不阻塞。"""
    logger.info("=" * 60)
    logger.info("Instant Mode 启动 (CSV 主通道 + 聚宽 API 兜底)")
    logger.info("=" * 60)

    timing_path = os.path.join(
        Config.FEATURE_HOME,
        "realtime_features",
        f"{Config.TRADE_DATE}_feature_timing.csv",
    )
    pipeline = FeaturePipeline()
    skipped_bars = []

    # ---- Step 1: 加载历史数据 ----
    df_history, prev_trade_date_1 = load_historical_data(symbols_arr, Config.TRADE_DATE)

    # ---- Step 2: 等到 09:26 再拉集合竞价 ----
    auction_start = datetime.strptime(f"{Config.TRADE_DATE} 09:26:00", "%Y-%m-%d %H:%M:%S")
    wait_sec = (auction_start - datetime.now()).total_seconds()
    if wait_sec > 0:
        logger.info(f"等待 {wait_sec:.0f}s 到 09:26...")
        time.sleep(wait_sec)

    # ---- Step 3: 集合竞价 + 缺失填充 ----
    df_auction = load_auction_data_today(symbols_arr)
    df_auction = _fill_missing_auction_symbols(
        df_auction, df_history, symbols_arr, prev_trade_date_1
    )

    # ---- Step 4: 扫描当天已有的 CSV bar ----
    market_open = datetime.strptime(f"{Config.TRADE_DATE} {Config.MORNING_START}:00", "%Y-%m-%d %H:%M:%S")
    if datetime.now() > market_open:
        df_today_bars = scan_existing_kbar_csvs()
    else:
        df_today_bars = pl.DataFrame()
        logger.info("尚未开盘，跳过扫描 CSV")

    # ---- Step 5: 构建 df_all ----
    parts = [df_history, df_auction]
    if len(df_today_bars) > 0:
        parts.append(df_today_bars)
    df_all = pl.concat(parts)
    df_all = df_all.unique(subset=["symbol", "time"], keep="last")
    df_all = df_all.sort(["symbol", "time"])

    # ---- Step 6: 最新时刻计算特征 ----
    t_start = time.time()
    df_feat = compute_features(pipeline, df_all)
    t_delta = time.time() - t_start

    latest_time = df_all["time"].max()
    append_to_output(df_feat, Config.OUTPUT_DIR, Config.TRADE_DATE)
    append_timing_record(timing_path, latest_time, t_delta, source="init")
    logger.info(f"初始特征计算完成 (最新 bar: {latest_time}), 耗时 {t_delta:.3f}s")

    last_processed_time = latest_time

    # ---- Step 7: 主循环 ----
    while True:
        now = datetime.now()

        if now.strftime("%H:%M") > Config.AFTERNOON_END:
            logger.info("已过收盘时间，脚本退出")
            break

        if not is_trading_time(now):
            logger.info(f"当前非交易时段 ({now.strftime('%H:%M')}), 等待中...")
            time.sleep(10)
            continue

        expected_time = now.replace(second=0, microsecond=0)

        # 09:30 由集合竞价提供
        if expected_time <= datetime.strptime(f"{Config.TRADE_DATE} 09:30:00", "%Y-%m-%d %H:%M:%S"):
            time.sleep(5)
            continue

        # 午休跨越
        hm = expected_time.strftime("%H:%M")
        if Config.MORNING_END < hm < Config.AFTERNOON_START:
            time.sleep(10)
            continue

        if expected_time <= last_processed_time:
            sleep_sec = 60 - now.second + Config.POLL_OFFSET_SEC
            time.sleep(max(sleep_sec, 1))
            continue

        # 等到 expected_time + POLL_OFFSET_SEC 再开始读
        target_wall = expected_time + timedelta(seconds=Config.POLL_OFFSET_SEC)
        wait_sec = (target_wall - datetime.now()).total_seconds()
        if wait_sec > 0:
            logger.info(
                f"等待 {wait_sec:.1f}s 到 {target_wall.strftime('%H:%M:%S')} "
                f"再读 bar {expected_time.strftime('%H:%M')}"
            )
            time.sleep(max(wait_sec, 0))

        # ---- 先恢复之前 skipped 的 bars (两级 fallback) ----
        recovered_dfs = []
        recovered_times = []
        still_skipped = []
        for sb in skipped_bars:
            df_sb, src_sb = fetch_bar_with_fallback(
                bar_time=sb,
                symbols_arr=symbols_arr,
                expected_min_symbols=1,  # 恢复时放宽校验
                csv_tolerance_sec=1.0,   # 恢复路径不应阻塞太久
            )
            if df_sb is not None and len(df_sb) > 0:
                recovered_dfs.append(df_sb)
                recovered_times.append((sb, src_sb))
            else:
                still_skipped.append(sb)
        if recovered_times:
            logger.info(
                f"恢复了 {len(recovered_times)} 个之前失败的 bar: "
                f"{[(t.strftime('%H:%M'), s) for t, s in recovered_times]}"
            )
        skipped_bars = still_skipped

        # ---- 读取本轮的 bar (CSV → JQ fallback) ----
        df_new, source = fetch_bar_with_fallback(
            bar_time=expected_time,
            symbols_arr=symbols_arr,
            expected_min_symbols=expected_min_symbols,
            csv_tolerance_sec=Config.CSV_TOLERANCE_SEC,
        )

        if df_new is None and not recovered_dfs:
            skipped_bars.append(expected_time)
            logger.warning(
                f"Bar {expected_time} 两级 fallback 均失败，记入待补列表 "
                f"(累计 {len(skipped_bars)} 个待补 bar)"
            )
            continue

        # ---- 合并新数据 ----
        new_dfs = recovered_dfs[:]
        if df_new is not None:
            new_dfs.append(df_new)
        df_incr_all = pl.concat(new_dfs) if new_dfs else pl.DataFrame()

        if len(df_incr_all) > 0:
            df_incr_all = df_incr_all.select(df_all.columns)
            df_all = pl.concat([df_all, df_incr_all])
            df_all = df_all.unique(subset=["symbol", "time"], keep="last")
            df_all = df_all.sort(["symbol", "time"])

        # ---- 缺失标的 forward fill ----
        df_all = _forward_fill_missing_symbols(df_all, symbols_arr)

        logger.info(
            f"内存数据更新: 总计 {len(df_all)} 行, "
            f"{df_all['symbol'].n_unique()} 只标的, "
            f"时间范围 {df_all['time'].min()} ~ {df_all['time'].max()}"
        )

        # ---- 计算特征并写入 ----
        existing_bars = get_existing_bar_times(Config.OUTPUT_DIR, Config.TRADE_DATE)
        new_times = sorted(set(df_incr_all["time"].unique().to_list())) if len(df_incr_all) > 0 else []
        bars_to_compute = [t for t in new_times if t not in existing_bars]

        if not bars_to_compute:
            logger.info("所有 bar 已有输出，跳过计算")
            last_processed_time = df_all["time"].max()
            continue

        # 标注每个 bar 的数据源 (主 bar 用 source，恢复 bar 查 recovered_times)
        src_map = {t: s for t, s in recovered_times}
        if df_new is not None:
            src_map[expected_time] = source

        for bar_time in sorted(bars_to_compute):
            t_start = time.time()
            df_for_feat = df_all.filter(pl.col("time") <= bar_time)
            df_feat = compute_features(pipeline, df_for_feat)
            append_to_output(df_feat, Config.OUTPUT_DIR, Config.TRADE_DATE)
            t_delta = time.time() - t_start
            bar_src = src_map.get(bar_time, "unknown")
            append_timing_record(timing_path, bar_time, t_delta, source=bar_src)
            logger.info(
                f"Bar {bar_time} 特征计算完成, 耗时 {t_delta:.3f}s, source={bar_src}"
            )

        last_processed_time = df_all["time"].max()
        logger.info(f"本轮处理完成, last_processed_time = {last_processed_time}")
        if skipped_bars:
            logger.info(f"仍有 {len(skipped_bars)} 个待补 bar: {skipped_bars}")
        logger.info("-" * 40)

    # ---- 结束 ----
    if skipped_bars:
        logger.warning(f"收盘时仍有 {len(skipped_bars)} 个未补全 bar: {skipped_bars}")
        logger.warning("请运行 --mode backfill 补全")
    logger.info("=" * 60)
    logger.info("Instant Mode 正常退出")
    logger.info("=" * 60)


# ============================================================
# 11. Backfill Mode
# ============================================================
def run_backfill(symbols_arr: list, expected_min_symbols: int = 100):
    """Backfill Mode: 补全当天所有缺失 bar 后退出。"""
    logger.info("=" * 60)
    logger.info("Backfill Mode 启动 (CSV 主通道 + 聚宽 API 兜底)")
    logger.info("=" * 60)

    timing_path = os.path.join(
        Config.FEATURE_HOME,
        "realtime_features",
        f"{Config.TRADE_DATE}_feature_timing.csv",
    )
    pipeline = FeaturePipeline()

    # ---- Step 1: 计算缺失 bar 列表 ----
    now = datetime.now()
    end_time = min(
        now,
        datetime.strptime(f"{Config.TRADE_DATE} 15:00:00", "%Y-%m-%d %H:%M:%S")
    )
    all_expected = get_all_expected_bar_times(Config.TRADE_DATE, end_time)
    auction_bar = datetime.strptime(f"{Config.TRADE_DATE} 09:30:00", "%Y-%m-%d %H:%M:%S")
    if end_time >= auction_bar:
        all_expected = [auction_bar] + all_expected

    existing = get_existing_bar_times(Config.OUTPUT_DIR, Config.TRADE_DATE)
    missing_bars = sorted([t for t in all_expected if t not in existing])

    if not missing_bars:
        logger.info("无缺失 bar，Backfill 无需执行，退出")
        return

    logger.info(
        f"缺失 {len(missing_bars)} 个 bar: "
        f"{missing_bars[0].strftime('%H:%M')} ~ {missing_bars[-1].strftime('%H:%M')}"
    )

    # ---- Step 2: 加载历史 + 集合竞价 + 当天 CSV ----
    df_history, prev_trade_date_1 = load_historical_data(symbols_arr, Config.TRADE_DATE)
    df_auction = load_auction_data_today(symbols_arr)
    df_auction = _fill_missing_auction_symbols(
        df_auction, df_history, symbols_arr, prev_trade_date_1
    )
    df_today_bars = scan_existing_kbar_csvs(end_time=end_time)

    parts = [df_history, df_auction]
    if len(df_today_bars) > 0:
        parts.append(df_today_bars)
    df_all = pl.concat(parts)
    df_all = df_all.unique(subset=["symbol", "time"], keep="last")
    df_all = df_all.sort(["symbol", "time"])

    logger.info(
        f"全量数据 (含 CSV): {len(df_all)} 行, "
        f"时间范围 {df_all['time'].min()} ~ {df_all['time'].max()}"
    )

    # ---- Step 3: 逐 bar 补全 (如果 df_all 里没有就用 fallback 抓) ----
    available_times = set(df_all["time"].unique().to_list())
    skipped = []

    for i, bar_time in enumerate(missing_bars):
        bar_src = "existing"

        if bar_time not in available_times:
            # CSV 没有 → 尝试从聚宽抓
            logger.info(
                f"[{i+1}/{len(missing_bars)}] Bar {bar_time.strftime('%H:%M:%S')} "
                f"不在 CSV 里, 尝试 fallback"
            )
            df_bar, bar_src = fetch_bar_with_fallback(
                bar_time=bar_time,
                symbols_arr=symbols_arr,
                expected_min_symbols=1,  # backfill 时放宽
                csv_tolerance_sec=2.0,   # CSV 已扫过一遍，再给 2s 宽限
            )
            if df_bar is None:
                logger.warning(
                    f"Bar {bar_time} 两级 fallback 均失败 (CSV 未落盘 + JQ 无数据)，跳过"
                )
                skipped.append(bar_time)
                continue

            # 合进 df_all
            df_bar = df_bar.select(df_all.columns)
            df_all = pl.concat([df_all, df_bar])
            df_all = df_all.unique(subset=["symbol", "time"], keep="last")
            df_all = df_all.sort(["symbol", "time"])
            available_times.add(bar_time)

        t_start = time.time()
        df_for_feat = df_all.filter(pl.col("time") <= bar_time)
        df_feat = compute_features(pipeline, df_for_feat)
        append_to_output(df_feat, Config.OUTPUT_DIR, Config.TRADE_DATE)
        t_delta = time.time() - t_start
        append_timing_record(timing_path, bar_time, t_delta, source=bar_src)
        logger.info(
            f"[{i+1}/{len(missing_bars)}] Bar {bar_time.strftime('%H:%M:%S')} "
            f"补全完成, 耗时 {t_delta:.3f}s, source={bar_src}"
        )

    # ---- 结束 ----
    if skipped:
        logger.warning(f"有 {len(skipped)} 个 bar 无法补全: {skipped}")
    logger.info("=" * 60)
    logger.info(
        f"Backfill Mode 完成, 共补全 {len(missing_bars) - len(skipped)}/{len(missing_bars)} 个 bar"
    )
    logger.info("=" * 60)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Realtime feature engine (CSV + JQ fallback)")
    parser.add_argument("--mode", choices=["instant", "backfill"], default="instant",
                        help="instant: 跳到最新bar实时跟踪; backfill: 补全缺失bar后退出")
    args = parser.parse_args()

    from dataloader.api_base.api_conf import auth
    auth(username='sihang', password='sihang123', env='ali')

    symbols = get_all_securities(types='stock', date=Config.TRADE_DATE)
    symbols_arr = list(symbols['symbol'].values)
    expected_min = int(len(symbols_arr) * 0.99)

    if args.mode == "instant":
        run_instant(symbols_arr=symbols_arr, expected_min_symbols=expected_min)
    else:
        run_backfill(symbols_arr=symbols_arr, expected_min_symbols=expected_min)
