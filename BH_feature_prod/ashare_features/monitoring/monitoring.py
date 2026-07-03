#!/usr/bin/env python3
"""
monitoring.py — 独立旁路观察者

通过轮询日志文件和 parquet 输出目录来采集实时特征计算的运行指标，
写入 SQLite 数据库供 Grafana 展示。不修改主流程任何代码。

运行：
    TRADE_DATE=2026-03-02 python -m monitoring.monitoring
    # 或通过 run_monitoring.sh
"""
import os
import sys
import re
import time
import glob as globmod
import logging
from datetime import datetime, timedelta
from typing import Optional

# 确保 import 能找到 monitoring 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring import config
from monitoring.collector import MetricsCollector

logger = logging.getLogger("monitoring.observer")


# ======================================================================
# LogParser — 增量日志解析
# ======================================================================
class LogParser:
    """增量读取日志文件，解析关键事件"""

    # 正则模式
    _RE_RETRY = re.compile(
        r"第 (\d+)/(\d+) 次拉取, 期望 bar: (\S+ \S+)"
    )
    _RE_COST = re.compile(
        r"耗时[:\s]*([\d.]+)s"
    )
    _RE_WRITE = re.compile(
        r".*?(\d{6})\.parquet.*?(\d+)"
    )
    _RE_ABANDON = re.compile(
        r"达到最大重试次数 (\d+)，放弃本轮拉取"
    )
    _RE_GET_PRICE_ERR = re.compile(
        r"get_price 异常 \(attempt (\d+)\): (.+)"
    )
    _RE_EXIT = re.compile(
        r"已过收盘时间，脚本退出|交易日结束，脚本正常退出"
    )
    _RE_BAR_VALIDATE = re.compile(
        r"Bar 校验通过: time=(\S+ \S+), symbols=(\d+)"
    )
    _RE_CSV_SUCCESS = re.compile(
        r"\[CSV\].*?kbar_(\d{4})\.csv.*?(\d+).*?(\d+).*?attempt\s+(\d+)"
    )
    _RE_JQ_FALLBACK_ATTEMPT = re.compile(
        r"\[JQ fallback attempt (\d+)/(\d+)\].*?(\d{2}:\d{2})"
    )
    _RE_JQ_SUCCESS = re.compile(
        r"\[JQ\].*?bar=(\d{2}:\d{2}).*?(\d+).*?(\d+)"
    )
    _RE_FALLBACK_FAIL = re.compile(
        r"Bar (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) 两级 fallback 均失败.*记入待补列表"
    )
    _RE_FALLBACK_SKIP = re.compile(
        r"Bar (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) 两级 fallback 均失败 .*跳过"
    )
    _RE_CLOSE_UNFILLED = re.compile(
        r"收盘时仍有 (\d+) 个未补全 bar: \[(.*)\]"
    )
    _RE_BAR_DONE = re.compile(
        r"(?:\[\d+/\d+\]\s*)?Bar\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
        r"([\d.]+)s,\s*source=([A-Za-z0-9_]+)"
    )
    _RE_INIT_DONE = re.compile(
        r".*?bar:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?([\d.]+)s"
    )

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._offset = 0
        # 跨轮询的状态
        self._current_bar_time: Optional[str] = None
        self._current_max_retry: int = 0
        self._last_cost: Optional[float] = None
        self._last_error_reason: str = ""
        self._last_n_symbols: int = 0
        self._pending_writes: dict[str, dict] = {}
        self.pipeline_exited = False

    def seek_to_end(self):
        """跳到文件末尾（用于只监控新增行）"""
        if os.path.exists(self.log_path):
            self._offset = os.path.getsize(self.log_path)

    def parse_new_lines(self) -> list:
        """
        读取日志文件新增的行，返回解析后的事件列表。

        事件格式: dict with 'type' key:
          - {'type': 'bar_complete', 'bar_time': str, 'cost_seconds': float,
             'retry_count': int, 'n_symbols': int}
          - {'type': 'abandon', 'bar_time': str, 'retry_count': int, 'reason': str}
          - {'type': 'exit'}
        """
        if not os.path.exists(self.log_path):
            return []

        events = []
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                new_content = f.read()
                self._offset = f.tell()
        except Exception as e:
            logger.warning(f"读取日志失败: {e}")
            return []

        if not new_content:
            return []

        for line in new_content.splitlines():
            m = self._RE_CSV_SUCCESS.search(line)
            if m:
                hhmm = m.group(1)
                self._current_bar_time = f"{hhmm[:2]}:{hhmm[2:4]}:00"
                self._last_n_symbols = int(m.group(3))
                # CSV polling attempts are local file attempts, not API retries.
                self._current_max_retry = 0
                continue

            m = self._RE_JQ_FALLBACK_ATTEMPT.search(line)
            if m:
                retry_n = int(m.group(1))
                hhmm = m.group(3)
                self._current_bar_time = f"{hhmm}:00"
                self._current_max_retry = max(self._current_max_retry, retry_n)
                continue

            m = self._RE_JQ_SUCCESS.search(line)
            if m:
                self._current_bar_time = f"{m.group(1)}:00"
                self._last_n_symbols = int(m.group(3))
                continue

            m = self._RE_BAR_DONE.search(line)
            if m:
                bar_time = self._datetime_to_time_str(m.group(1))
                cost_seconds = float(m.group(2))
                event = self._pending_writes.pop(bar_time, None) or {
                    "type": "bar_complete",
                    "bar_time": bar_time,
                    "n_symbols": self._last_n_symbols,
                }
                event["cost_seconds"] = cost_seconds
                event["retry_count"] = event.get("retry_count", self._current_max_retry)
                events.append(event)
                self._reset_current_bar_state()
                continue

            m = self._RE_INIT_DONE.search(line)
            if m:
                bar_time = self._datetime_to_time_str(m.group(1))
                event = self._pending_writes.pop(bar_time, None) or {
                    "type": "bar_complete",
                    "bar_time": bar_time,
                    "n_symbols": self._last_n_symbols,
                }
                event["cost_seconds"] = float(m.group(2))
                event["retry_count"] = 0
                events.append(event)
                self._reset_current_bar_state()
                continue

            m = self._RE_FALLBACK_SKIP.search(line)
            if m:
                bar_time = self._datetime_to_time_str(m.group(1))
                events.append({
                    "type": "abandon",
                    "bar_time": bar_time,
                    "retry_count": self._current_max_retry,
                    "reason": "csv and jqdata fallback failed",
                })
                self._reset_current_bar_state()
                continue

            m = self._RE_FALLBACK_FAIL.search(line)
            if m:
                self._current_bar_time = self._datetime_to_time_str(m.group(1))
                self._last_error_reason = "csv and jqdata fallback failed; pending backfill"
                continue

            m = self._RE_CLOSE_UNFILLED.search(line)
            if m:
                for bar_time in self._extract_datetimes_from_text(m.group(2)):
                    events.append({
                        "type": "abandon",
                        "bar_time": self._datetime_to_time_str(bar_time),
                        "retry_count": 0,
                        "reason": "market close with unfilled pending bar",
                    })
                self._reset_current_bar_state()
                continue

            # 重试
            m = self._RE_RETRY.search(line)
            if m:
                retry_n = int(m.group(1))
                bar_dt = m.group(3)
                # 从 datetime string 提取 HH:MM:SS
                self._current_bar_time = self._datetime_to_time_str(bar_dt)
                self._current_max_retry = max(self._current_max_retry, retry_n)
                continue

            # Bar 校验通过 (记录 bar_time 和 n_symbols)
            m = self._RE_BAR_VALIDATE.search(line)
            if m:
                bar_dt = m.group(1)
                self._current_bar_time = self._datetime_to_time_str(bar_dt)
                self._last_n_symbols = int(m.group(2))
                continue

            # 耗时
            m = self._RE_COST.search(line)
            if m:
                self._last_cost = float(m.group(1))
                continue

            # get_price 异常
            m = self._RE_GET_PRICE_ERR.search(line)
            if m:
                self._last_error_reason = m.group(2).strip()
                continue

            # 写入分片 — bar 完成的确定性事件
            m = self._RE_WRITE.search(line)
            if m:
                hhmmss = m.group(1)
                n_rows = int(m.group(2))
                bar_time = f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
                events.append({
                    "type": "bar_complete",
                    "bar_time": bar_time,
                    "cost_seconds": self._last_cost or 0.0,
                    "retry_count": self._current_max_retry,
                    "n_symbols": n_rows,
                })
                if self._last_cost is None:
                    self._pending_writes[bar_time] = events.pop()
                # 重置
                self._reset_current_bar_state(keep_pending=True)
                continue

            # 放弃
            m = self._RE_ABANDON.search(line)
            if m:
                retry_count = int(m.group(1))
                bar_time = self._current_bar_time or "unknown"
                events.append({
                    "type": "abandon",
                    "bar_time": bar_time,
                    "retry_count": retry_count,
                    "reason": self._last_error_reason or "max retry exceeded",
                })
                self._reset_current_bar_state()
                continue

            # 退出
            if self._RE_EXIT.search(line):
                self.pipeline_exited = True
                events.append({"type": "exit"})
                continue

        return events

    def _reset_current_bar_state(self, keep_pending: bool = False):
        self._current_bar_time = None
        self._current_max_retry = 0
        self._last_cost = None
        self._last_error_reason = ""
        self._last_n_symbols = 0
        if not keep_pending:
            self._pending_writes.clear()

    @staticmethod
    def _datetime_to_time_str(dt_str: str) -> str:
        """'2026-03-02 09:31:00' -> '09:31:00'"""
        parts = dt_str.strip().split()
        if len(parts) >= 2:
            return parts[1]
        return dt_str

    @staticmethod
    def _extract_datetimes_from_text(text: str) -> list[str]:
        return re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)


# ======================================================================
# ParquetWatcher — 新文件发现与读取
# ======================================================================
class ParquetWatcher:
    """监控 parquet 输出目录，发现并读取新文件"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._processed_files: set = set()

    def scan_existing(self) -> list:
        """返回目录中已有的 parquet 文件名（不标记为已处理）"""
        pattern = os.path.join(self.output_dir, "*.parquet")
        return sorted(globmod.glob(pattern))

    def mark_processed(self, filepath: str):
        self._processed_files.add(os.path.basename(filepath))

    def check_new_files(self) -> list:
        """返回新增的 parquet 文件路径（尚未处理过的）"""
        pattern = os.path.join(self.output_dir, "*.parquet")
        all_files = sorted(globmod.glob(pattern))
        new_files = [
            f for f in all_files
            if os.path.basename(f) not in self._processed_files
        ]
        return new_files

    @staticmethod
    def read_parquet(filepath: str):
        """读取单个 parquet 文件，返回 polars DataFrame"""
        import polars as pl
        return pl.read_parquet(filepath)

    def read_all_parquets(self):
        """读取目录下所有 parquet 并 concat，用于收盘后校验"""
        import polars as pl
        pattern = os.path.join(self.output_dir, "*.parquet")
        files = sorted(globmod.glob(pattern))
        if not files:
            return None
        dfs = [pl.read_parquet(f) for f in files]
        return pl.concat(dfs)

    @staticmethod
    def filename_to_bar_time(filename: str) -> str:
        """'093100.parquet' -> '09:31:00'"""
        base = os.path.basename(filename).replace(".parquet", "")
        if len(base) == 6 and base.isdigit():
            return f"{base[:2]}:{base[2:4]}:{base[4:6]}"
        return base


# ======================================================================
# Validator — 收盘后一致性校验
# ======================================================================

class Validator:
    """对比实时 parquet 与离线 parquet，写入 validation 表"""

    def __init__(self, collector: MetricsCollector, trade_date: str):
        self.collector = collector
        self.trade_date = trade_date
        self.offline_dir = config.OFFLINE_FEATURES_DIR

    def run(self, df_online):
        """执行校验。df_online: 实时全天 concat 后的 polars DataFrame"""
        import polars as pl

        offline_path = os.path.join(self.offline_dir, f"{self.trade_date}.parquet")
        if not os.path.exists(offline_path):
            logger.warning(f"离线文件不存在: {offline_path}，跳过校验")
            return False

        logger.info(f"开始一致性校验: {offline_path}")
        df_offline = pl.read_parquet(offline_path)

        # 统一列名: symbol->code, time->datetime (如有必要)
        if "symbol" in df_offline.columns and "code" not in df_offline.columns:
            df_offline = df_offline.rename({"symbol": "code"})
        if "time" in df_offline.columns and "datetime" not in df_offline.columns:
            df_offline = df_offline.rename({"time": "datetime"})
        if "symbol" in df_online.columns and "code" not in df_online.columns:
            df_online = df_online.rename({"symbol": "code"})
        if "time" in df_online.columns and "datetime" not in df_online.columns:
            df_online = df_online.rename({"time": "datetime"})

        key_cols = ["code", "datetime"]

        # 替换 inf/-inf 为 0, fill nan 为 0
        numeric_cols_off = [
            c for c in df_offline.columns
            if df_offline[c].dtype in [pl.Float32, pl.Float64]
        ]
        numeric_cols_on = [
            c for c in df_online.columns
            if df_online[c].dtype in [pl.Float32, pl.Float64]
        ]
        df_offline = df_offline.with_columns(
            pl.col(c).replace([float("inf"), float("-inf")], 0.0).fill_nan(0.0)
            for c in numeric_cols_off
        )
        df_online = df_online.with_columns(
            pl.col(c).replace([float("inf"), float("-inf")], 0.0).fill_nan(0.0)
            for c in numeric_cols_on
        )

        # 过滤掉 15:00 收盘时间点样本
        before_off, before_on = len(df_offline), len(df_online)
        close_mask = (pl.col("datetime").dt.hour() == 15) & (pl.col("datetime").dt.minute() == 0)
        df_offline = df_offline.filter(~close_mask)
        df_online = df_online.filter(~close_mask)
        logger.info(
            f"过滤 15:00 收盘样本: offline {before_off} -> {len(df_offline)}, "
            f"online {before_on} -> {len(df_online)}"
        )

        # Inner join
        df_off_j = df_offline.join(
            df_online.select(key_cols).unique(), on=key_cols, how="inner"
        )
        df_on_j = df_online.join(
            df_offline.select(key_cols).unique(), on=key_cols, how="inner"
        )

        # 排序对齐
        df_off_j = df_off_j.sort(key_cols)
        df_on_j = df_on_j.sort(key_cols)

        total_rows = len(df_off_j)
        common_cols = [
            c for c in df_off_j.columns
            if c in df_on_j.columns and c not in key_cols
        ]

        col_results = []
        for col in common_cols:
            s1 = df_off_j[col]
            s2 = df_on_j[col]

            if s1.dtype in [pl.Float32, pl.Float64]:
                abs_diff = (s1 - s2).abs()
                relative_error = abs_diff / (s1.abs() + 1e-10)
                atol = 1e4 if col == "money" else 1e-2
                equal_mask = (s1.is_null() & s2.is_null()) | (
                    (relative_error < 0.001) | (abs_diff < atol)  # 两个条件满足其一即可
                )
            else:
                equal_mask = (s1 == s2) | (s1.is_null() & s2.is_null())

            equal_ratio = float(equal_mask.sum()) / max(total_rows, 1)
            max_abs_diff = 0.0
            correlation = None

            if s1.dtype in [pl.Float32, pl.Float64]:
                diff = (s1.cast(pl.Float64) - s2.cast(pl.Float64)).abs()
                diff_clean = diff.drop_nulls()
                if len(diff_clean) > 0:
                    max_abs_diff = float(diff_clean.max())

                # pearson correlation
                try:
                    correlation = float(
                        s1.cast(pl.Float64).pearson_corr(s2.cast(pl.Float64))
                    )
                except Exception:
                    correlation = None

            is_known = col in config.KNOWN_DIVERGENT_FEATURES
            col_results.append((
                col, equal_ratio, max_abs_diff, correlation, int(is_known)
            ))

        # 汇总
        matched = sum(1 for _, er, _, _, _ in col_results if er >= 0.999)
        diverged = len(col_results) - matched
        mean_eq = (
            sum(er for _, er, _, _, _ in col_results) / max(len(col_results), 1)
        )

        self.collector.record_validation_summary(
            total_rows=total_rows,
            total_columns=len(common_cols),
            matched_columns=matched,
            diverged_columns=diverged,
            mean_equal_ratio=mean_eq,
        )
        self.collector.record_validation_columns(col_results)

        logger.info(
            f"校验完成: {total_rows} 行, {len(common_cols)} 列, "
            f"matched={matched}, diverged={diverged}, mean_eq={mean_eq:.6f}"
        )
        return True


# ======================================================================
# 主循环
# ======================================================================
def main():
    trade_date = config.TRADE_DATE

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(
                    config.LOG_DIR,
                    f"{trade_date.replace('-', '')}_monitoring.log",
                ),
                encoding="utf-8",
            ),
        ],
    )

    logger.info(f"监控启动: trade_date={trade_date}")

    # 初始化组件
    collector = MetricsCollector(trade_date=trade_date)

    output_dir = os.path.join(config.REALTIME_OUTPUT_DIR, trade_date.replace("-", ""))
    # 兼容带横线和不带横线的目录名
    if not os.path.isdir(output_dir):
        output_dir = os.path.join(config.REALTIME_OUTPUT_DIR, trade_date)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Parquet 输出目录: {output_dir}")

    watcher = ParquetWatcher(output_dir)

    log_path = os.environ.get("PIPELINE_LOG", "")
    if not log_path:
        log_filename = f"{trade_date.replace('-', '')}_realtime_feature.log"
        log_path = os.path.join(config.LOG_DIR, log_filename)
        # 兼容带横线的日志文件名
        if not os.path.exists(log_path):
            log_filename_alt = f"{trade_date}_realtime_feature.log"
            log_path_alt = os.path.join(config.LOG_DIR, log_filename_alt)
            if os.path.exists(log_path_alt):
                log_path = log_path_alt
    logger.info(f"日志文件: {log_path}")

    log_parser = LogParser(log_path)

    # ---- 启动模式判断 ----
    skip_main_loop = False
    recorded = collector.get_recorded_bar_times()
    existing_files = watcher.scan_existing()

    if recorded and existing_files:
        # 断点续传：跳过已处理的文件
        for fp in existing_files:
            bt = watcher.filename_to_bar_time(fp)
            if bt in recorded:
                watcher.mark_processed(fp)
        # 日志跳到末尾，只关注新行
        log_parser.seek_to_end()
        logger.info(
            f"断点续传: DB 已有 {len(recorded)} 条记录, "
            f"目录有 {len(existing_files)} 个文件, "
            f"跳过 {len(watcher._processed_files)} 个已处理文件"
        )
        # 所有文件都已处理完，标记为可直接跳到收盘后阶段
        if len(watcher._processed_files) == len(existing_files):
            skip_main_loop = True
            logger.info("所有 bar 已处理完毕，跳过主循环直接进入收盘后校验")
    elif not recorded and existing_files:
        # 回填模式：处理所有已有文件
        logger.info(
            f"回填模式: DB 无记录, 目录有 {len(existing_files)} 个文件, "
            "将处理所有已有文件"
        )
    else:
        logger.info("等待模式: 无已有数据，等待新文件")

    # ---- 主轮询循环 ----
    while not skip_main_loop:
        now = datetime.now()

        # A. 解析日志新行
        events = log_parser.parse_new_lines()
        for ev in events:
            if ev["type"] == "bar_complete":
                collector.upsert_bar_metrics(
                    bar_time=ev["bar_time"],
                    cost_seconds=ev["cost_seconds"],
                    retry_count=ev["retry_count"],
                    status="ok",
                )
                logger.info(
                    f"[LOG] bar {ev['bar_time']} 完成: "
                    f"cost={ev['cost_seconds']:.1f}s, "
                    f"retry={ev['retry_count']}, "
                    f"rows={ev['n_symbols']}"
                )
            elif ev["type"] == "abandon":
                collector.record_abandon(
                    bar_time=ev["bar_time"],
                    retry_count=ev["retry_count"],
                    reason=ev["reason"],
                )
                logger.warning(
                    f"[LOG] bar {ev['bar_time']} 放弃: "
                    f"retry={ev['retry_count']}, reason={ev['reason']}"
                )
            elif ev["type"] == "exit":
                logger.info("[LOG] 检测到主流程退出信号")

        # B. 扫描新 parquet 文件
        new_files = watcher.check_new_files()
        for fp in new_files:
            bar_time = watcher.filename_to_bar_time(fp)
            try:
                df = watcher.read_parquet(fp)
            except Exception as e:
                logger.warning(f"读取 {fp} 失败 (可能仍在写入): {e}")
                continue

            # 统计 null/inf
            n_null, n_inf = 0, 0
            n_symbols = len(df)
            try:
                n_null, n_inf = MetricsCollector._count_null_inf(df)
            except Exception as e:
                logger.warning(f"统计 null/inf 失败: {e}")

            collector.upsert_bar_metrics(
                bar_time=bar_time,
                n_symbols=n_symbols,
                n_null_cells=n_null,
                n_inf_cells=n_inf,
            )

            # 特征统计
            try:
                collector.record_feature_stats(bar_time, df)
            except Exception as e:
                logger.warning(f"记录 feature_stats 失败: {e}")

            watcher.mark_processed(fp)
            logger.info(
                f"[PQ] {os.path.basename(fp)}: "
                f"symbols={n_symbols}, null={n_null}, inf={n_inf}"
            )

        # C. 退出条件判断
        hour_min = now.hour * 100 + now.minute
        if log_parser.pipeline_exited and hour_min >= 1505:
            logger.info("主流程已退出且已过 15:05，进入收盘后处理")
            break
        if hour_min >= 1600:
            logger.info("已过 16:00，强制进入收盘后处理")
            break

        time.sleep(config.POLL_INTERVAL)

    # ---- 收盘后：最后一轮扫描 ----
    logger.info("执行收盘后最终扫描...")
    final_events = log_parser.parse_new_lines()
    for ev in final_events:
        if ev["type"] == "bar_complete":
            collector.upsert_bar_metrics(
                bar_time=ev["bar_time"],
                cost_seconds=ev["cost_seconds"],
                retry_count=ev["retry_count"],
                status="ok",
            )
        elif ev["type"] == "abandon":
            collector.record_abandon(
                bar_time=ev["bar_time"],
                retry_count=ev["retry_count"],
                reason=ev["reason"],
            )

    final_files = watcher.check_new_files()
    for fp in final_files:
        bar_time = watcher.filename_to_bar_time(fp)
        try:
            df = watcher.read_parquet(fp)
            n_null, n_inf = MetricsCollector._count_null_inf(df)
            collector.upsert_bar_metrics(
                bar_time=bar_time,
                n_symbols=len(df),
                n_null_cells=n_null,
                n_inf_cells=n_inf,
            )
            collector.record_feature_stats(bar_time, df)
            watcher.mark_processed(fp)
        except Exception as e:
            logger.warning(f"收盘后读取 {fp} 失败: {e}")

    # ---- 收盘后校验 ----
    validator = Validator(collector, trade_date)
    offline_path = os.path.join(config.OFFLINE_FEATURES_DIR, f"{trade_date}.parquet")

    if os.path.exists(offline_path):
        logger.info("离线文件已存在，立即执行校验")
        df_online = watcher.read_all_parquets()
        if df_online is not None:
            validator.run(df_online)
    else:
        logger.info(
            f"等待离线文件（最多 {config.POST_CLOSE_WAIT_MINUTES} 分钟）..."
        )
        waited = 0
        while waited < config.POST_CLOSE_WAIT_MINUTES * 60:
            time.sleep(60)
            waited += 60
            if os.path.exists(offline_path):
                logger.info(f"离线文件已出现 (等待 {waited}s)")
                df_online = watcher.read_all_parquets()
                if df_online is not None:
                    validator.run(df_online)
                break
        else:
            logger.warning("等待超时，跳过一致性校验")

    # ---- 汇总 ----
    recorded_final = collector.get_recorded_bar_times()
    logger.info(f"监控结束: 共记录 {len(recorded_final)} 个 bar")
    collector.close()


if __name__ == "__main__":
    main()
