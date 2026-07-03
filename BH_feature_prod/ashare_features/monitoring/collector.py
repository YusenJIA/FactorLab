"""
MetricsCollector — 将实时特征计算的运行指标写入 SQLite

零外部依赖：仅使用 sqlite3 / logging / datetime（Python stdlib）
不 import 任何项目模块（factor_engine、dataloader 等）
"""
import sqlite3
import logging
from datetime import datetime
from typing import Optional

from . import config
from .alert import send_abandon_alert

logger = logging.getLogger("monitoring.collector")


class MetricsCollector:
    """每个交易日实例化一次，内部持有一个 SQLite 连接"""

    def __init__(self, trade_date: str, db_path: Optional[str] = None):
        self.trade_date = trade_date
        self.db_path = db_path or config.DB_PATH
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")  # 允许读写并发
        self._create_tables()
        logger.info(f"MetricsCollector 初始化: db={self.db_path}, date={trade_date}")

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------
    def _create_tables(self):
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS bar_metrics (
                trade_date   TEXT    NOT NULL,
                bar_time     TEXT    NOT NULL,
                cost_seconds REAL,
                n_symbols    INTEGER,
                n_null_cells INTEGER DEFAULT 0,
                n_inf_cells  INTEGER DEFAULT 0,
                retry_count  INTEGER DEFAULT 0,
                status       TEXT    DEFAULT 'ok',
                created_at   TEXT    DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (trade_date, bar_time)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS abandon_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date  TEXT    NOT NULL,
                bar_time    TEXT    NOT NULL,
                retry_count INTEGER,
                reason      TEXT,
                email_sent  INTEGER DEFAULT 0,
                created_at  TEXT    DEFAULT (datetime('now', 'localtime'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS feature_stats (
                trade_date   TEXT NOT NULL,
                bar_time     TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                mean_val     REAL,
                std_val      REAL,
                min_val      REAL,
                max_val      REAL,
                null_count   INTEGER DEFAULT 0,
                inf_count    INTEGER DEFAULT 0,
                created_at   TEXT DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (trade_date, bar_time, feature_name)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS validation_summary (
                trade_date       TEXT PRIMARY KEY,
                total_rows       INTEGER,
                total_columns    INTEGER,
                matched_columns  INTEGER,
                diverged_columns INTEGER,
                mean_equal_ratio REAL,
                created_at       TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS validation_columns (
                trade_date        TEXT NOT NULL,
                feature_name      TEXT NOT NULL,
                equal_ratio       REAL,
                max_abs_diff      REAL,
                correlation       REAL,
                is_known_divergent INTEGER DEFAULT 0,
                created_at        TEXT DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (trade_date, feature_name)
            )
        """)
        c.commit()

    # ------------------------------------------------------------------
    # 记录一个 bar 的计算指标
    # ------------------------------------------------------------------
    def record_bar(
        self,
        bar_time,
        cost_seconds: float,
        n_symbols: int,
        retry_count: int = 0,
        status: str = "ok",
        df_feat=None,
    ):
        """
        记录单个 bar 的计算指标。

        参数:
            bar_time: datetime 或 str，bar 的时间
            cost_seconds: 特征计算耗时（秒）
            n_symbols: 本 bar 的股票数量
            retry_count: 拉取重试次数
            status: 'ok' 或 'abandon'
            df_feat: 可选，polars DataFrame，用于统计 null/inf
        """
        bar_time_str = self._to_time_str(bar_time)
        n_null = 0
        n_inf = 0

        if df_feat is not None:
            try:
                n_null, n_inf = self._count_null_inf(df_feat)
            except Exception as e:
                logger.warning(f"统计 null/inf 失败: {e}")

        self._conn.execute(
            """INSERT OR REPLACE INTO bar_metrics
               (trade_date, bar_time, cost_seconds, n_symbols,
                n_null_cells, n_inf_cells, retry_count, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.trade_date, bar_time_str, cost_seconds, n_symbols,
             n_null, n_inf, retry_count, status),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 记录放弃事件 + 发邮件
    # ------------------------------------------------------------------
    def record_abandon(self, bar_time, retry_count: int, reason: str = ""):
        bar_time_str = self._to_time_str(bar_time)
        email_sent = 0

        try:
            ok = send_abandon_alert(self.trade_date, bar_time_str, retry_count, reason)
            email_sent = 1 if ok else 0
        except Exception as e:
            logger.warning(f"发送告警邮件异常: {e}")

        self._conn.execute(
            """INSERT INTO abandon_events
               (trade_date, bar_time, retry_count, reason, email_sent)
               VALUES (?, ?, ?, ?, ?)""",
            (self.trade_date, bar_time_str, retry_count, reason, email_sent),
        )
        # 同时在 bar_metrics 中标记
        self._conn.execute(
            """INSERT OR REPLACE INTO bar_metrics
               (trade_date, bar_time, cost_seconds, n_symbols,
                retry_count, status)
               VALUES (?, ?, 0, 0, ?, 'abandon')""",
            (self.trade_date, bar_time_str, retry_count),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 记录关键特征的统计量
    # ------------------------------------------------------------------
    def record_feature_stats(self, bar_time, df_feat, features=None):
        """
        对 tracked features 计算描述统计并写入 feature_stats 表。

        参数:
            bar_time: datetime 或 str
            df_feat: polars DataFrame（特征计算结果）
            features: 可选，要跟踪的特征名列表，默认用 config.TRACKED_FEATURES
        """
        bar_time_str = self._to_time_str(bar_time)
        features = features or config.TRACKED_FEATURES
        available = set(df_feat.columns)

        rows = []
        for feat in features:
            if feat not in available:
                continue
            try:
                col = df_feat[feat]
                # polars: null_count, 然后 drop_nulls 后统计
                null_count = col.null_count()
                col_clean = col.drop_nulls()
                if len(col_clean) == 0:
                    rows.append((self.trade_date, bar_time_str, feat,
                                 None, None, None, None, null_count, 0))
                    continue

                # 检测 inf: polars 中 is_infinite()
                inf_count = col_clean.is_infinite().sum()
                col_finite = col_clean.filter(~col_clean.is_infinite())

                if len(col_finite) == 0:
                    rows.append((self.trade_date, bar_time_str, feat,
                                 None, None, None, None, null_count, inf_count))
                    continue

                rows.append((
                    self.trade_date, bar_time_str, feat,
                    float(col_finite.mean()),
                    float(col_finite.std()),
                    float(col_finite.min()),
                    float(col_finite.max()),
                    null_count,
                    int(inf_count),
                ))
            except Exception as e:
                logger.warning(f"统计特征 {feat} 失败: {e}")

        if rows:
            self._conn.executemany(
                """INSERT OR REPLACE INTO feature_stats
                   (trade_date, bar_time, feature_name,
                    mean_val, std_val, min_val, max_val,
                    null_count, inf_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 局部更新 bar_metrics（日志/parquet 各提供部分字段）
    # ------------------------------------------------------------------
    def upsert_bar_metrics(self, bar_time, **fields):
        """
        插入或局部更新 bar_metrics 的指定字段。
        已有字段不会被 None 值覆盖。
        """
        bar_time_str = self._to_time_str(bar_time)
        row = self._conn.execute(
            "SELECT 1 FROM bar_metrics WHERE trade_date=? AND bar_time=?",
            (self.trade_date, bar_time_str),
        ).fetchone()

        if row is None:
            # 先插入默认行
            self._conn.execute(
                """INSERT INTO bar_metrics
                   (trade_date, bar_time) VALUES (?, ?)""",
                (self.trade_date, bar_time_str),
            )

        # 逐字段 UPDATE（仅非 None）
        allowed = {
            "cost_seconds", "n_symbols", "n_null_cells", "n_inf_cells",
            "retry_count", "status",
        }
        for k, v in fields.items():
            if k in allowed and v is not None:
                self._conn.execute(
                    f"UPDATE bar_metrics SET {k}=? "
                    "WHERE trade_date=? AND bar_time=?",
                    (v, self.trade_date, bar_time_str),
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 记录校验结果
    # ------------------------------------------------------------------
    def record_validation_summary(
        self,
        total_rows: int,
        total_columns: int,
        matched_columns: int,
        diverged_columns: int,
        mean_equal_ratio: float,
    ):
        self._conn.execute(
            """INSERT OR REPLACE INTO validation_summary
               (trade_date, total_rows, total_columns,
                matched_columns, diverged_columns, mean_equal_ratio)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self.trade_date, total_rows, total_columns,
             matched_columns, diverged_columns, mean_equal_ratio),
        )
        self._conn.commit()

    def record_validation_columns(self, rows: list):
        """
        rows: [(feature_name, equal_ratio, max_abs_diff, correlation, is_known_divergent), ...]
        """
        self._conn.executemany(
            """INSERT OR REPLACE INTO validation_columns
               (trade_date, feature_name, equal_ratio,
                max_abs_diff, correlation, is_known_divergent)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(self.trade_date, *r) for r in rows],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 查询已记录的 bar_time 列表（用于断点续传）
    # ------------------------------------------------------------------
    def get_recorded_bar_times(self) -> set:
        """返回当前 trade_date 已有 bar_metrics 记录的 bar_time 集合"""
        cur = self._conn.execute(
            "SELECT bar_time FROM bar_metrics WHERE trade_date=?",
            (self.trade_date,),
        )
        return {row[0] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _to_time_str(t) -> str:
        if isinstance(t, datetime):
            return t.strftime("%H:%M:%S")
        return str(t)

    @staticmethod
    def _count_null_inf(df) -> tuple:
        """统计整个 DataFrame 的 null 和 inf 数量（仅数值列）"""
        n_null = 0
        n_inf = 0
        for col_name in df.columns:
            col = df[col_name]
            n_null += col.null_count()
            try:
                n_inf += col.is_infinite().sum()
            except Exception:
                pass  # 非数值列跳过
        return int(n_null), int(n_inf)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
