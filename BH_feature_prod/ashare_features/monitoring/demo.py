#!/usr/bin/env python3
"""
监控系统独立 Demo — 零外部依赖，模拟一个交易日的指标写入

运行：python -m monitoring.demo   (从 ashare_feature/ 目录)
或：  python monitoring/demo.py
"""
import os
import sys
import sqlite3
import random
import math

# 确保 import 能找到 monitoring 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.collector import MetricsCollector


def generate_bar_times():
    """生成一个交易日的所有 bar 时间 (09:31 ~ 11:30, 13:01 ~ 15:00)"""
    times = []
    # 上午: 09:31 ~ 11:30 (120 bars)
    for h in range(9, 12):
        start_m = 31 if h == 9 else 0
        end_m = 31 if h == 11 else 60
        for m in range(start_m, end_m):
            times.append(f"{h:02d}:{m:02d}:00")
    # 下午: 13:01 ~ 15:00 (120 bars)
    for h in range(13, 16):
        start_m = 1 if h == 13 else 0
        end_m = 1 if h == 15 else 60
        for m in range(start_m, end_m):
            times.append(f"{h:02d}:{m:02d}:00")
    return times


def run_demo():
    trade_date = "2026-03-02"
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_monitor.db")

    # 清理旧 demo 数据库
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"=== 监控系统 Demo ===")
    print(f"交易日: {trade_date}")
    print(f"数据库: {db_path}")
    print()

    collector = MetricsCollector(trade_date=trade_date, db_path=db_path)

    bar_times = generate_bar_times()
    print(f"模拟 {len(bar_times)} 个 bar 的指标数据...")

    # 模拟每个 bar
    abandon_count = 0
    for i, bt in enumerate(bar_times):
        # 模拟：大部分 bar 正常，2 个 bar 放弃
        if i in (45, 180):  # 10:16 和 14:21 模拟放弃
            collector.record_abandon(
                bar_time=bt,
                retry_count=30,
                reason="max retry exceeded, symbols < expected",
            )
            abandon_count += 1
            continue

        # 正常 bar
        cost = random.gauss(25, 8)  # 平均 25s，标准差 8s
        cost = max(5, min(cost, 58))
        n_symbols = random.randint(5100, 5220)
        retry = random.choices([0, 0, 0, 1, 2], weights=[70, 10, 5, 10, 5])[0]
        n_null = random.randint(0, 500)
        n_inf = random.randint(0, 50)

        collector.record_bar(
            bar_time=bt,
            cost_seconds=round(cost, 3),
            n_symbols=n_symbols,
            retry_count=retry,
            status="ok",
            df_feat=None,  # demo 中不传 DataFrame，直接用模拟值
        )

        # 手动更新 null/inf（因为 demo 无真实 DataFrame）
        collector._conn.execute(
            "UPDATE bar_metrics SET n_null_cells=?, n_inf_cells=? "
            "WHERE trade_date=? AND bar_time=?",
            (n_null, n_inf, trade_date, bt),
        )

    # 模拟 feature_stats（手造几个关键特征的统计量）
    tracked = [
        "volume_fomo_241min", "panic_sell_241min",
        "social_attention_5min", "kyle_lambda",
        "fomo_surge_5min", "round_1_distance",
    ]
    sample_times = [bar_times[0], bar_times[60], bar_times[120], bar_times[-1]]
    rows = []
    for bt in sample_times:
        for feat in tracked:
            rows.append((
                trade_date, bt, feat,
                round(random.gauss(0, 1), 4),    # mean
                round(abs(random.gauss(1, 0.5)), 4),  # std
                round(random.gauss(-3, 1), 4),   # min
                round(random.gauss(3, 1), 4),    # max
                random.randint(0, 100),           # null_count
                random.randint(0, 20) if "social" in feat else 0,  # inf_count
            ))
    collector._conn.executemany(
        """INSERT OR REPLACE INTO feature_stats
           (trade_date, bar_time, feature_name,
            mean_val, std_val, min_val, max_val, null_count, inf_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    collector._conn.commit()

    print(f"  写入 {len(bar_times) - abandon_count} 条 bar_metrics")
    print(f"  写入 {abandon_count} 条 abandon_events")
    print(f"  写入 {len(rows)} 条 feature_stats")
    print()

    # ---- 验证查询 ----
    conn = sqlite3.connect(db_path)

    print("--- bar_metrics (前 5 条) ---")
    for row in conn.execute(
        "SELECT bar_time, cost_seconds, n_symbols, n_null_cells, n_inf_cells, retry_count, status "
        "FROM bar_metrics WHERE trade_date=? ORDER BY bar_time LIMIT 5",
        (trade_date,),
    ):
        print(f"  {row}")

    print()
    print("--- abandon_events ---")
    for row in conn.execute(
        "SELECT bar_time, retry_count, reason FROM abandon_events WHERE trade_date=?",
        (trade_date,),
    ):
        print(f"  {row}")

    print()
    print("--- feature_stats (sample) ---")
    for row in conn.execute(
        "SELECT bar_time, feature_name, mean_val, std_val, null_count, inf_count "
        "FROM feature_stats WHERE trade_date=? LIMIT 8",
        (trade_date,),
    ):
        print(f"  {row}")

    print()
    print("--- 统计摘要 ---")
    (total_bars,) = conn.execute(
        "SELECT COUNT(*) FROM bar_metrics WHERE trade_date=?", (trade_date,)
    ).fetchone()
    (ok_bars,) = conn.execute(
        "SELECT COUNT(*) FROM bar_metrics WHERE trade_date=? AND status='ok'", (trade_date,)
    ).fetchone()
    (abandon_bars,) = conn.execute(
        "SELECT COUNT(*) FROM bar_metrics WHERE trade_date=? AND status='abandon'", (trade_date,)
    ).fetchone()
    row = conn.execute(
        "SELECT AVG(cost_seconds), MAX(cost_seconds), AVG(n_symbols) "
        "FROM bar_metrics WHERE trade_date=? AND status='ok'",
        (trade_date,),
    ).fetchone()
    print(f"  总 bars: {total_bars} (ok={ok_bars}, abandon={abandon_bars})")
    print(f"  平均耗时: {row[0]:.2f}s, 最大耗时: {row[1]:.2f}s, 平均标的数: {row[2]:.0f}")

    # Grafana 常用查询示例
    print()
    print("=== Grafana 查询示例 (可直接用于 SQLite datasource) ===")
    print()
    print("-- 计算耗时趋势:")
    print("SELECT bar_time as time, cost_seconds FROM bar_metrics")
    print("  WHERE trade_date='2026-03-02' AND status='ok' ORDER BY bar_time")
    print()
    print("-- 放弃事件计数:")
    print("SELECT COUNT(*) as abandon_count FROM abandon_events")
    print("  WHERE trade_date='2026-03-02'")
    print()
    print("-- 关键特征 inf 趋势:")
    print("SELECT bar_time as time, feature_name, inf_count FROM feature_stats")
    print("  WHERE trade_date='2026-03-02' AND inf_count > 0 ORDER BY bar_time")

    conn.close()
    collector.close()
    print()
    print(f"Demo 完成! 数据库文件: {db_path}")


if __name__ == "__main__":
    run_demo()
