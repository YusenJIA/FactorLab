# V4 Dual-Mode Realtime Feature Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `realtime_feature_cmp.py` with two modes — instant (skip to latest bar immediately) and backfill (fill all missing bars) — while fixing the skipped-bar bug from v3.

**Architecture:** Single-file script with `--mode instant|backfill` CLI arg. Both modes share all utility functions from v3. `run_instant()` and `run_backfill()` are the two entry points. The main loop in instant mode is driven by wall-clock time instead of `last_processed_time`, and failed bars are tracked in a `skipped_bars` list for later recovery.

**Tech Stack:** Python, polars, argparse. Same dependencies as v3.

**Source:** Copy from `realtime_feature_cmp_v3.py` (v3 with `_safe_pandas_convert` fix applied).

---

### Task 1: Create v4 skeleton — copy v3 + add argparse

**Files:**
- Create: `/home/yusen/ashare_feature/realtime_feature_cmp.py`
- Reference: `/home/yusen/ashare_feature/realtime_feature_cmp_v3.py`

**Step 1:** Copy v3 to v4

```bash
cp /home/yusen/ashare_feature/realtime_feature_cmp_v3.py /home/yusen/ashare_feature/realtime_feature_cmp.py
```

**Step 2:** Update docstring at top of v4 to reflect dual-mode design:

```python
"""
实时股票特征计算引擎 V4 (双模式)
=================================
两种运行模式：
  --mode instant   (默认) 跳到最新时刻立即跟踪，中间缺失bar不阻塞
  --mode backfill  补全当天所有缺失bar后退出
"""
```

**Step 3:** Add `argparse` at the `if __name__ == "__main__"` block (replace current entry point):

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V4 realtime feature engine")
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
```

**Step 4:** Delete the old `run()` function (v3 lines 908-1189). We will replace it with `run_instant()` and `run_backfill()` in subsequent tasks.

---

### Task 2: Add helper functions — `get_all_expected_bar_times` and `get_existing_bar_times`

**Files:**
- Modify: `/home/yusen/ashare_feature/realtime_feature_cmp.py`

**Step 1:** Add `get_all_expected_bar_times()` after the `next_bar_time()` function:

```python
def get_all_expected_bar_times(trade_date: str, end_time: Optional[datetime] = None) -> list[datetime]:
    """
    生成从 09:30 到 end_time (或 15:00) 的完整 bar 时间序列。
    包括: 09:30, 09:31, ..., 11:30, 13:01, 13:02, ..., 15:00
    """
    morning_start = datetime.strptime(f"{trade_date} 09:30:00", "%Y-%m-%d %H:%M:%S")
    morning_end = datetime.strptime(f"{trade_date} 11:30:00", "%Y-%m-%d %H:%M:%S")
    afternoon_start = datetime.strptime(f"{trade_date} 13:01:00", "%Y-%m-%d %H:%M:%S")
    afternoon_end = datetime.strptime(f"{trade_date} 15:00:00", "%Y-%m-%d %H:%M:%S")

    if end_time is None:
        end_time = afternoon_end

    bars = []
    # Morning: 09:30 ~ 11:30
    t = morning_start
    while t <= morning_end and t <= end_time:
        bars.append(t)
        t += timedelta(minutes=1)
    # Afternoon: 13:01 ~ 15:00
    t = afternoon_start
    while t <= afternoon_end and t <= end_time:
        bars.append(t)
        t += timedelta(minutes=1)

    return bars
```

**Step 2:** Add `get_existing_bar_times()`:

```python
def get_existing_bar_times(output_dir: str, trade_date: str) -> set[datetime]:
    """
    扫描输出目录中已有的 parquet 文件名，返回已处理的 bar 时间集合。
    """
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
```

---

### Task 3: Implement `run_instant()` — instant mode

**Files:**
- Modify: `/home/yusen/ashare_feature/realtime_feature_cmp.py`

This is the core new function. Key differences from v3 `run()`:
- On startup: loads full today data but only computes features for the latest bar
- Main loop: `expected_time` based on wall-clock, not `last_processed_time + 1`
- Failed bars go to `skipped_bars` list, recovered when data becomes available

**Step 1:** Add the `_load_today_bars()` helper (used by both modes to fetch all today's bars in one shot):

```python
def _load_today_bars(symbols_arr: list) -> Optional[pl.DataFrame]:
    """拉取当天从开盘到现在的所有 1min bar 数据。"""
    start_date = datetime.strptime(f"{Config.TRADE_DATE} 08:50:00", "%Y-%m-%d %H:%M:%S")
    try:
        df = get_price(
            symbols_arr,
            start_date=start_date,
            end_date=datetime.now(),
            frequency='1m',
            count=2000000,
        )
        if not isinstance(df, pl.DataFrame):
            df = _pandas_to_polars(df)
        logger.info(f"当天全量数据拉取完成: {len(df)} 行, 时间范围 {df['time'].min()} ~ {df['time'].max()}")
        return df
    except Exception as e:
        logger.error(f"拉取当天全量数据失败: {e}")
        return None
```

**Step 2:** Add `run_instant()`:

```python
def run_instant(symbols_arr: list, expected_min_symbols: int = 100):
    """
    Instant Mode: 跳到最新时刻立即跟踪，中间缺失 bar 不阻塞。
    """
    logger.info("=" * 60)
    logger.info("V4 Instant Mode 启动")
    logger.info("=" * 60)

    timing_path = f"/home/yusen/ashare_feature/realtime_features/{Config.TRADE_DATE}_feature_timing.csv"
    pipeline = FeaturePipeline()
    skipped_bars = []  # 记录拉取失败的 bar 时间

    # ---- Step 1: 加载历史数据 ----
    df_history, prev_trade_date_1 = load_historical_data(symbols_arr, Config.TRADE_DATE)

    # ---- Step 2: 等待到 09:26 再拉取集合竞价 ----
    auction_start = datetime.strptime(f"{Config.TRADE_DATE} 09:26:00", "%Y-%m-%d %H:%M:%S")
    wait_sec = (auction_start - datetime.now()).total_seconds()
    if wait_sec > 0:
        logger.info(f"等待 {wait_sec:.0f}s 到 09:26...")
        time.sleep(wait_sec)

    # ---- Step 3: 拉取集合竞价 + 补全缺失标的 ----
    df_auction = load_auction_data_today(symbols_arr)
    df_auction = _fill_missing_auction_symbols(
        df_auction, df_history, symbols_arr, prev_trade_date_1
    )

    # ---- Step 4: 拉取当天已有全量 bar ----
    df_today_bars = _load_today_bars(symbols_arr)

    # ---- Step 5: 构建 df_all ----
    parts = [df_history, df_auction]
    if df_today_bars is not None and len(df_today_bars) > 0:
        parts.append(df_today_bars)
    df_all = pl.concat(parts)
    df_all = df_all.unique(subset=["symbol", "time"], keep="last")
    df_all = df_all.sort(["symbol", "time"])

    # ---- Step 6: 只对最新时刻计算特征 ----
    t_start = time.time()
    df_feat = compute_features(pipeline, df_all)
    t_delta = time.time() - t_start

    latest_time = df_all["time"].max()
    append_to_output(df_feat, Config.OUTPUT_DIR, Config.TRADE_DATE)
    append_timing_record(timing_path, latest_time, t_delta)
    logger.info(f"初始特征计算完成 (最新 bar: {latest_time}), 耗时 {t_delta:.3f}s")

    last_processed_time = latest_time

    # ---- Step 7: 主循环 (墙钟驱动) ----
    while True:
        now = datetime.now()

        if now.strftime("%H:%M") >= Config.AFTERNOON_END:
            logger.info("已过收盘时间，脚本退出")
            break

        if not is_trading_time(now):
            logger.info(f"当前非交易时段 ({now.strftime('%H:%M')}), 等待中...")
            time.sleep(10)
            continue

        # 基于墙钟确定期望 bar
        expected_time = now.replace(second=0, microsecond=0)
        # 如果 expected_time 已经处理过，等到下一分钟
        if expected_time <= last_processed_time:
            sleep_sec = 60 - now.second + Config.POLL_OFFSET_SEC
            time.sleep(max(sleep_sec, 1))
            continue

        # 等待到整分钟 + offset
        target_wall = expected_time + timedelta(seconds=Config.POLL_OFFSET_SEC)
        wait_sec = (target_wall - datetime.now()).total_seconds()
        if wait_sec > 0:
            logger.info(f"等待 {wait_sec:.1f}s 到 {target_wall.strftime('%H:%M:%S')} 再拉取 bar {expected_time.strftime('%H:%M')}")
            time.sleep(max(wait_sec, 0))

        # ---- 拉取数据 ----
        df_new = fetch_latest_bar_with_retry(
            symbols_arr, expected_time, expected_min_symbols
        )

        if df_new is None:
            skipped_bars.append(expected_time)
            logger.warning(
                f"Bar {expected_time} 拉取失败，记入待补列表 "
                f"(累计 {len(skipped_bars)} 个待补 bar)"
            )
            continue  # 不更新 last_processed_time

        # ---- 增量提取 (包含 skipped bars 的恢复) ----
        # 检查 skipped bars 是否在本次数据中
        available_times = set(df_new["time"].unique().to_list())
        recovered = [t for t in skipped_bars if t in available_times]
        if recovered:
            logger.info(f"从本次数据中恢复了 {len(recovered)} 个之前失败的 bar: {recovered}")
            skipped_bars = [t for t in skipped_bars if t not in available_times]

        # 提取所有 > last_processed_time 的增量
        df_incr = df_new.filter(pl.col("time") > last_processed_time)
        if len(df_incr) == 0 and not recovered:
            logger.info("本轮无新增数据")
            continue

        # Append 到 df_all
        if len(df_incr) > 0:
            df_incr = df_incr.select(df_all.columns)
            df_all = pl.concat([df_all, df_incr])
            df_all = df_all.unique(subset=["symbol", "time"], keep="last")
            df_all = df_all.sort(["symbol", "time"])

        # ---- 缺失标的 forward fill ----
        df_all = _forward_fill_missing_symbols(df_all, symbols_arr)

        logger.info(
            f"内存数据更新: 总计 {len(df_all)} 行, "
            f"{df_all['symbol'].n_unique()} 只标的, "
            f"时间范围 {df_all['time'].min()} ~ {df_all['time'].max()}"
        )

        # ---- 特征计算 + 写入 ----
        # 确定需要计算特征的 bar 列表
        existing_bars = get_existing_bar_times(Config.OUTPUT_DIR, Config.TRADE_DATE)
        new_times = sorted(set(df_incr["time"].unique().to_list()) | set(recovered))
        bars_to_compute = [t for t in new_times if t not in existing_bars]

        if not bars_to_compute:
            logger.info("所有 bar 已有输出，跳过计算")
            last_processed_time = df_all["time"].max()
            continue

        for bar_time in sorted(bars_to_compute):
            t_start = time.time()
            df_for_feat = df_all.filter(pl.col("time") <= bar_time)
            df_feat = compute_features(pipeline, df_for_feat)
            append_to_output(df_feat, Config.OUTPUT_DIR, Config.TRADE_DATE)
            t_delta = time.time() - t_start
            append_timing_record(timing_path, bar_time, t_delta)
            logger.info(f"Bar {bar_time} 特征计算完成, 耗时 {t_delta:.3f}s")

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
```

**Step 3:** Extract `_fill_missing_auction_symbols()` from v3 lines 944-985 (the auction symbol fill logic, currently inline in `run()`):

```python
def _fill_missing_auction_symbols(
    df_auction: pl.DataFrame,
    df_history: pl.DataFrame,
    symbols_arr: list,
    prev_trade_date_1: str,
) -> pl.DataFrame:
    """补全集合竞价缺失标的：用前一交易日收盘价填充。"""
    auction_symbols = set(df_auction["symbol"].to_list())
    expected_symbols = set(s[:6] for s in symbols_arr)
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
```

**Step 4:** Extract `_forward_fill_missing_symbols()` from v3 lines 1114-1147:

```python
def _forward_fill_missing_symbols(df_all: pl.DataFrame, symbols_arr: list) -> pl.DataFrame:
    """缺失标的 forward fill（仅当缺失比例 <= 1% 时）。"""
    latest_time = df_all["time"].max()
    all_syms = set(s[:6] for s in symbols_arr)
    current_syms = set(df_all.filter(pl.col("time") == latest_time)["symbol"].to_list())
    missing = all_syms - current_syms

    if 0 < len(missing) <= int(len(symbols_arr) * 0.01):
        prev_time = df_all.filter(pl.col("time") < latest_time)["time"].max()
        if prev_time is not None:
            df_fill = (
                df_all.filter(
                    (pl.col("time") == prev_time) & pl.col("symbol").is_in(list(missing))
                ).with_columns(pl.lit(latest_time).alias("time"))
            )
            if len(df_fill) > 0:
                df_all = pl.concat([df_all, df_fill])
                df_all = df_all.unique(subset=["symbol", "time"], keep="last")
                df_all = df_all.sort(["symbol", "time"])
                logger.info(f"Forward fill: 填充 {len(df_fill)} 只标的 (缺失 {len(missing)}/{len(symbols_arr)})")

    return df_all
```

---

### Task 4: Implement `run_backfill()` — backfill mode

**Files:**
- Modify: `/home/yusen/ashare_feature/realtime_feature_cmp.py`

```python
def run_backfill(symbols_arr: list, expected_min_symbols: int = 100):
    """
    Backfill Mode: 补全当天所有缺失 bar 后退出。
    """
    logger.info("=" * 60)
    logger.info("V4 Backfill Mode 启动")
    logger.info("=" * 60)

    timing_path = f"/home/yusen/ashare_feature/realtime_features/{Config.TRADE_DATE}_feature_timing.csv"
    pipeline = FeaturePipeline()

    # ---- Step 1: 计算缺失 bar 列表 ----
    now = datetime.now()
    end_time = min(
        now,
        datetime.strptime(f"{Config.TRADE_DATE} 15:00:00", "%Y-%m-%d %H:%M:%S")
    )
    all_expected = get_all_expected_bar_times(Config.TRADE_DATE, end_time)
    existing = get_existing_bar_times(Config.OUTPUT_DIR, Config.TRADE_DATE)
    missing_bars = sorted([t for t in all_expected if t not in existing])

    if not missing_bars:
        logger.info("无缺失 bar，Backfill 无需执行，退出")
        return

    logger.info(
        f"缺失 {len(missing_bars)} 个 bar: "
        f"{missing_bars[0].strftime('%H:%M')} ~ {missing_bars[-1].strftime('%H:%M')}"
    )

    # ---- Step 2: 加载全量数据 ----
    df_history, prev_trade_date_1 = load_historical_data(symbols_arr, Config.TRADE_DATE)

    df_auction = load_auction_data_today(symbols_arr)
    df_auction = _fill_missing_auction_symbols(
        df_auction, df_history, symbols_arr, prev_trade_date_1
    )

    df_today_bars = _load_today_bars(symbols_arr)

    parts = [df_history, df_auction]
    if df_today_bars is not None and len(df_today_bars) > 0:
        parts.append(df_today_bars)
    df_all = pl.concat(parts)
    df_all = df_all.unique(subset=["symbol", "time"], keep="last")
    df_all = df_all.sort(["symbol", "time"])

    logger.info(
        f"全量数据: {len(df_all)} 行, "
        f"时间范围 {df_all['time'].min()} ~ {df_all['time'].max()}"
    )

    # ---- Step 3: 逐 bar 补全 ----
    available_times = set(df_all["time"].unique().to_list())
    skipped = []

    for i, bar_time in enumerate(missing_bars):
        if bar_time not in available_times:
            logger.warning(f"Bar {bar_time} 在行情数据中不存在，跳过")
            skipped.append(bar_time)
            continue

        t_start = time.time()
        df_for_feat = df_all.filter(pl.col("time") <= bar_time)
        df_feat = compute_features(pipeline, df_for_feat)
        append_to_output(df_feat, Config.OUTPUT_DIR, Config.TRADE_DATE)
        t_delta = time.time() - t_start
        append_timing_record(timing_path, bar_time, t_delta)
        logger.info(
            f"[{i+1}/{len(missing_bars)}] Bar {bar_time.strftime('%H:%M:%S')} "
            f"补全完成, 耗时 {t_delta:.3f}s"
        )

    # ---- 结束 ----
    if skipped:
        logger.warning(f"有 {len(skipped)} 个 bar 无行情数据无法补全: {skipped}")
    logger.info("=" * 60)
    logger.info(f"Backfill Mode 完成, 共补全 {len(missing_bars) - len(skipped)}/{len(missing_bars)} 个 bar")
    logger.info("=" * 60)
```

---

### Task 5: Create launcher script + cleanup

**Files:**
- Create: `/home/yusen/ashare_feature/run_test_api.sh`

```bash
#!/bin/bash
# run_test_api.sh

MODE=${1:-instant}
LOG_DIR="/home/yusen/ashare_feature/logs"
TRADE_DATE=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/${TRADE_DATE}_realtime_feature_v4.log"

mkdir -p ${LOG_DIR}

nohup python /home/yusen/ashare_feature/realtime_feature_cmp.py --mode ${MODE} >> ${LOG_FILE} 2>&1 &

echo "V4 脚本已启动 (mode=${MODE}), PID: $!"
echo "日志文件: ${LOG_FILE}"
```

Usage:
```bash
bash run_test_api.sh           # instant mode (default)
bash run_test_api.sh backfill  # backfill mode
```

---

### Task 6: Delete old `run()` and `fetch_missing_bars_after_restart()`

**Files:**
- Modify: `/home/yusen/ashare_feature/realtime_feature_cmp.py`

Remove the v3 `run()` function and `fetch_missing_bars_after_restart()` which are no longer needed — their logic has been replaced by `run_instant()` and `run_backfill()`.

---

### Verification

1. **Syntax check:** `python -c "import py_compile; py_compile.compile('/home/yusen/ashare_feature/realtime_feature_cmp.py', doraise=True)"`
2. **Help text:** `python /home/yusen/ashare_feature/realtime_feature_cmp.py --help`
3. **Backfill dry run (during market hours):** `python /home/yusen/ashare_feature/realtime_feature_cmp.py --mode backfill` — should report missing bars and start filling
4. **Instant mode:** `bash run_test_api.sh` — should skip to latest bar and enter main loop
