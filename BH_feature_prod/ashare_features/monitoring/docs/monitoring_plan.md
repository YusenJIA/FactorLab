# monitoring.py 实现方案

## Context

`realtime_feature_cmp_v3.py` 每分钟产出 parquet 文件和日志，但目前没有任何监控。需要一个**独立的旁路观察者** `monitoring.py`，通过轮询日志和 parquet 文件来采集指标写入 SQLite，供 Grafana 展示。不修改主流程任何代码。

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `monitoring/monitoring.py` | 新建 | 主监控脚本 |
| `monitoring/config.py` | 修改 | 新增监控相关配置项 |
| `monitoring/collector.py` | 修改 | 新增 `record_validation_summary()` 和 `record_validation_columns()` |
| `monitoring/run_monitoring.sh` | 新建 | nohup 启动脚本 |

## 整体架构

```
realtime_feature_cmp_v3.py (不改动)
  ├── 写入 --> /data/BH/{DATE}/{HHMMSS}.parquet
  └── 追加 --> logs/{DATE}_realtime_feature.log

monitoring.py (独立观察者, 每60秒轮询)
  ├── 扫描新 parquet --> 提取 n_symbols, null/inf, 特征统计
  ├── 追踪日志新行  --> 解析 cost_seconds, retry_count, abandon
  ├── 写入 --> monitor.db (bar_metrics, feature_stats, abandon_events)
  └── 收盘后 --> 读取离线 parquet, 执行一致性校验, 写入 validation 表
```

## 核心设计

### 1. LogParser 类 — 增量日志解析

- 维护 `_offset`（字节偏移），每次只读新增行（tail -f 模式）
- 用正则提取关键事件：

| 事件 | 正则匹配目标 | 提取字段 |
|------|-------------|---------|
| 重试 | `"第 {N}/{MAX} 次拉取, 期望 bar: {datetime}"` | retry_count, bar_time |
| 耗时 | `"特征计算耗时: {N}s"` | cost_seconds |
| 写入 | `"写入分片: .../{HHMMSS}.parquet, 共 {N} 行"` | bar_time (精确) |
| 放弃 | `"达到最大重试次数 {N}，放弃本轮拉取"` | retry_count |
| 错误 | `"get_price 异常 (attempt {N}): {reason}"` | abandon reason |
| 退出 | `"已过收盘时间，脚本退出"` | pipeline_exited 标志 |

- 以 `"写入分片"` 为 bar 完成的确定性事件（从文件名获取精确 bar_time）
- 维护跨轮询的状态：`_current_bar_time`, `_current_max_retry`

### 2. ParquetWatcher 类 — 新文件发现

- `_processed_files: Set[str]` 跟踪已处理的文件名
- `scan_existing()`: 启动时标记已有文件
- `check_new_files()`: 返回新增文件列表（仅在读取成功后才标记为已处理，避免写入中途读到损坏文件）
- `read_parquet(filename)`: 读取单个 parquet，返回 polars DataFrame
- `read_all_parquets()`: 收盘后 concat 所有文件，用于校验
- `filename_to_bar_time("093100.parquet")` → `"09:31:00"`

### 3. Validator 类 — 收盘后校验

复用 `validation_test.ipynb` 的逻辑：
1. 读取离线 parquet (`offline_features/{DATE}.parquet`)
2. 替换 inf/-inf 为 0，fill nan 为 0
3. Inner join on `(code, datetime)`
4. 逐列比较：float 列用 `abs(diff) < 1e-2` 判断相等
5. 计算 `equal_ratio`, `max_abs_diff`, `pearson_corr`
6. 标记已知差异特征（`volume_fomo_241min`, `panic_sell_241min`）
7. 写入 `validation_summary` + `validation_columns` 表

### 4. 主循环

```
main():
  初始化 MetricsCollector, LogParser, ParquetWatcher

  启动检查:
    DB有记录 + 目录有文件 → 断点续传（跳过已处理）
    DB空 + 目录有文件 → 回填模式（处理所有已有文件）
    都空 → 等待模式

  LOOP (每60秒):
    A. 读日志新行 → 解析事件 → 写 bar_metrics / abandon_events
    B. 扫描新 parquet → 读取 → 写 feature_stats, 更新 bar_metrics 的 n_symbols/null/inf
    C. sleep(60)
    退出条件: (pipeline_exited AND 过15:05) OR 过16:00

  收盘后校验:
    读全部实时 parquet
    等待离线文件（最多30分钟，每60秒检查一次）
    执行比对 → 写入 validation 表

  collector.close()
```

### 5. 日志与 parquet 数据合并策略

两个数据源更新同一条 `bar_metrics` 记录：
- **日志** 提供：`cost_seconds`, `retry_count`, `status`
- **Parquet** 提供：`n_symbols`, `n_null_cells`, `n_inf_cells`
- 先到的先写入，后到的用 `UPDATE` 补充（不覆盖已有字段）
- 主键 `(trade_date, bar_time)` 保证关联正确

## config.py 新增配置

```python
TRADE_DATE = os.environ.get("TRADE_DATE", "")  # 空则用当天日期
REALTIME_OUTPUT_DIR = "/data/BH"
LOG_DIR = "<parent>/logs"
OFFLINE_FEATURES_DIR = "<parent>/offline_features"
POLL_INTERVAL = 60  # 秒
POST_CLOSE_WAIT_MINUTES = 30
KNOWN_DIVERGENT_FEATURES = ["volume_fomo_241min", "panic_sell_241min"]
```

## collector.py 新增方法

```python
def record_validation_summary(self, total_rows, total_columns, matched_columns, diverged_columns, mean_equal_ratio)
def record_validation_columns(self, rows: list[tuple])  # (feature_name, equal_ratio, max_abs_diff, correlation, is_known_divergent)
```

## 验证方式

```bash
# 1. 用 demo 数据测试基本流程
python -m monitoring.demo  # 生成 demo_monitor.db

# 2. 对已有的历史交易日数据做回填测试
TRADE_DATE=2026-03-02 python -m monitoring.monitoring
# 预期：读取 /data/BH/2026-03-02/ 下所有 parquet + 对应日志，写入 monitor.db

# 3. 检查 Grafana 仪表盘
# 切换 datasource 到 FeatureMonitor (sqlite-prod)，选择对应 trade_date

# 4. 收盘后校验（如果 offline_features/2026-03-02.parquet 存在）
# 检查 validation_summary 和 validation_columns 表是否有数据
```

---

## 实现总结（2026-03-02 完成）

### 实际文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `monitoring/config.py` | 修改 | 新增 `TRADE_DATE`, `REALTIME_OUTPUT_DIR`, `LOG_DIR`, `OFFLINE_FEATURES_DIR`, `POLL_INTERVAL`, `POST_CLOSE_WAIT_MINUTES`, `KNOWN_DIVERGENT_FEATURES` |
| `monitoring/collector.py` | 修改 | 新增 `upsert_bar_metrics()`, `record_validation_summary()`, `record_validation_columns()`, `get_recorded_bar_times()` |
| `monitoring/monitoring.py` | 新建 | LogParser + ParquetWatcher + Validator + 主循环（约 400 行） |
| `monitoring/run_monitoring.sh` | 新建 | nohup 启动脚本 |
| `monitoring/docs/monitoring_plan.md` | 新建 | 本方案文档 |

### collector.py 新增方法（实际）

```python
# 局部更新 bar_metrics（日志/parquet 各提供部分字段，先到先写，后到 UPDATE 补充）
def upsert_bar_metrics(self, bar_time, **fields)

# 收盘后校验结果写入
def record_validation_summary(self, total_rows, total_columns, matched_columns, diverged_columns, mean_equal_ratio)
def record_validation_columns(self, rows: list)  # [(feature_name, equal_ratio, max_abs_diff, correlation, is_known_divergent), ...]

# 查询已记录的 bar_time（用于断点续传判断）
def get_recorded_bar_times(self) -> set
```

### 测试结果

以 `2026-02-24` 真实数据验证：

| 测试项 | 结果 |
|--------|------|
| **LogParser** | 解析 241 个 bar_complete + exit 事件，retry_count/cost_seconds 提取正确 |
| **ParquetWatcher** | 扫描 241 个 parquet 文件，`filename_to_bar_time` 转换正确 |
| **集成测试（日志+parquet→collector）** | 241 条 bar_metrics（cost 来自日志，symbols/null/inf 来自 parquet），36 条 feature_stats |
| **bar 09:30:00 合并示例** | `cost=22.346s, symbols=5190, null=0, inf=99, retry=0, status=ok` |
| **Validator** | offline vs realtime 对比：1,250,790 行 × 581 列，`mean_equal_ratio=0.999999`，0 发散列 |

### 使用方式

```bash
# 实时监控（当天，配合主流程运行）
cd /home/yusen/ashare_feature
./monitoring/run_monitoring.sh

# 指定交易日回填历史数据
TRADE_DATE=2026-02-24 python -m monitoring.monitoring

# 查看监控日志
tail -f logs/20260302_monitoring.log
```

### 三种启动模式

| 条件 | 模式 | 行为 |
|------|------|------|
| DB 有记录 + 目录有文件 | 断点续传 | 跳过已处理文件，日志跳到末尾只关注新行 |
| DB 空 + 目录有文件 | 回填模式 | 处理目录下所有已有文件 + 从头解析日志 |
| 都空 | 等待模式 | 轮询等待新文件和日志出现 |
