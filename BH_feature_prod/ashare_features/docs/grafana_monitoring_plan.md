# Grafana 特征计算监控系统方案

## Context

实时特征计算脚本 (`realtime_feature_cmp_v3.py`) 目前仅有文件日志，无法实时观测计算健康状况。当出现"放弃本轮拉取"事件时无主动告警。盘后离线特征与实时特征的对比验证仍为手动操作（`validation_test.ipynb`），且数据分布在两台服务器上（阿里云 + 南科大），无自动化流程。

本方案目标：用最轻量的方式建立可视化监控 + 邮件告警 + 盘后自动化验证。

---

## 架构方案：Grafana + SQLite（最轻量）

```
┌───────────────── 阿里云服务器 (本机) ─────────────────┐
│                                                        │
│  realtime_feature_cmp_v3.py                            │
│    └─► MetricsCollector (新模块, 进程内)               │
│          ├─► SQLite: ~/ashare_feature/monitor.db       │
│          └─► 邮件告警 (smtplib, abandon时触发)         │
│                                                        │
│  Grafana OSS (免安装二进制, 端口 3100)                  │
│    └── SQLite datasource 插件 → 读取 monitor.db       │
│                                                        │
│  post_market_validator.py (cron, 盘后自动运行)         │
│    └── 读取离线 parquet + 实时 parquet → 对比写入 DB   │
└────────────────────────────────────────────────────────┘

┌───────────────── 南科大服务器 ────────────────────────┐
│  离线特征计算完成后                                     │
│    └── scp 上传 {date}.parquet 到阿里云                │
└────────────────────────────────────────────────────────┘
```

**为什么选 SQLite 而不是 Prometheus/InfluxDB：**
- `sqlite3` 是 Python 内置库，零依赖
- 无需额外守护进程，不会与现有 Dagster (端口 3000/3001) 冲突
- 数据天然持久化，重启不丢失
- Grafana 通过 `frser-sqlite-datasource` 社区插件即可查询

---

## 实施步骤

### 步骤 1：安装 Grafana（阿里云，无需 root）

- 下载 Grafana OSS standalone tar.gz，解压到 `~/grafana`
- 配置 `~/grafana/conf/custom.ini`：端口 3100，SMTP 邮件配置
- 安装 SQLite datasource 插件：`./bin/grafana cli plugins install frser-sqlite-datasource`
- 启动：`nohup ./bin/grafana server --homepath . &`

### 步骤 2：创建 `metrics_collector.py`（新文件）

> 文件：`/home/yusen/ashare_feature/metrics_collector.py`

SQLite 数据库表设计：

**`bar_metrics` 表**（每分钟一行，~241行/天）
| 字段 | 类型 | 说明 |
|------|------|------|
| trade_date | TEXT | 交易日 |
| bar_time | TEXT | Bar 时间 |
| cost_seconds | REAL | 计算耗时 |
| n_symbols | INTEGER | 股票数量 |
| n_null_cells | INTEGER | 空值总数 |
| n_inf_cells | INTEGER | inf 总数 |
| retry_count | INTEGER | 重试次数 |
| status | TEXT | 'ok' / 'abandon' |

**`abandon_events` 表**（放弃事件，希望很少出现）
| 字段 | 类型 | 说明 |
|------|------|------|
| trade_date | TEXT | 交易日 |
| bar_time | TEXT | 被放弃的 bar |
| retry_count | INTEGER | 重试次数 |
| reason | TEXT | 原因 |
| email_sent | INTEGER | 是否已发邮件 |

**`feature_stats` 表**（~20个关键特征的统计量）
| 字段 | 类型 | 说明 |
|------|------|------|
| trade_date, bar_time, feature_name | TEXT | 主键 |
| mean_val, std_val, min_val, max_val | REAL | 描述统计 |
| null_count, inf_count | INTEGER | 数据质量 |

监控的关键特征（~20个）：已知发散的 `volume_fomo_241min`、`panic_sell_241min`，产生 inf 的 `social_attention_*`，以及每个 Processor 的代表性特征。

**邮件告警**：使用 `smtplib` + QQ邮箱/163邮箱 SMTP，abandon 事件触发时立即发送。

### 步骤 3：修改 `realtime_feature_cmp_v3.py`（约 30 行改动）

> 文件：`/home/yusen/ashare_feature/realtime_feature_cmp_v3.py`

改动点：
1. **`run()` 函数开头**（~891行）：初始化 `MetricsCollector`
2. **`fetch_latest_bar_with_retry()`**：返回值增加 `retry_count`（改为返回 tuple）
3. **`if df_new is None:` 分支**（~1076行）：调用 `metrics.record_abandon()` 记录放弃事件 + 发送邮件
4. **特征计算完成后**（~1108行）：调用 `metrics.record_bar()` 和 `metrics.record_feature_stats()`

所有监控调用包裹在 `try/except` 中，监控失败不影响主流程。

### 步骤 4：创建 `post_market_validator.py`（新文件）

> 文件：`/home/yusen/ashare_feature/post_market_validator.py`

功能：
1. 检查 `offline_features/{date}.parquet` 是否已到达
2. 加载离线 parquet + 实时 parquet（`/data/BH/{date}/*.parquet`）
3. 复用 `validation_test.ipynb` 中 `validate_dataframes()` 的对比逻辑
4. 写入 SQLite：

**`validation_summary` 表**（每天一行）：总行数、匹配列数、发散列数、平均 equal_ratio

**`validation_columns` 表**（每列每天一行）：equal_ratio、max_abs_diff、correlation、is_known_divergent 标记

通过 cron 在盘后 16:00/16:30/17:00 自动运行（多次运行以等待离线文件到达）。

### 步骤 5：南科大服务器设置

- 配置 SSH key 免密登录到阿里云
- 创建 `upload_offline.sh`：离线计算完成后 `scp` 上传 parquet 到阿里云 `~/ashare_feature/offline_features/`
- cron 定时 15:50 和 16:20 执行
- **若网络不通**：备选方案为通过 OSS bucket 中转，或手动拷贝

### 步骤 6：配置 Grafana 仪表盘

**仪表盘 1：盘中实时监控**（自动刷新 30s）
| 面板 | 类型 | 内容 |
|------|------|------|
| 计算耗时趋势 | 时间序列 | cost_seconds 随 bar_time 变化，阈值线 55s |
| 放弃事件计数 | Stat | 当天 abandon 次数，0=绿 1+=红 |
| 今日进度 | Gauge | 已处理 bars / 241 |
| 数据质量表 | Table | 最近20条 bar 的 null/inf 统计 |
| 重试分布 | 柱状图 | retry_count 分布 |
| 关键特征统计 | 时间序列 | 问题特征的 mean/std 趋势 |

**仪表盘 2：盘后验证**
| 面板 | 类型 | 内容 |
|------|------|------|
| 验证状态 | Stat | 当天 mean_equal_ratio 百分比 |
| 历史趋势 | 时间序列 | 每日验证质量走势 |
| 发散特征明细 | Table | equal_ratio < 1.0 的特征，区分已知/未知发散 |
| 已知 vs 未知发散 | Pie | 帮助快速识别新增问题 |

---

## 涉及的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `ashare_feature/metrics_collector.py` | **新建** | MetricsCollector 类 + SQLite 写入 + 邮件告警 |
| `ashare_feature/post_market_validator.py` | **新建** | 盘后自动验证脚本 |
| `ashare_feature/realtime_feature_cmp_v3.py` | **修改** | 集成 MetricsCollector（~30行） |
| `ashare_feature/start_monitoring.sh` | **新建** | Grafana 启动脚本 |

参考文件（复用现有逻辑）：
- `ashare_feature/validation_test.ipynb` — 对比逻辑的参考实现
- `ashare_feature/CLAUDE.md` — 已知发散特征列表

---

## 验证方式

1. **MetricsCollector 单元验证**：独立运行，确认 SQLite 表创建正确、数据写入正常
2. **邮件告警验证**：手动触发一次 `send_email_alert()`，确认邮件收到
3. **盘中集成验证**：启动 realtime 脚本，观察 monitor.db 是否每分钟新增记录
4. **Grafana 验证**：打开仪表盘，确认面板有数据显示
5. **盘后验证**：手动运行 `post_market_validator.py --trade-date 2026-02-26`（用历史数据），确认验证结果写入 DB
