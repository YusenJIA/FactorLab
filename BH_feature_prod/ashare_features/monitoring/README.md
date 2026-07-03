# A-Share Feature Pipeline Monitoring

A-share 特征工程流水线的轻量级监控子系统。在实时特征计算过程中，将每根 bar 的计算指标、放弃事件和特征级统计写入 SQLite，并通过 Grafana Dashboard 进行可视化。

**零外部依赖** — Python 端仅使用标准库（`sqlite3` / `smtplib` / `logging`），不引入 polars、pandas 等包，避免干扰主流水线的 import 链。

## 快速开始

### 1. 运行 Demo（生成模拟数据）

```bash
cd ~/ashare_feature
python -m monitoring.demo
```

在 `monitoring/demo_monitor.db` 中写入一个完整交易日（240 根 bar）的模拟指标。

### 2. 启动 Grafana 看板

```bash
cd ~/ashare_feature/monitoring/grafana
bash start.sh
```

首次运行自动下载 Grafana 11.4.0（清华镜像）和 SQLite 插件（~3 分钟），后续秒启动。

打开 http://localhost:3100 → 顶部选 **DemoMonitor** 数据源和 **trade_date** 即可看到图表。

### 3. 集成到实时计算

```python
from monitoring import MetricsCollector

collector = MetricsCollector(trade_date="2026-03-02")

# 每根 bar 计算完成后
collector.record_bar(
    bar_time="09:31:00",
    cost_seconds=23.5,
    n_symbols=5180,
    retry_count=0,
    status="ok",
    df_feat=df_features,   # polars DataFrame，用于统计 null/inf
)
collector.record_feature_stats("09:31:00", df_features)

# 放弃某根 bar 时
collector.record_abandon("10:16:00", retry_count=30, reason="symbols < expected")

collector.close()
```

## 架构总览

```
realtime_feature_cmp.py
  │
  │  record_bar() / record_abandon() / record_feature_stats()
  ▼
┌──────────────────────────────────────────────┐
│  monitoring (Python, stdlib only)             │
│                                               │
│  collector.py   MetricsCollector              │
│       │              ├─ record_bar()          │
│       │              ├─ record_abandon()      │
│       │              └─ record_feature_stats()│
│       │                                       │
│  alert.py       send_abandon_alert()  ──► 邮件│
│  config.py      DB 路径 / 邮件 / 特征列表     │
│  demo.py        模拟数据生成                   │
└───────────────┬──────────────────────────────┘
                │ SQLite (WAL mode)
                ▼
         monitor.db / demo_monitor.db
                │
                │ frser-sqlite-datasource
                ▼
┌──────────────────────────────────────────────┐
│  Grafana 11.4.0 (standalone binary, :3100)   │
│                                               │
│  10-panel Dashboard                           │
│    ├─ 计算耗时趋势      ├─ 标的数量           │
│    ├─ Null/Inf 统计     ├─ 重试次数           │
│    ├─ 放弃事件表        ├─ 放弃计数           │
│    ├─ 关键特征均值      ├─ 特征 Inf 计数      │
│    └─ 一致性概览        └─ 一致性详情         │
└──────────────────────────────────────────────┘
```

## 模块说明

### Python 模块

| 文件 | 职责 |
|------|------|
| `__init__.py` | 导出 `MetricsCollector` |
| `config.py` | 所有配置项：DB 路径、邮件、被跟踪特征列表 |
| `collector.py` | 核心采集器，管理 SQLite 连接和写入 |
| `alert.py` | 放弃告警邮件发送（SMTP_SSL） |
| `demo.py` | 独立 demo，生成 240 bar 模拟数据 |

### Grafana

| 文件 | 职责 |
|------|------|
| `grafana/start.sh` | 一键安装 + 启动脚本（无 sudo / 无 Docker） |
| `grafana/provisioning/datasources/sqlite.yml` | 数据源配置（prod + demo 两个 SQLite） |
| `grafana/provisioning/dashboards/dashboards.yml` | Dashboard 文件加载配置 |
| `grafana/dashboards/feature-monitoring.json` | 10 面板 Dashboard 定义 |

## SQLite 数据库设计

共 5 张表，`MetricsCollector.__init__()` 时自动建表。

### bar_metrics — 每 bar 计算指标

| 列 | 类型 | 说明 |
|----|------|------|
| `trade_date` | TEXT | 交易日（PK） |
| `bar_time` | TEXT | Bar 时间 HH:MM:SS（PK） |
| `cost_seconds` | REAL | 计算耗时（秒） |
| `n_symbols` | INTEGER | 标的数量 |
| `n_null_cells` | INTEGER | 全部特征列的 null 单元格总数 |
| `n_inf_cells` | INTEGER | 全部特征列的 inf 单元格总数 |
| `retry_count` | INTEGER | 数据拉取重试次数 |
| `status` | TEXT | `ok` 或 `abandon` |
| `created_at` | TEXT | 记录时间 |

### abandon_events — 放弃事件

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | INTEGER | 自增主键 |
| `trade_date` | TEXT | 交易日 |
| `bar_time` | TEXT | Bar 时间 |
| `retry_count` | INTEGER | 已重试次数 |
| `reason` | TEXT | 放弃原因 |
| `email_sent` | INTEGER | 邮件是否已发送（0/1） |
| `created_at` | TEXT | 记录时间 |

### feature_stats — 特征级统计

| 列 | 类型 | 说明 |
|----|------|------|
| `trade_date` | TEXT | 交易日（PK） |
| `bar_time` | TEXT | Bar 时间（PK） |
| `feature_name` | TEXT | 特征名（PK） |
| `mean_val` | REAL | 均值（排除 null/inf） |
| `std_val` | REAL | 标准差 |
| `min_val` | REAL | 最小值 |
| `max_val` | REAL | 最大值 |
| `null_count` | INTEGER | null 数量 |
| `inf_count` | INTEGER | inf 数量 |

### validation_summary — 离线 vs 实时一致性概览

| 列 | 类型 | 说明 |
|----|------|------|
| `trade_date` | TEXT | 交易日（PK） |
| `total_rows` | INTEGER | 总行数 |
| `total_columns` | INTEGER | 总列数 |
| `matched_columns` | INTEGER | 完全一致列数 |
| `diverged_columns` | INTEGER | 存在差异列数 |
| `mean_equal_ratio` | REAL | 平均一致率 |

### validation_columns — 特征级一致性详情

| 列 | 类型 | 说明 |
|----|------|------|
| `trade_date` | TEXT | 交易日（PK） |
| `feature_name` | TEXT | 特征名（PK） |
| `equal_ratio` | REAL | 一致率 |
| `max_abs_diff` | REAL | 最大绝对偏差 |
| `correlation` | REAL | 相关系数 |
| `is_known_divergent` | INTEGER | 是否为已知差异特征 |

## Dashboard 面板

Dashboard 提供两个变量：

- **datasource** — 切换 FeatureMonitor（生产）和 DemoMonitor（演示）
- **trade_date** — 从 `bar_metrics` 中动态获取可选交易日

10 个面板分四行排列：

| 行 | 左 | 右 |
|----|----|----|
| 1 | 计算耗时趋势（折线图，55s/58s 阈值） | 标的数量（折线图） |
| 2 | Null/Inf 统计（折线图，双系列） | 重试次数（柱状图） |
| 3 | 放弃事件表（表格） | 放弃计数（stat，1/5 阈值） |
| 4 | 关键特征均值（多线折线图） | 特征 Inf 计数（水平柱状图） |
| 5 | 一致性概览（stat，95%/99% 阈值） | 一致性详情（表格，按一致率升序） |

## 被跟踪特征

`config.TRACKED_FEATURES` 中定义了 12 个代表性特征，覆盖全部 7 个处理器及已知问题特征：

| 特征 | 来源 | 备注 |
|------|------|------|
| `round_1_distance` | RoundNumberProcessor | |
| `fomo_surge_5min` | FOMOFUDProcessor | |
| `buy_high_pattern_5min` | RetailPatternProcessor | |
| `price_clustering_5min` | HerdingProcessor | |
| `kyle_lambda` | MicrostructureProcessor | |
| `sentiment_overheat_5min` | SentimentCycleProcessor | |
| `trading_activity_5min` | AttentionProcessor | |
| `volume_fomo_241min` | FOMOFUDProcessor | 已知离线/实时差异大 |
| `panic_sell_241min` | FOMOFUDProcessor | 已知离线/实时差异大 |
| `social_attention_5min` | AttentionProcessor | 除零产生 inf |
| `social_attention_60min` | AttentionProcessor | 除零产生 inf |
| `social_attention_241min` | AttentionProcessor | 除零产生 inf |

## 配置

### 数据库路径

默认 `../monitor.db`（即 `ashare_feature/monitor.db`），可通过环境变量覆盖：

```bash
export MONITOR_DB_PATH=/path/to/custom.db
```

### 邮件告警

`config.py` 中配置，默认关闭：

```python
EMAIL_ENABLED = False
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SMTP_USER = ""          # QQ 邮箱
SMTP_PASSWORD = ""      # 授权码
EMAIL_RECEIVERS = []    # 收件人列表
```

开启后，每次 `record_abandon()` 会自动发送告警邮件。

### Grafana

| 配置项 | 值 |
|--------|----|
| 端口 | 3100 |
| 登录 | admin / admin |
| 匿名访问 | Viewer 角色 |
| 时区 | Asia/Shanghai |
| 安装路径 | `~/grafana` |
| 日志 | `monitoring/grafana/logs/grafana.log` |

## 设计要点

- **零依赖**：Python 端不 import polars/pandas，通过鸭子类型操作传入的 polars DataFrame
- **幂等写入**：`bar_metrics` 和 `feature_stats` 使用 `INSERT OR REPLACE`，重跑同一 bar 不会产生重复记录
- **WAL 模式**：SQLite 启用 WAL，支持 Grafana 读和 Python 写并发
- **无 Docker / 无 sudo**：Grafana 以独立二进制部署，deb 包通过 `dpkg-deb -x` 免 root 解压
- **国内加速**：Grafana 从清华镜像下载，SQLite 插件走 GitHub 代理
