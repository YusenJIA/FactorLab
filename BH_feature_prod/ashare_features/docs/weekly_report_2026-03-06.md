# Weekly Report 2026-03-06

## 1. 实盘特征计算框架 (ashare_feature)

**本周重点：V4 双模式实盘引擎开发 + pyarrow 类型修复**

### 完成事项

1. **pyarrow nullable 类型修复（v3 + v4）**
   - 发现 `pl.from_pandas()` 在 pandas 返回 nullable 类型（Int64, Float64, StringDtype）时失败
   - 新增 `_safe_pandas_convert()` 辅助函数，在 3 个调用点（`load_auction_data_today`, `load_auction_data_prev`, `_pandas_to_polars`）统一转换
   - v3 生产脚本已修复并在线运行

2. **V4 双模式引擎设计与实现**
   - 完成设计文档：`docs/plans/2026-03-06-v4-dual-mode-design.md`
   - 创建 `realtime_feature_cmp.py` + `run_test_api.sh`
   - **Instant 模式**（默认）：启动后跳到最新 bar 立即跟踪，主循环改为墙钟驱动，失败 bar 记入 `skipped_bars` 列表不阻塞后续计算
   - **Backfill 模式**：扫描已有 parquet，逐一补全缺失 bar，完成后自动退出
   - **修复 v3 核心 bug**：v3 中 `last_processed_time` 永远前进导致失败 bar 永久丢失；v4 中失败 bar 可在后续数据到达时自动恢复
   - 新增工具函数：`get_all_expected_bar_times()`, `get_existing_bar_times()`, `_load_today_bars()`, `_fill_missing_auction_symbols()`, `_forward_fill_missing_symbols()`

### 待办 / 风险
- V4 尚未在实盘测试，需盘后（15:00+）验证
- Config.OUTPUT_DIR 当前指向测试目录，上线前需改回 `/data/BH`
- V3 仍为���产版本，V4 替换需谨慎

---

## 2. Grafana 监控系统 (monitor)

**本周重点：监控框架 V2 从零搭建 + Grafana 集成 + 离线数据推送**

### 完成事项

1. **监控框架 V2 架构设计与实现（03-04 ~ 03-05）**
   - 从零创建独立 `monitor` 包，采用 **ABC 基类 + pipeline 子类** 模式
   - 4 个基类：`BaseMonitor`（主轮询循环 + ParquetWatcher）、`BaseCollector`（SQLite DB 管理）、`BaseLogParser`（增量日志解析）、`BaseValidator`（离线 vs 实时一致性校验）
   - BH pipeline 完整实现：`bh/collector.py`, `bh/log_parser.py`（7 个正则）, `bh/validator.py`, `bh/monitor.py`, `bh/config.py`
   - 零外部依赖（纯 stdlib），polars 通过 duck-typing 使用
   - 核心特性：upsert 双源写入（日志 + parquet 各提供部分字段）、断点续传、动态特征管理（`watch_features` CLI）

2. **Grafana + SQLite 集成（03-03 ~ 03-04）**
   - 完成 Grafana 11.4.0 + `frser-sqlite-datasource` v4.0.1 部署
   - 排查并解决 5 个关键踩坑问题（详见 `grafana-sqlite-troubleshooting.md`）：
     - 插件安全阻止列表需通过 `custom.ini` 配置（环境变量无效）
     - Dashboard JSON target 必须同时写 `queryText` + `rawQueryText`
     - 时序面板必须配置 `timeColumns`
   - 完成 Dashboard JSON provisioning 配置、RBAC 权限自动化脚本 (`setup_rbac.py`)

3. **文档体系建设（03-05）**
   - `monitoring-v2-deployment-guide.md`：完整部署指南，分 PM 和研究员两个角色手册
   - `BH_monitor_template.md`：BH 参考实现，��日志正则、表 schema、实测数据
   - `grafana-sqlite-troubleshooting.md`：踩坑记录
   - `directory-structure.md`：目录结构速查
   - CLAUDE.md + README.md
   - Claude Code skills：`/onboarding`（新用户引导）、`/launch-daily-monitoring`（每日启动）

4. **离线数据推送方案（03-05 ~ 03-06）**
   - 设计并实现 `scripts/push_offline.sh`：南科大服务器通过 SCP 推送 7 个 category parquet 到阿里云
   - 重构 `BaseValidator`：`_get_offline_path()` → `_load_offline_data()` + `offline_ready()`
   - `BHValidator` 支持加载并合并多 category parquet 文件
   - `BaseMonitor._post_close()` 适配新 validator API

5. **路径解耦（03-05）**
   - 将所有硬编码路径解耦为 config 变量 + 环境变量覆盖
   - DB 路径约定：`/data/{PIPELINE}/monitor.db`

### 待办 / 风险
- 离线推送脚本尚需在南科大服务器端部署并测试 SSH key
- 盘中实时监控完整流程需在交易日验证
- Dashboard 面板内容待根据实际数据微调
- 尚未配置 cron 自动化启动
