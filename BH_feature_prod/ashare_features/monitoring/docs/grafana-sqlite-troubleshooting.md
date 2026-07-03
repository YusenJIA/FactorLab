# Grafana + SQLite 插件踩坑记录

> 日期：2026-03-03
> 环境：Grafana 11.4.0 + frser-sqlite-datasource v4.0.1（非 sudo 本地安装）

## 背景

monitoring 模块作为旁路观察者，解析实盘日志 + 读取 parquet 输出，将指标写入 SQLite (`monitor.db`)。
Grafana 通过 frser-sqlite-datasource 插件读取该 SQLite 展示 Dashboard。

**问题**：Dashboard 所有面板（计算耗时、标的数量、Null/Inf 统计、重试次数）均为空白，
显示 "Data is missing a time field" 或无数据。

---

## 排查过程

### 1. 确认数据存在

```bash
sqlite3 /home/yusen/ashare_feature/monitor.db \
  "SELECT COUNT(*) FROM bar_metrics WHERE trade_date='2026-03-03'"
# 返回 30（正常）
```

数据已入库，SQLite 本身没问题。

### 2. Grafana 数据源健康检查

```bash
curl -s -X POST http://localhost:3100/api/datasources/1/health -u admin:admin
# {"message":"Data source is working","status":"OK"}
```

健康检查通过，但实际查询返回空数据。**健康检查不代表查询可用**。

### 3. API 查询始终返回空（第一个坑）

```bash
curl -s -X POST http://localhost:3100/api/ds/query -u admin:admin \
  -H "Content-Type: application/json" -d '{
  "queries": [{"refId":"A","datasource":{"uid":"sqlite-prod"},
    "rawQueryText":"SELECT * FROM bar_metrics LIMIT 3"}],
  "from":"now-1h","to":"now"}'
```

返回 `status: 200`，但 `fields: []`，`values: []`。**无错误，静默返回空**。

### 4. 手动创建数据源报 "path blocked"

```bash
# 创建指向 /tmp/test_grafana.db 的数据源
curl -s -X POST http://localhost:3100/api/datasources ...
# 查询报错："path contains blocked term from GF_PLUGIN_BLOCK_LIST"
```

**原因**：文件名 `test_grafana.db` 包含子串 `grafana.db`，命中了插件的内部阻止列表。

### 5. 尝试环境变量禁用阻止列表（无效）

```bash
GF_PLUGIN_UNSAFE_DISABLE_GRAFANA_INTERNAL_BLOCKLIST=true \
GF_PLUGIN_UNSAFE_DISABLE_QUERY_ONLY_PATH_OPTIONS=true \
nohup grafana-server ...
```

**结论**：环境变量对此插件无效。v4.0.0 的安全配置必须写在 `grafana.ini`（或 `custom.ini`）中。

### 6. 正确的插件配置方式（custom.ini）

在 `custom.ini` 中添加 `[plugin.frser-sqlite-datasource]` section：

```ini
[plugin.frser-sqlite-datasource]
unsafe_disable_query_only_path_option = true
unsafe_disable_security_blocklist = true
unsafe_disable_grafana_internal_blocklist = true
block_list = ""
```

重启 Grafana 后，阻止列表相关错误消失。**但查询仍然返回空**。

### 7. 发现 API 查询字段名问题（第二个坑，根因）

v4.0.1 插件后端实际读取的字段是 `queryText`，而非 `rawQueryText`：

```bash
# 失败 — 只有 rawQueryText
{"rawQueryText": "SELECT * FROM bar_metrics LIMIT 3"}
# → fields: [], values: []（静默忽略）

# 成功 — 必须包含 queryText
{"queryText": "SELECT * FROM bar_metrics LIMIT 3",
 "rawQueryText": "SELECT * FROM bar_metrics LIMIT 3"}
# → 正确返回数据
```

**原因**：Grafana 前端在渲染 Dashboard 时会自动将 `rawQueryText` 映射为 `queryText` 发送给后端。
但 Dashboard JSON 中只存储了 `rawQueryText`，如果 Grafana 前端的映射没生效（或版本差异），
插件后端就收不到查询，静默返回空结果。

### 8. Dashboard "Data is missing a time field"（第三个坑）

即使查询有数据返回，时序面板仍报错。

**原因**：frser-sqlite-datasource 插件需要 `timeColumns` 字段告诉它哪个列是时间列。
SQLite 没有原生时间类型，插件不会自动推断。

---

## 最终修复

### A. Grafana custom.ini

文件路径：`~/grafana/conf/custom.ini`

```ini
[server]
http_port = 3100

[security]
admin_user = admin
admin_password = admin

[auth.anonymous]
enabled = true
org_role = Viewer

[paths]
provisioning = /home/yusen/ashare_feature/monitoring/grafana/provisioning

[plugins]
allow_loading_unsigned_plugins = frser-sqlite-datasource

[plugin.frser-sqlite-datasource]
unsafe_disable_query_only_path_option = true
unsafe_disable_security_blocklist = true
unsafe_disable_grafana_internal_blocklist = true
block_list = ""

[date_formats]
default_timezone = Asia/Shanghai
```

### B. Dashboard JSON targets

每个 target 必须同时包含 `queryText`、`rawQueryText`，时序面板还需要 `timeColumns`：

```json
{
  "rawQueryText": "SELECT trade_date || 'T' || bar_time || '+08:00' AS time, cost_seconds FROM bar_metrics WHERE trade_date = '${trade_date}' ORDER BY bar_time",
  "queryText": "SELECT trade_date || 'T' || bar_time || '+08:00' AS time, cost_seconds FROM bar_metrics WHERE trade_date = '${trade_date}' ORDER BY bar_time",
  "timeColumns": ["time"],
  "refId": "A"
}
```

**关键要素**：
- `queryText`：插件后端实际读取的 SQL 字段
- `rawQueryText`：Grafana 前端使用的字段（两者内容相同）
- `timeColumns: ["time"]`：告诉插件将 `time` 列解析为时间戳
- SQL 中 `trade_date || 'T' || bar_time || '+08:00'` 生成 RFC3339 格式（如 `2026-03-03T09:30:00+08:00`）

### C. 数据源 provisioning

文件路径：`monitoring/grafana/provisioning/datasources/sqlite.yml`

```yaml
apiVersion: 1
datasources:
  - name: FeatureMonitor
    type: frser-sqlite-datasource
    uid: sqlite-prod
    access: proxy
    jsonData:
      path: /home/yusen/ashare_feature/monitor.db
    editable: false
```

**注意**：`path` 放在 `jsonData` 中（不是 `secureJsonData`）。

---

## 坑的总结

| # | 问题 | 表现 | 根因 | 解决 |
|---|------|------|------|------|
| 1 | v4.0.0 安全阻止列表 | API 创建的数据源查询报 "path blocked" | 插件内置阻止列表默认启用，block `grafana.db`、`.env`、`/proc/` 等 | `custom.ini` 添加 `[plugin.frser-sqlite-datasource]` 配置 |
| 2 | 查询字段名 | 查询返回 200 但 fields 为空，无报错 | 插件后端读 `queryText`，Dashboard JSON 只存了 `rawQueryText` | target 中同时写 `queryText` 和 `rawQueryText` |
| 3 | 缺少 timeColumns | 面板显示 "Data is missing a time field" | SQLite 无原生时间类型，插件不自动推断 | target 中加 `timeColumns: ["time"]` |
| 4 | 环境变量无效 | 设了 `GF_PLUGIN_UNSAFE_*` 环境变量但没效果 | 此插件只读 `grafana.ini` 的 `[plugin.frser-sqlite-datasource]` section | 配置写 `custom.ini` 而非环境变量 |
| 5 | 健康检查误导 | health 返回 OK 但查询为空 | health check 只验证文件可打开，不验证查询能力 | 不要依赖 health check 判断查询是否正常 |

---

## 参考

- [frser-sqlite-datasource GitHub](https://github.com/fr-ser/grafana-sqlite-datasource)
- [v4.0.0 CHANGELOG](https://github.com/fr-ser/grafana-sqlite-datasource/blob/main/CHANGELOG.md) — 安全相关 breaking changes
- [插件 README - Configuration 节](https://github.com/fr-ser/grafana-sqlite-datasource#configuration) — `grafana.ini` 配置格式
- [插件 README - Time Formatted Columns](https://github.com/fr-ser/grafana-sqlite-datasource#support-for-time-formatted-columns) — 时间列处理
