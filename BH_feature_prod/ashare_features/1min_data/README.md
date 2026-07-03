# 1min_data 数据说明

## 数据内容

A股全市场（~5200只）1分钟OHLCV行情数据。

- **offline/** — 离线pipeline计算的分钟行情
- **online/** — 实盘pipeline（/data/BH）提取的分钟行情

## 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| time | datetime[μs] | 分钟bar时间戳 |
| symbol | str | 6位股票代码（如"000001"） |
| open | float64 | 开盘价 |
| close | float64 | 收盘价 |
| high | float64 | 最高价 |
| low | float64 | 最低价 |
| volume | float64 | 成交量 |
| money | float64 | 成交额 |

## 时间范围

- 每天包含 09:30 - 15:00 共 **241根** 分钟K线
- 09:30 为集合竞价撮合价格，0竞价的股票使用昨日收盘价填充
- 午间休市跳过（11:30 → 13:01）
