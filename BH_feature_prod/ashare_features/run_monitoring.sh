#!/bin/bash
# run_monitoring.sh — 启动每日监控 sidecar
# 用法: ./run_monitoring.sh [YYYY-MM-DD]
#   不传参数则使用今天日期

set -e

TRADE_DATE="${1:-$(date +%Y-%m-%d)}"
YYYYMMDD=$(echo "$TRADE_DATE" | tr -d '-')
BASE_DIR="${ASHARE_FEATURE_HOME:-/home/yusen/ashare_feature}"
LOG_DIR="${ASHARE_LOG_DIR:-${BASE_DIR}/logs_test}"
MONITOR_LOG="${LOG_DIR}/${YYYYMMDD}_bh_monitoring.log"
PIPELINE_LOG="${LOG_DIR}/${TRADE_DATE}_realtime_feature.log"
OUTPUT_DIR="${ASHARE_OUTPUT_DIR:-/data/BH}"
PARQUET_DIR="${OUTPUT_DIR}/${TRADE_DATE}"
DB_PATH="${MONITOR_DB_PATH:-${OUTPUT_DIR}/monitor.db}"

mkdir -p "$LOG_DIR"

echo "=== 监控启动检查 ==="
echo "交易日: ${TRADE_DATE}"
echo ""

# 1. 检查 pipeline 是否在运行
if ! pgrep -f "realtime_feature_cmp" > /dev/null; then
    echo "[警告] 实时 pipeline 未运行，监控无数据可采集"
    read -p "是否继续启动？(y/N) " confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && exit 1
fi

# 2. 检查是否已有 monitor 进程
EXISTING_PID=$(pgrep -f "python -m monitoring.monitoring" || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "[跳过] 监控已在运行 (PID: ${EXISTING_PID})"
    echo "日志: ${MONITOR_LOG}"
    exit 0
fi

# 3. 检查 pipeline 日志是否存在
if [[ ! -f "$PIPELINE_LOG" ]]; then
    echo "[警告] Pipeline 日志不存在: ${PIPELINE_LOG}"
    read -p "是否继续启动？(y/N) " confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && exit 1
fi

# 4. 启动监控
cd "$BASE_DIR"
TRADE_DATE="$TRADE_DATE" MONITOR_DB_PATH="$DB_PATH" ASHARE_FEATURE_HOME="$BASE_DIR" \
    ASHARE_LOG_DIR="$LOG_DIR" ASHARE_OUTPUT_DIR="$OUTPUT_DIR" \
    nohup python -m monitoring.monitoring >> "$MONITOR_LOG" 2>&1 &
MON_PID=$!

echo ""
echo "监控已启动 (PID: ${MON_PID})"
echo ""

# 5. 等待并验证
sleep 2
if ps -p "$MON_PID" > /dev/null 2>&1; then
    echo "=== 启动成功 ==="
else
    echo "[错误] 进程已退出，查看日志:"
    tail -10 "$MONITOR_LOG"
    exit 1
fi

echo ""
echo "  Monitor PID : ${MON_PID}"
echo "  Trade date  : ${TRADE_DATE}"
echo "  DB path     : ${DB_PATH}"
echo "  Monitor log : ${MONITOR_LOG}"
echo "  Parquet dir : ${PARQUET_DIR}"
echo "  Pipeline log: ${PIPELINE_LOG}"
echo ""
echo "最近日志:"
tail -5 "$MONITOR_LOG"
