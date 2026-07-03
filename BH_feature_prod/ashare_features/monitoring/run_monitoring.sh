#!/bin/bash
# run_monitoring.sh — 启动旁路监控脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FEATURE_HOME="${ASHARE_FEATURE_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${ASHARE_LOG_DIR:-${FEATURE_HOME}/logs}"
TRADE_DATE="${TRADE_DATE:-$(date +%Y-%m-%d)}"
LOG_FILE="${LOG_DIR}/${TRADE_DATE//\-/}_monitoring.log"

mkdir -p "${LOG_DIR}"

export TRADE_DATE
export ASHARE_FEATURE_HOME="${FEATURE_HOME}"

cd "${FEATURE_HOME}"
nohup python -m monitoring.monitoring >> "${LOG_FILE}" 2>&1 &

echo "监控已启动，PID: $!"
echo "交易日: ${TRADE_DATE}"
echo "日志文件: ${LOG_FILE}"
