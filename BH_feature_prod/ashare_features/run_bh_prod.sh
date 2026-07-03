#!/bin/bash
# run_test_api.sh

MODE=${1:-instant}
FEATURE_HOME="${ASHARE_FEATURE_HOME:-/home/yusen/ashare_feature}"
PYTHON_BIN="${ASHARE_PYTHON_BIN:-/home/yusen/miniconda3/envs/yusen/bin/python}"
LOG_DIR="${ASHARE_LOG_DIR:-${FEATURE_HOME}/logs_test}"
TRADE_DATE=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/${TRADE_DATE}_realtime_feature.log"

mkdir -p ${LOG_DIR}

TRADE_DATE="${TRADE_DATE}" ASHARE_FEATURE_HOME="${FEATURE_HOME}" \
    nohup "${PYTHON_BIN}" "${FEATURE_HOME}/realtime_feature_cmp.py" --mode ${MODE} >> ${LOG_FILE} 2>&1 &

echo "test 脚本已启动 (mode=${MODE}), PID: $!"
echo "日志文件: ${LOG_FILE}"
