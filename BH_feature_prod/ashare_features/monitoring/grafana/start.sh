#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
GRAFANA_VERSION="11.4.0"
GRAFANA_HOME="$HOME/grafana"
GRAFANA_PORT=3100
PROVISIONING_DIR="$(cd "$(dirname "$0")/provisioning" && pwd)"

DEB_FILE="grafana_${GRAFANA_VERSION}_amd64.deb"
# Use Tsinghua mirror (China) for fast download
DOWNLOAD_URL="https://mirrors.tuna.tsinghua.edu.cn/grafana/apt/pool/main/g/grafana/${DEB_FILE}"

# ── 1. Download & extract if not present ─────────────────────────────
if [ ! -x "$GRAFANA_HOME/bin/grafana-server" ]; then
    echo ">>> Downloading Grafana OSS ${GRAFANA_VERSION} from Tsinghua mirror ..."
    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' EXIT

    wget -q --show-progress -O "$TMP_DIR/$DEB_FILE" "$DOWNLOAD_URL"

    echo ">>> Extracting to $GRAFANA_HOME ..."
    mkdir -p "$GRAFANA_HOME"
    # Extract deb without sudo: dpkg-deb -x extracts data payload
    dpkg-deb -x "$TMP_DIR/$DEB_FILE" "$TMP_DIR/extracted"
    # Grafana deb installs to usr/share/grafana/
    cp -a "$TMP_DIR/extracted/usr/share/grafana/." "$GRAFANA_HOME/"

    rm -rf "$TMP_DIR"
    trap - EXIT
    echo ">>> Grafana installed at $GRAFANA_HOME"
fi

# ── 2. Install SQLite plugin ─────────────────────────────────────────
PLUGIN_DIR="$GRAFANA_HOME/data/plugins"
SQLITE_PLUGIN_VER="4.0.1"
if [ ! -d "$PLUGIN_DIR/frser-sqlite-datasource" ]; then
    echo ">>> Downloading SQLite datasource plugin v${SQLITE_PLUGIN_VER} ..."
    PLUGIN_ZIP="frser-sqlite-datasource-${SQLITE_PLUGIN_VER}.zip"
    PLUGIN_URL="https://gh-proxy.com/https://github.com/fr-ser/grafana-sqlite-datasource/releases/download/v${SQLITE_PLUGIN_VER}/${PLUGIN_ZIP}"
    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' EXIT

    wget -q --show-progress -O "$TMP_DIR/$PLUGIN_ZIP" "$PLUGIN_URL"

    echo ">>> Extracting plugin ..."
    mkdir -p "$PLUGIN_DIR"
    unzip -q "$TMP_DIR/$PLUGIN_ZIP" -d "$PLUGIN_DIR/"

    rm -rf "$TMP_DIR"
    trap - EXIT
    echo ">>> Plugin installed"
fi

# ── 3. Generate custom.ini ───────────────────────────────────────────
CUSTOM_INI="$GRAFANA_HOME/conf/custom.ini"
echo ">>> Writing $CUSTOM_INI ..."
cat > "$CUSTOM_INI" <<EOF
[server]
http_port = ${GRAFANA_PORT}

[security]
admin_user = admin
admin_password = admin

[auth.anonymous]
enabled = true
org_role = Viewer

[dashboards]
default_home_dashboard_path =

[paths]
provisioning = ${PROVISIONING_DIR}

[plugins]
allow_loading_unsigned_plugins = frser-sqlite-datasource

[date_formats]
default_timezone = Asia/Shanghai
EOF
echo ">>> Config written"

# ── 4. Stop previous instance if running ─────────────────────────────
if pgrep -f "grafana-server.*--homepath $GRAFANA_HOME" > /dev/null 2>&1; then
    echo ">>> Stopping existing Grafana instance ..."
    pkill -f "grafana-server.*--homepath $GRAFANA_HOME" || true
    sleep 1
fi

# ── 5. Start Grafana ─────────────────────────────────────────────────
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/grafana.log"

echo ">>> Starting Grafana on port ${GRAFANA_PORT} ..."
nohup "$GRAFANA_HOME/bin/grafana-server" \
    --homepath "$GRAFANA_HOME" \
    --config "$CUSTOM_INI" \
    > "$LOG_FILE" 2>&1 &

GRAFANA_PID=$!
echo ">>> Grafana PID: $GRAFANA_PID"
echo ">>> Log: $LOG_FILE"

# Wait a moment and verify it's running
sleep 2
if kill -0 "$GRAFANA_PID" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "  Grafana is running!"
    echo "  URL:   http://localhost:${GRAFANA_PORT}"
    echo "  Login: admin / admin"
    echo "=========================================="
else
    echo "ERROR: Grafana failed to start. Check $LOG_FILE"
    tail -20 "$LOG_FILE"
    exit 1
fi
