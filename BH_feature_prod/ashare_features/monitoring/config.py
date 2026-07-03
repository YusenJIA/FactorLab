"""
监控系统配置 — 纯 stdlib，零外部依赖
"""
import os
from datetime import datetime

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_HOME = os.environ.get("ASHARE_FEATURE_HOME", _PARENT_DIR)

# SQLite 数据库路径
DB_PATH = os.environ.get(
    "MONITOR_DB_PATH",
    "/data/BH/monitor.db",
)

# ---- monitoring.py 旁路观察者配置 ----
TRADE_DATE = os.environ.get("TRADE_DATE", "") or datetime.now().strftime("%Y-%m-%d")
REALTIME_OUTPUT_DIR = os.environ.get("ASHARE_OUTPUT_DIR", "/data/BH")
LOG_DIR = os.environ.get("ASHARE_LOG_DIR", os.path.join(FEATURE_HOME, "logs"))
OFFLINE_FEATURES_DIR = os.environ.get(
    "ASHARE_OFFLINE_FEATURES_DIR",
    os.path.join(FEATURE_HOME, "offline_features"),
)
POLL_INTERVAL = 60            # 轮询间隔（秒）
POST_CLOSE_WAIT_MINUTES = 30  # 收盘后等待离线文件的最大时间（分钟）
KNOWN_DIVERGENT_FEATURES = ["volume_fomo_241min", "panic_sell_241min"]

# 邮件告警配置（按需填写，留空则不发送）
EMAIL_ENABLED = False
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SMTP_USER = ""        # 发件邮箱
SMTP_PASSWORD = ""    # 授权码
EMAIL_RECEIVERS = []  # 收件人列表

# 需要跟踪统计量的关键特征（每个 Processor 选几个代表 + 已知问题特征）
TRACKED_FEATURES = [
    # 已知发散 / inf 问题
    "volume_fomo_241min",
    "panic_sell_241min",
    "social_attention_5min",
    "social_attention_60min",
    "social_attention_241min",
    # 各 Processor 代表性特征
    "round_1_distance",           # RoundNumber
    "fomo_surge_5min",            # FOMOFUD
    "buy_high_pattern_5min",      # RetailPattern
    "price_clustering_5min",      # Herding
    "kyle_lambda",                # Microstructure
    "sentiment_overheat_5min",    # SentimentCycle
    "trading_activity_5min",      # Attention
]
