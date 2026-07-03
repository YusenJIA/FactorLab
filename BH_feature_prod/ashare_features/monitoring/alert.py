"""
邮件告警 — 仅使用 stdlib (smtplib + email)
"""
import smtplib
import logging
from email.mime.text import MIMEText
from email.utils import formatdate

from . import config

logger = logging.getLogger("monitoring.alert")


def send_abandon_alert(trade_date: str, bar_time: str, retry_count: int, reason: str) -> bool:
    """发送 abandon 事件邮件告警，返回是否发送成功"""
    if not config.EMAIL_ENABLED or not config.SMTP_USER or not config.EMAIL_RECEIVERS:
        logger.debug("邮件告警未配置，跳过发送")
        return False

    subject = f"[特征监控告警] {trade_date} bar {bar_time} 放弃拉取"
    body = (
        f"交易日: {trade_date}\n"
        f"Bar 时间: {bar_time}\n"
        f"重试次数: {retry_count}\n"
        f"原因: {reason}\n"
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = config.SMTP_USER
    msg["To"] = ", ".join(config.EMAIL_RECEIVERS)
    msg["Date"] = formatdate(localtime=True)

    try:
        with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as smtp:
            smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.sendmail(config.SMTP_USER, config.EMAIL_RECEIVERS, msg.as_string())
        logger.info(f"告警邮件已发送: {subject}")
        return True
    except Exception as e:
        logger.error(f"告警邮件发送失败: {e}")
        return False
