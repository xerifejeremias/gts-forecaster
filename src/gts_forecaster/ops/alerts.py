"""Alerts when an automatic job fails or its forecasts are flagged.

Always appended to runs/alerts.log. Optionally sent by e-mail (SMTP with STARTTLS; the password comes from
an environment variable) and/or to a Teams / Slack incoming webhook (URL from an environment variable).
Sending never raises: a failed e-mail or webhook is logged, the job's own result is what counts.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from gts_forecaster.ops.schedule import AlertSettings

logger = logging.getLogger(__name__)


def send_alert(subject: str, body: str, settings: AlertSettings, log_path: Path) -> list[str]:
    """Record and send one alert. Returns the channels it reached (log, email, webhook)."""
    reached = []
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {subject}\n    " + body.replace("\n", "\n    ") + "\n")
    reached.append("log")
    if settings.email_to and settings.smtp_host:
        try:
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = settings.email_from or settings.smtp_user or "gts-forecaster"
            message["To"] = ", ".join(settings.email_to)
            message.set_content(body)
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
                smtp.starttls()
                password = os.environ.get(settings.smtp_password_env)
                if settings.smtp_user and password:
                    smtp.login(settings.smtp_user, password)
                smtp.send_message(message)
            reached.append("email")
        except Exception as exc:  # noqa: BLE001 - an alert must never break the job
            logger.warning("alert e-mail could not be sent: %s", exc)
    if settings.webhook_env:
        url = os.environ.get(settings.webhook_env)
        if not url:
            logger.warning("alert webhook: environment variable %s is not set", settings.webhook_env)
        else:
            try:
                data = json.dumps({"text": f"{subject}\n{body}"}).encode("utf-8")
                request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=30):
                    pass
                reached.append("webhook")
            except Exception as exc:  # noqa: BLE001
                logger.warning("alert webhook could not be reached: %s", exc)
    return reached
