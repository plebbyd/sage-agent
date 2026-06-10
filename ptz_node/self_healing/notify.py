"""Notifications for the self-healing skill — Slack and email, stdlib only.

Channels (any combination, configured under ``self_healing.notify``):
  * slack webhook  — POST to ``SLACK_WEBHOOK_URL`` (simplest, no deps).
  * slack bot      — chat.postMessage with ``SLACK_BOT_TOKEN`` + channel.
  * email          — SMTP via ``smtplib`` (TLS), creds from env.

Secrets always come from environment variables, never from config files.
``send()`` is best-effort and never raises — a broken notifier must not mask
the original incident.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.cfg = ((config or {}).get("self_healing") or {}).get("notify") or {}
        self.channels = [c.lower() for c in (self.cfg.get("channels") or [])]

    def enabled(self) -> bool:
        return bool(self.channels)

    def send(self, subject: str, body: str,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for ch in self.channels:
            try:
                if ch == "slack":
                    results["slack"] = self._slack(subject, body)
                elif ch == "email":
                    results["email"] = self._email(subject, body)
                else:
                    results[ch] = {"ok": False, "error": f"unknown channel {ch!r}"}
            except Exception as exc:  # never raise from a notifier
                results[ch] = {"ok": False, "error": str(exc)}
                logger.warning("notify %s failed: %s", ch, exc)
        if extra is not None:
            extra["notify"] = results
        return results

    # -- slack -------------------------------------------------------------

    def _slack(self, subject: str, body: str) -> dict[str, Any]:
        text = f"*{subject}*\n{body}"
        webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
        if webhook:
            return self._post_json(webhook, {"text": text})

        token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
        channel = (
            os.environ.get("SLACK_CHANNEL", "").strip()
            or str(self.cfg.get("slack_channel") or "").strip()
        )
        if token and channel:
            return self._post_json(
                "https://slack.com/api/chat.postMessage",
                {"channel": channel, "text": text},
                headers={"Authorization": f"Bearer {token}"},
            )
        return {"ok": False,
                "error": "set SLACK_WEBHOOK_URL, or SLACK_BOT_TOKEN + SLACK_CHANNEL"}

    def _post_json(self, url: str, payload: dict[str, Any],
                   headers: dict[str, str] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            text = resp.read().decode("utf-8", "replace")
            ok = resp.status < 300 and ('"ok":false' not in text.replace(" ", ""))
            return {"ok": ok, "status": resp.status, "body": text[:300]}

    # -- email -------------------------------------------------------------

    def _email(self, subject: str, body: str) -> dict[str, Any]:
        host = os.environ.get("SMTP_HOST", "").strip()
        if not host:
            return {"ok": False, "error": "set SMTP_HOST (and SMTP_USER/SMTP_PASSWORD)"}
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER", "").strip()
        password = os.environ.get("SMTP_PASSWORD", "").strip()
        sender = (
            os.environ.get("SMTP_FROM", "").strip()
            or user
            or str(self.cfg.get("email_from") or "").strip()
        )
        recipients = [
            r.strip() for r in (
                os.environ.get("SMTP_TO", "").strip()
                or ",".join(self.cfg.get("email_to") or [])
            ).split(",") if r.strip()
        ]
        if not sender or not recipients:
            return {"ok": False, "error": "set SMTP_FROM and SMTP_TO (or notify.email_to)"}

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)

        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15.0) as server:
            server.ehlo()
            try:
                server.starttls(context=ctx)
                server.ehlo()
            except smtplib.SMTPException:
                pass  # server may not support STARTTLS (e.g. localhost relay)
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return {"ok": True, "recipients": recipients}
