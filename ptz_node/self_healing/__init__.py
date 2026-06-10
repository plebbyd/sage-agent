"""Self-diagnosis / self-healing subsystem.

Components:
  * :mod:`snapshot` — periodic, size-bounded system/OS state snapshots.
  * :mod:`notify`   — Slack (webhook/bot) + SMTP email notifications.
  * :mod:`healer`   — strong-LLM diagnosis, patch proposal, auto/approval apply.
  * :mod:`guard`    — error capture (context manager / decorator / excepthook).
"""
