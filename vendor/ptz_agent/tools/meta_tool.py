"""tools/meta_tool.py — Orchestration tools available only to the master agent.

These tools talk to the supervisor over the local IPC socket. They are
gated behind ``MSA_AGENT_ROLE=master`` so worker processes can't spawn
more workers (Q6 from the design discussion).

Tools:
    run_task_now       — fire a worker for a free-form prompt
    schedule_task      — register a recurring task (interval or cron)
    list_workers       — list workers (any state)
    worker_status      — full record for one worker
    wait_for_worker    — block until a worker finishes; return its response
    cancel_worker      — SIGTERM a running worker
    worker_log         — last N transcript lines for a worker
    list_tasks         — list scheduled tasks
    delete_task        — remove a scheduled task

All worker-targeted tools accept the worker id under any of these
argument names: ``worker_id``, ``id``, ``worker``. Gemma-class models
keep mixing them, and the supervisor RPC only ever needs one string,
so we just normalise on entry rather than penalising the model.
"""

from __future__ import annotations

import json
import logging
import os
import time

from msa.ipc import Client, IPCError
from msa.store import EVT_FAILED, EVT_FINISHED, EVT_RESPONSE
from msa.tools import BaseTool

logger = logging.getLogger(__name__)


def _is_master() -> bool:
    return os.environ.get("MSA_AGENT_ROLE", "worker") == "master"


def _client() -> Client:
    return Client()


# ---------------------------------------------------------------------------
# Master-only tools
#
# Each plugin checks ``_is_master()`` at import time. If we're inside a
# worker, the tool's ``run`` short-circuits with an explanatory error.
# We could omit the registration entirely, but having the tool present
# with a clear error message is a better debugging experience when a
# task description accidentally references a master tool.
# ---------------------------------------------------------------------------

_NOT_MASTER = (
    "ERROR: this tool is only available to the master chat agent; "
    "workers cannot spawn or schedule other workers."
)


def _supervisor_down(exc: Exception) -> str:
    return (
        f"ERROR: supervisor unavailable ({exc}). Run "
        "`python -m msa.supervisor --daemonize` and retry."
    )


def _normalise_worker_id(worker_id: str = "", **kwargs) -> str:
    """Accept worker_id / id / worker interchangeably, return the first set.

    Returns "" if none are populated, so callers can do an empty-string
    check and emit a single uniform error message regardless of which
    field name the model tried to use.
    """
    for val in (worker_id, kwargs.get("id"), kwargs.get("worker")):
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _format_worker(w: dict) -> str:
    parts = [
        f"id={w['id']}",
        f"state={w['state']}",
        f"prompt={(w.get('prompt') or '')[:60]!r}",
        f"iter={w.get('iterations')}/{w.get('max_iterations')}",
        f"tokens={w.get('total_tokens')}",
    ]
    rt = w.get("runtime_seconds")
    if rt is not None:
        parts.append(f"runtime={rt:.1f}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class RunTaskNowTool(BaseTool):
    name = "run_task_now"
    description = (
        "Spawn a background worker NOW with a natural-language prompt. "
        "Returns the worker id. Args: prompt (str, required), "
        "name (str, optional)."
    )

    def run(self, prompt: str = "", name: str | None = None, **_) -> str:
        if not _is_master():
            return _NOT_MASTER
        if not prompt:
            return "ERROR: prompt is required"
        try:
            w = _client().run_task_now(prompt=prompt, name=name)
        except IPCError as exc:
            return _supervisor_down(exc)
        return json.dumps({"worker_id": w["id"], "state": w["state"]})


class ScheduleTaskTool(BaseTool):
    name = "schedule_task"
    description = (
        "Register a recurring task. Use either `cron` (e.g. '0 * * * *') "
        "OR `interval_seconds` (e.g. 3600) — not both. Args: name (str), "
        "prompt (str), cron (str, optional), interval_seconds (int, "
        "optional), enabled (bool, default true)."
    )

    def run(
        self,
        name: str = "",
        prompt: str = "",
        cron: str | None = None,
        interval_seconds: int | None = None,
        enabled: bool = True,
        **_,
    ) -> str:
        if not _is_master():
            return _NOT_MASTER
        if not name or not prompt:
            return "ERROR: name and prompt are required"
        if (cron is None) == (interval_seconds is None):
            return ("ERROR: exactly one of `cron` or `interval_seconds` "
                     "is required")
        try:
            t = _client().schedule_task(
                name=name, prompt=prompt,
                cron=cron, interval_seconds=interval_seconds,
                enabled=enabled,
            )
        except IPCError as exc:
            return _supervisor_down(exc)
        return json.dumps({
            "task": t["name"],
            "next_run": t.get("next_run"),
            "enabled": t.get("enabled"),
        })


class ListWorkersTool(BaseTool):
    name = "list_workers"
    description = (
        "List recent workers. Args: states (list of "
        "pending|running|completed|failed|cancelled, optional), "
        "limit (int, default 20)."
    )

    def run(self, states: list | None = None, limit: int = 20, **_) -> str:
        if not _is_master():
            return _NOT_MASTER
        try:
            ws = _client().list_workers(states=states, limit=int(limit))
        except IPCError as exc:
            return _supervisor_down(exc)
        if not ws:
            return "(no workers)"
        return "\n".join(_format_worker(w) for w in ws)


class WorkerStatusTool(BaseTool):
    name = "worker_status"
    description = (
        "Full record for one worker — state, iteration count, tokens, "
        "runtime, and (once it finishes) the worker's final `respond` "
        "message in the `result` field. Args: worker_id (str — also "
        "accepts `id` or `worker`)."
    )

    def run(self, worker_id: str = "", **kwargs) -> str:
        if not _is_master():
            return _NOT_MASTER
        wid = _normalise_worker_id(worker_id, **kwargs)
        if not wid:
            return ("ERROR: worker_id required (e.g. "
                    '{"worker_id": "w-abc123"})')
        try:
            w = _client().get_worker(wid)
        except IPCError as exc:
            return _supervisor_down(exc)
        if not w:
            return f"ERROR: worker {wid!r} not found"
        return json.dumps(w, indent=2, default=str)


class WaitForWorkerTool(BaseTool):
    name = "wait_for_worker"
    description = (
        "Block until a worker reaches a terminal state and return its "
        "`respond` message. Use this immediately after `run_task_now` "
        "to retrieve the spawned worker's answer instead of polling "
        "`worker_status`. Args: worker_id (str — also accepts `id` or "
        "`worker`), timeout_seconds (float, default 300). Returns JSON: "
        "{state, response, iterations, total_tokens, runtime_seconds}."
    )

    def run(self, worker_id: str = "",
            timeout_seconds: float = 300.0, **kwargs) -> str:
        if not _is_master():
            return _NOT_MASTER
        wid = _normalise_worker_id(worker_id, **kwargs)
        if not wid:
            return ("ERROR: worker_id required (e.g. "
                    '{"worker_id": "w-abc123"})')

        deadline = time.time() + float(timeout_seconds)
        # Stream events for this worker. The supervisor yields events
        # historically (after_id=0 → from the start) plus live ones, so
        # if the worker already finished we still see EVT_FINISHED.
        client = _client()
        response_msg: str | None = None
        try:
            for ev in client.stream_events(worker_id=wid):
                if time.time() > deadline:
                    return json.dumps({
                        "state": "timeout",
                        "error": f"timed out after {timeout_seconds}s",
                    })
                kind = ev.get("kind")
                payload = ev.get("payload") or {}
                if kind == EVT_RESPONSE:
                    response_msg = payload.get("message") or response_msg
                elif kind in (EVT_FINISHED, EVT_FAILED):
                    break
        except IPCError as exc:
            return _supervisor_down(exc)

        # Pull the final record so we have authoritative state + stats.
        try:
            w = client.get_worker(wid)
        except IPCError as exc:
            return _supervisor_down(exc)
        if not w:
            return f"ERROR: worker {wid!r} not found"

        return json.dumps({
            "state": w.get("state"),
            "response": response_msg or w.get("result") or "",
            "iterations": w.get("iterations"),
            "total_tokens": w.get("total_tokens"),
            "runtime_seconds": w.get("runtime_seconds"),
            "error": w.get("error"),
        }, indent=2, default=str)


class CancelWorkerTool(BaseTool):
    name = "cancel_worker"
    description = (
        "SIGTERM a running worker. Args: worker_id (str — also accepts "
        "`id` or `worker`)."
    )

    def run(self, worker_id: str = "", **kwargs) -> str:
        if not _is_master():
            return _NOT_MASTER
        wid = _normalise_worker_id(worker_id, **kwargs)
        if not wid:
            return ("ERROR: worker_id required (e.g. "
                    '{"worker_id": "w-abc123"})')
        try:
            r = _client().cancel_worker(wid)
        except IPCError as exc:
            return _supervisor_down(exc)
        return json.dumps(r)


class WorkerLogTool(BaseTool):
    name = "worker_log"
    description = (
        "Return the last N transcript entries for a worker. Args: "
        "worker_id (str — also accepts `id` or `worker`), tail (int, "
        "default 20)."
    )

    def run(self, worker_id: str = "", tail: int = 20, **kwargs) -> str:
        if not _is_master():
            return _NOT_MASTER
        wid = _normalise_worker_id(worker_id, **kwargs)
        if not wid:
            return ("ERROR: worker_id required (e.g. "
                    '{"worker_id": "w-abc123"})')
        try:
            entries = _client().get_transcript(wid)
        except IPCError as exc:
            return _supervisor_down(exc)
        if not entries:
            return "(empty transcript)"
        tailed = entries[-int(tail):]
        return json.dumps(tailed, indent=2, default=str)


class ListTasksTool(BaseTool):
    name = "list_tasks"
    description = "List scheduled tasks."

    def run(self, **_) -> str:
        if not _is_master():
            return _NOT_MASTER
        try:
            tasks = _client().list_tasks()
        except IPCError as exc:
            return _supervisor_down(exc)
        if not tasks:
            return "(no scheduled tasks)"
        lines = []
        for t in tasks:
            sched = (
                f"every {t['interval_seconds']}s"
                if t.get("interval_seconds")
                else f"cron {t.get('cron')}"
            )
            lines.append(
                f"{t['name']}  [{sched}]  enabled={t.get('enabled')}  "
                f"runs={t.get('run_count')}  next_run={t.get('next_run')}"
            )
        return "\n".join(lines)


class DeleteTaskTool(BaseTool):
    name = "delete_task"
    description = "Delete a scheduled task. Args: name (str)."

    def run(self, name: str = "", **_) -> str:
        if not _is_master():
            return _NOT_MASTER
        if not name:
            return "ERROR: name required"
        try:
            r = _client().delete_task(name)
        except IPCError as exc:
            return _supervisor_down(exc)
        return json.dumps(r)
