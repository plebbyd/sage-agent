"""msa/supervisor.py — Long-running orchestrator.

The supervisor owns:
    * the IPC server (unix socket; chat + web UI clients)
    * the scheduler (fires due tasks → enqueues workers)
    * the worker pool (cap = config.supervisor.concurrency)
    * the worker spawner (subprocess + reaper + cancel via SIGTERM)

It does NOT run any LLM inference itself. Workers do that, each in
its own subprocess. The supervisor only orchestrates.

Run modes
---------

    python -m msa.supervisor                # runs in the foreground
    python -m msa.supervisor --daemonize    # double-fork; PID file in $HOME/.msa
    python -m msa.supervisor --no-scheduler # IPC only (useful for tests)

Auto-start: the chat CLI calls ``ensure_running()``. If no socket is
listening, it spawns ``python -m msa.supervisor --daemonize`` and waits
up to ~5 s for the socket to appear.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from . import ipc
from .config import load_config
from .store import (
    ACTIVE_STATES,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_PENDING,
    STATE_RUNNING,
    Store,
    Worker,
)

logger = logging.getLogger(__name__)


def _merge_env_file(env: dict, path: Path | None = None) -> None:
    """Parse a shell env file (``KEY=VAL`` or ``export KEY="VAL"`` lines)
    and merge it into ``env`` in-place. Values written to disk by tools
    like ``ptz_find_camera`` overwrite whatever was previously in env.

    Lines whose value contains unresolved shell expansion (``$VAR`` or
    ``${...}``) are SKIPPED -- we have no shell to expand them, and the
    parent env (already populated by bash when the user sourced the
    file at shell startup) already has the correct expanded value.
    Without this skip we'd clobber e.g. ``OLLAMA_KEEP_ALIVE=10m`` with
    the literal string ``${OLLAMA_KEEP_ALIVE:-10m}``.

    Silent on missing file or unreadable lines -- a fresh checkout has no
    env file and that's fine.
    """
    if path is None:
        path = Path(os.environ.get(
            "MSA_ENV_FILE", str(Path.home() / ".msa.env")
        ))
    try:
        text = path.read_text()
    except (OSError, FileNotFoundError):
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if not key:
            continue
        # Skip values with shell expansion we can't resolve here. The
        # parent env -- which bash has already populated by sourcing
        # the file at shell-startup time -- already has the correct
        # expanded value, so leaving env[key] alone is the right call.
        if "$" in val:
            continue
        env[key] = val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def default_pid_path() -> Path:
    return Path.home() / ".msa" / "supervisord.pid"


def is_running(socket_path: Path | None = None) -> bool:
    return ipc.is_supervisor_running(socket_path)


def ensure_running(*, timeout: float = 8.0) -> bool:
    """If no supervisor is listening, fork+exec one in the background."""
    if is_running():
        return True
    logger.info("starting supervisor in background…")
    py = sys.executable or "python3"
    log_path = Path.home() / ".msa" / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = (log_path / "supervisor.log").open("ab", buffering=0)
    subprocess.Popen(
        [py, "-m", "msa.supervisor", "--daemonize"],
        stdout=log_file, stderr=log_file,
        close_fds=True, start_new_session=True,
    )
    return ipc.wait_for_supervisor(timeout=timeout)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

@dataclass
class _RunningProc:
    worker_id: str
    proc: subprocess.Popen
    started_at: float


class Supervisor:
    def __init__(self, config_path: str = "config/config.yaml",
                  *, with_scheduler: bool = True):
        self.cfg = load_config(config_path)
        self.config_path = config_path
        self.store = Store()
        self.with_scheduler = with_scheduler

        sup_cfg = self.cfg.get("supervisor", {}) or {}
        self.concurrency = int(sup_cfg.get("concurrency", 2))
        self.poll_seconds = float(sup_cfg.get("poll_seconds", 1.5))

        self.running: dict[str, _RunningProc] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

        # IPC server with all methods registered.
        self.server = ipc.Server()
        self._register_methods(self.server)

    # ---- lifecycle -------------------------------------------------------

    def serve_forever(self) -> None:
        self.server.start()
        # Signal handlers only install from the main thread — skip the
        # registration if we were started inside a worker thread (e.g.
        # by a test harness). The shutdown event still works.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._shutdown_handler)
            signal.signal(signal.SIGINT, self._shutdown_handler)

        # On startup, recover any "running" rows whose PIDs are gone
        # (supervisor crashed while a worker was alive). Mark them failed.
        self._recover_orphans()

        threads: list[threading.Thread] = []
        threads.append(threading.Thread(target=self._spawner_loop,
                                          name="msa-spawner", daemon=True))
        threads.append(threading.Thread(target=self._reaper_loop,
                                          name="msa-reaper", daemon=True))
        if self.with_scheduler:
            threads.append(threading.Thread(target=self._scheduler_loop,
                                              name="msa-scheduler",
                                              daemon=True))
        for t in threads:
            t.start()

        logger.info("supervisor up; concurrency=%d, scheduler=%s",
                     self.concurrency, self.with_scheduler)
        try:
            self._shutdown.wait()
        finally:
            self._cleanup()

    def _shutdown_handler(self, signum, _frame):
        logger.info("supervisor received signal %s; shutting down", signum)
        self._shutdown.set()

    def _cleanup(self) -> None:
        # Send SIGTERM to live workers; they handle it gracefully.
        with self._lock:
            for rp in list(self.running.values()):
                try:
                    rp.proc.terminate()
                except OSError:
                    pass
        # Give them a couple seconds.
        for _ in range(20):
            with self._lock:
                if not self.running:
                    break
            time.sleep(0.1)
        # Force-kill stragglers.
        with self._lock:
            for rp in list(self.running.values()):
                try:
                    rp.proc.kill()
                except OSError:
                    pass
        self.server.stop()

    # ---- background loops -----------------------------------------------

    def _spawner_loop(self) -> None:
        """Pull worker IDs out of self.queue and spawn subprocesses (cap)."""
        while not self._shutdown.is_set():
            # Capacity check.
            with self._lock:
                slots = self.concurrency - len(self.running)
            if slots <= 0:
                time.sleep(0.2)
                continue
            try:
                wid = self.queue.get(timeout=0.5)
            except queue.Empty:
                # Also pull pending workers from the store directly (so a
                # client that crashed before ack still gets its work done).
                self._enqueue_pending_from_store()
                continue
            self._spawn_worker(wid)

    def _reaper_loop(self) -> None:
        """Notice when worker subprocesses exit and clean their entry."""
        while not self._shutdown.is_set():
            with self._lock:
                items = list(self.running.items())
            for wid, rp in items:
                ret = rp.proc.poll()
                if ret is None:
                    continue
                # Worker exited. Clean up.
                with self._lock:
                    self.running.pop(wid, None)
                logger.info("worker %s exited rc=%s", wid, ret)
                # If the worker died without setting state, mark it failed.
                w = self.store.get_worker(wid)
                if w and w.state == STATE_RUNNING:
                    self.store.transition_worker(
                        wid,
                        state=STATE_FAILED if ret != 0 else STATE_COMPLETED,
                        error=(None if ret == 0 else f"exit code {ret}"),
                    )
            time.sleep(0.5)

    def _scheduler_loop(self) -> None:
        """Fire due scheduled tasks → enqueue a fresh worker."""
        while not self._shutdown.is_set():
            try:
                self._tick_scheduler()
            except Exception:
                logger.exception("scheduler tick failed")
            time.sleep(self.poll_seconds)

    # ---- scheduler internals --------------------------------------------

    def _tick_scheduler(self) -> None:
        now = time.time()
        for task in self.store.due_tasks(now=now):
            wid = self.store.create_worker(
                task.prompt,
                spawned_by=f"scheduler:{task.name}",
                model=self.cfg["model"]["model"],
                max_iterations=int(self.cfg.get("max_iterations", 12)),
                metadata={"task_name": task.name},
            ).id
            next_run = self._compute_next_run(task, now=now)
            self.store.mark_task_run(task.name, when=now, next_run=next_run)
            logger.info("scheduler: enqueued worker %s for task %s; next_run=%s",
                          wid, task.name, next_run)
            self.queue.put(wid)

    def _compute_next_run(self, task, *, now: float) -> float | None:
        if task.interval_seconds:
            return now + int(task.interval_seconds)
        if task.cron:
            try:
                from croniter import croniter
            except ImportError:
                logger.warning("cron task %s but `croniter` not installed; "
                                "skipping recurrence", task.name)
                return None
            return croniter(task.cron, now).get_next(float)
        return None

    # ---- worker spawning -------------------------------------------------

    def _spawn_worker(self, worker_id: str) -> None:
        w = self.store.get_worker(worker_id)
        if w is None or w.state in (STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED):
            return

        py = sys.executable or "python3"
        cmd = [py, "-m", "msa.worker", "--id", worker_id,
                "--config", self.config_path]
        if w.spawned_by == "chat" and w.metadata.get("is_master"):
            cmd.append("--master")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Each worker gets its own role tag (used by tools/meta_tool to
        # decide which tools register).
        env["MSA_AGENT_ROLE"] = (
            "master" if w.metadata.get("is_master") else "worker"
        )
        # Re-source ~/.msa.env (or whatever MSA_ENV_FILE points at) so a
        # previous worker's ptz_find_camera result -- written to disk --
        # propagates to the next worker spawn without restarting the
        # supervisor. We parse `KEY=VALUE` / `export KEY="VALUE"` lines
        # ourselves to avoid pulling python-dotenv as a dep.
        _merge_env_file(env)

        try:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, start_new_session=True,
            )
        except Exception as exc:
            logger.exception("failed to spawn worker %s", worker_id)
            self.store.transition_worker(
                worker_id, state=STATE_FAILED,
                error=f"spawn failed: {exc}",
            )
            return

        with self._lock:
            self.running[worker_id] = _RunningProc(
                worker_id=worker_id, proc=proc, started_at=time.time(),
            )
        logger.info("spawned worker %s pid=%d", worker_id, proc.pid)

    def _enqueue_pending_from_store(self) -> None:
        # Anything pending that isn't currently running gets re-enqueued.
        with self._lock:
            in_flight = set(self.running.keys())
        for w in self.store.list_workers(states=[STATE_PENDING], limit=50):
            if w.id in in_flight:
                continue
            self.queue.put(w.id)

    def _recover_orphans(self) -> None:
        for w in self.store.list_workers(states=ACTIVE_STATES, limit=500):
            if w.state == STATE_RUNNING and not _pid_alive(w.pid):
                logger.warning("recovering orphan worker %s (state=running, "
                                "pid=%s gone)", w.id, w.pid)
                self.store.transition_worker(
                    w.id, state=STATE_FAILED,
                    error="supervisor restart; pid was gone",
                )
            elif w.state == STATE_PENDING:
                self.queue.put(w.id)

    # ---- IPC method registration ----------------------------------------

    def _register_methods(self, srv: ipc.Server) -> None:
        @srv.method()
        def ping():
            return {"ok": True, "version": 1, "concurrency": self.concurrency,
                    "running": len(self.running)}

        @srv.method()
        def schedule_task(name, prompt, *, cron=None, interval_seconds=None,
                            enabled=True):
            if not name or not prompt:
                raise ValueError("name and prompt required")
            if (cron is None) == (interval_seconds is None):
                raise ValueError(
                    "exactly one of cron or interval_seconds is required"
                )
            now = time.time()
            next_run = (
                now + int(interval_seconds) if interval_seconds
                else self._compute_next_run_for_cron(cron, now)
            )
            t = self.store.upsert_task(
                name=name, prompt=prompt,
                cron=cron, interval_seconds=interval_seconds,
                enabled=bool(enabled), next_run=next_run,
            )
            return _task_to_dict(t)

        @srv.method()
        def list_tasks():
            return [_task_to_dict(t) for t in self.store.list_tasks()]

        @srv.method()
        def delete_task(name):
            return {"deleted": self.store.delete_task(name)}

        @srv.method()
        def run_task_now(prompt, *, name=None, parent_id=None,
                          spawned_by="chat", is_master=False):
            if not prompt:
                raise ValueError("prompt required")
            metadata = {"name": name} if name else {}
            if is_master:
                metadata["is_master"] = True
            w = self.store.create_worker(
                prompt,
                spawned_by=spawned_by,
                parent_id=parent_id,
                model=self.cfg["model"]["model"],
                max_iterations=int(self.cfg.get("max_iterations", 12)),
                metadata=metadata,
            )
            self.queue.put(w.id)
            return _worker_to_dict(w)

        @srv.method()
        def list_workers(*, states=None, limit=50):
            return [_worker_to_dict(w) for w in
                     self.store.list_workers(states=states, limit=limit)]

        @srv.method()
        def get_worker(worker_id):
            w = self.store.get_worker(worker_id)
            return _worker_to_dict(w) if w else None

        @srv.method()
        def cancel_worker(worker_id):
            with self._lock:
                rp = self.running.get(worker_id)
            if rp:
                try:
                    rp.proc.terminate()
                except OSError as exc:
                    return {"ok": False, "error": str(exc)}
                return {"ok": True, "signal": "SIGTERM"}
            # Not running. If pending, mark cancelled in store.
            w = self.store.get_worker(worker_id)
            if w is None:
                raise ValueError(f"worker {worker_id!r} not found")
            if w.state == STATE_PENDING:
                self.store.transition_worker(
                    worker_id, state=STATE_CANCELLED,
                    error="cancelled before start",
                )
                return {"ok": True, "signal": "pending-removed"}
            return {"ok": False, "error": f"worker is {w.state}"}

        @srv.method()
        def get_transcript(worker_id, *, since_seq=0):
            return self.store.get_transcript(worker_id, since_seq=since_seq)

        @srv.subscription()
        def subscribe_events(*, worker_id=None, after_id=0) -> Iterator[dict]:
            """Long-lived subscription that polls events table.

            Simple polling beats inventing a pub/sub layer; the events
            table is small + indexed.
            """
            cursor = int(after_id or 0)
            while not self._shutdown.is_set():
                events = self.store.events_since(
                    worker_id=worker_id, after_id=cursor, limit=200,
                )
                for e in events:
                    cursor = e.id
                    yield {
                        "id": e.id, "worker_id": e.worker_id,
                        "ts": e.ts, "kind": e.kind, "payload": e.payload,
                    }
                time.sleep(0.5)

    def _compute_next_run_for_cron(self, cron: str, now: float) -> float | None:
        try:
            from croniter import croniter
        except ImportError:
            raise RuntimeError(
                "cron schedules require `croniter`; pip install croniter"
            )
        return croniter(cron, now).get_next(float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _worker_to_dict(w: Worker) -> dict:
    d = asdict(w)
    d["total_tokens"] = w.total_tokens
    d["runtime_seconds"] = w.runtime_seconds
    return d


def _task_to_dict(t) -> dict:
    return asdict(t)


# ---------------------------------------------------------------------------
# Daemonisation (POSIX double fork)
# ---------------------------------------------------------------------------

def daemonize() -> None:
    if os.fork():
        sys.exit(0)
    os.setsid()
    if os.fork():
        sys.exit(0)
    os.umask(0o077)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    os.dup2(null, 2)
    os.close(null)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m msa.supervisor")
    p.add_argument("--daemonize", action="store_true",
                    help="POSIX double-fork; PID written to ~/.msa/supervisord.pid")
    p.add_argument("--no-scheduler", action="store_true",
                    help="don't start the scheduler thread (IPC only; for tests)")
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args(argv)

    if args.daemonize:
        daemonize()
        try:
            default_pid_path().write_text(str(os.getpid()))
        except OSError:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sup = Supervisor(config_path=args.config,
                       with_scheduler=not args.no_scheduler)
    sup.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
