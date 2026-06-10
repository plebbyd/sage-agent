"""msa/store.py — SQLite-backed persistent state for the MSA supervisor.

Owns:
    * workers          one row per agent run (chat session, scheduled run, etc.)
    * transcripts      ordered messages per worker (user/assistant/tool/event)
    * scheduled_tasks  recurring task definitions (interval or cron)
    * events           append-only timeline used by the chat REPL + web UI
                       to render live progress

All other modules go through this module — no raw SQL elsewhere. Reads
are lock-free; writes go through ``_with_write`` which holds a process-
local lock so the writer's BEGIN IMMEDIATE doesn't lose to a reader.

The supervisor is the only writer for ``workers`` / ``scheduled_tasks``,
but workers append to ``transcripts`` / ``events`` directly (faster than
RPC for high-volume rows).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Worker lifecycle states. Subset is closed; transitions are
# pending → running → (completed | failed | cancelled).
STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

ACTIVE_STATES = (STATE_PENDING, STATE_RUNNING)
TERMINAL_STATES = (STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED)

# Event kinds the supervisor / worker emit. Used by the chat REPL and
# the web UI to render the live timeline.
EVT_STARTED = "started"
EVT_TOOL_CALL = "tool_call"
EVT_TOOL_RESULT = "tool_result"
EVT_RESPONSE = "response"
EVT_PROGRESS = "progress"
EVT_FINISHED = "finished"
EVT_FAILED = "failed"
EVT_CANCELLED = "cancelled"
EVT_LOG = "log"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = r"""
CREATE TABLE IF NOT EXISTS workers (
    id              TEXT PRIMARY KEY,
    parent_id       TEXT,
    spawned_by      TEXT NOT NULL,         -- 'chat' | 'scheduler:<name>' | 'cli'
    prompt          TEXT NOT NULL,
    state           TEXT NOT NULL,         -- pending|running|completed|failed|cancelled
    pid             INTEGER,
    model           TEXT,
    max_iterations  INTEGER NOT NULL DEFAULT 12,
    iterations      INTEGER NOT NULL DEFAULT 0,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    result          TEXT,
    error           TEXT,
    created_at      REAL NOT NULL,
    started_at      REAL,
    finished_at     REAL,
    metadata        TEXT                   -- arbitrary JSON
);
CREATE INDEX IF NOT EXISTS idx_workers_state ON workers(state);
CREATE INDEX IF NOT EXISTS idx_workers_parent ON workers(parent_id);
CREATE INDEX IF NOT EXISTS idx_workers_created ON workers(created_at);

CREATE TABLE IF NOT EXISTS transcripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id   TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    ts          REAL NOT NULL,
    role        TEXT NOT NULL,             -- system|user|assistant|tool|event
    content     TEXT NOT NULL,
    meta        TEXT,                      -- JSON: tool name/args/tokens/etc.
    FOREIGN KEY (worker_id) REFERENCES workers(id)
);
CREATE INDEX IF NOT EXISTS idx_transcripts_worker ON transcripts(worker_id, seq);

CREATE TABLE IF NOT EXISTS scheduled_tasks (
    name             TEXT PRIMARY KEY,
    prompt           TEXT NOT NULL,
    cron             TEXT,                 -- '*/5 * * * *' style; mutually exclusive w/ interval
    interval_seconds INTEGER,
    enabled          INTEGER NOT NULL DEFAULT 1,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    last_run         REAL,
    next_run         REAL,
    run_count        INTEGER NOT NULL DEFAULT 0,
    metadata         TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id   TEXT NOT NULL,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT,                      -- JSON
    FOREIGN KEY (worker_id) REFERENCES workers(id)
);
CREATE INDEX IF NOT EXISTS idx_events_worker_ts ON events(worker_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


# ---------------------------------------------------------------------------
# Dataclasses (typed views over the rows)
# ---------------------------------------------------------------------------

@dataclass
class Worker:
    id: str
    parent_id: str | None
    spawned_by: str
    prompt: str
    state: str
    pid: int | None
    model: str | None
    max_iterations: int
    iterations: int
    prompt_tokens: int
    completion_tokens: int
    result: str | None
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def runtime_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


@dataclass
class ScheduledTask:
    name: str
    prompt: str
    cron: str | None
    interval_seconds: int | None
    enabled: bool
    created_at: float
    updated_at: float
    last_run: float | None
    next_run: float | None
    run_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Event:
    id: int
    worker_id: str
    ts: float
    kind: str
    payload: dict


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def default_db_path() -> Path:
    """Resolve a database location that works in dev and on systemd."""
    env = os.environ.get("MSA_DB_PATH")
    if env:
        return Path(env)
    home = Path.home() / ".msa"
    home.mkdir(parents=True, exist_ok=True)
    return home / "state.db"


class Store:
    """Thread-safe SQLite wrapper.

    Use one instance per process. Internally, each writer acquires a
    process-local lock to serialize ``BEGIN IMMEDIATE`` transactions;
    readers use independent connections (sqlite handles cross-process
    locking via WAL).
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        # Initialise the schema on the first connection.
        with self._connect() as con:
            con.executescript(_SCHEMA)

    # ---------------------------- connections -----------------------------

    @contextmanager
    def _connect(self):
        """Yield a thread-local connection (one per thread, kept open)."""
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(
                str(self.path),
                timeout=10,
                isolation_level=None,        # autocommit; we BEGIN explicitly
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA foreign_keys=ON")
            self._local.con = con
        try:
            yield con
        except Exception:
            try:
                con.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise

    @contextmanager
    def _write(self):
        """Hold the write lock and an IMMEDIATE transaction."""
        with self._write_lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    # ------------------------------ workers -------------------------------

    @staticmethod
    def new_worker_id() -> str:
        return f"w-{uuid.uuid4().hex[:8]}"

    def create_worker(
        self,
        prompt: str,
        *,
        spawned_by: str,
        parent_id: str | None = None,
        model: str | None = None,
        max_iterations: int = 12,
        worker_id: str | None = None,
        metadata: dict | None = None,
        state: str = STATE_PENDING,
    ) -> Worker:
        wid = worker_id or self.new_worker_id()
        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self._write() as con:
            con.execute(
                """INSERT INTO workers
                    (id, parent_id, spawned_by, prompt, state, model,
                     max_iterations, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (wid, parent_id, spawned_by, prompt, state, model,
                 max_iterations, now, meta_json),
            )
        w = self.get_worker(wid)
        assert w is not None
        return w

    def get_worker(self, worker_id: str) -> Worker | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM workers WHERE id = ?", (worker_id,)
            ).fetchone()
        return _row_to_worker(row) if row else None

    def list_workers(
        self,
        *,
        states: Iterable[str] | None = None,
        parent_id: str | None = None,
        limit: int = 100,
    ) -> list[Worker]:
        clauses, params = [], []
        if states:
            placeholders = ",".join("?" for _ in states)
            clauses.append(f"state IN ({placeholders})")
            params.extend(states)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT * FROM workers {where} "
            f"ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [_row_to_worker(r) for r in rows]

    def update_worker(self, worker_id: str, **fields) -> None:
        if not fields:
            return
        if "metadata" in fields and not isinstance(fields["metadata"], str):
            fields["metadata"] = json.dumps(fields["metadata"])
        keys = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [worker_id]
        with self._write() as con:
            con.execute(
                f"UPDATE workers SET {keys} WHERE id = ?", params
            )

    def add_tokens(
        self, worker_id: str, *,
        prompt_tokens: int = 0, completion_tokens: int = 0,
    ) -> None:
        if prompt_tokens == 0 and completion_tokens == 0:
            return
        with self._write() as con:
            con.execute(
                """UPDATE workers
                       SET prompt_tokens = prompt_tokens + ?,
                           completion_tokens = completion_tokens + ?
                     WHERE id = ?""",
                (prompt_tokens, completion_tokens, worker_id),
            )

    def increment_iterations(self, worker_id: str) -> int:
        with self._write() as con:
            cur = con.execute(
                "UPDATE workers SET iterations = iterations + 1 "
                "WHERE id = ? RETURNING iterations",
                (worker_id,),
            )
            row = cur.fetchone()
        return row[0] if row else 0

    def transition_worker(
        self, worker_id: str, *,
        state: str,
        pid: int | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        fields: dict[str, Any] = {"state": state}
        if state == STATE_RUNNING:
            fields["started_at"] = now
            if pid is not None:
                fields["pid"] = pid
        elif state in TERMINAL_STATES:
            fields["finished_at"] = now
            if result is not None:
                fields["result"] = result
            if error is not None:
                fields["error"] = error
        elif pid is not None:
            fields["pid"] = pid
        self.update_worker(worker_id, **fields)

    # ---------------------------- transcripts -----------------------------

    def append_transcript(
        self,
        worker_id: str,
        role: str,
        content: str,
        meta: dict | None = None,
    ) -> int:
        now = time.time()
        meta_json = json.dumps(meta or {})
        with self._write() as con:
            cur = con.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM transcripts "
                "WHERE worker_id = ?", (worker_id,),
            )
            seq = cur.fetchone()[0]
            con.execute(
                """INSERT INTO transcripts
                    (worker_id, seq, ts, role, content, meta)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (worker_id, seq, now, role, content, meta_json),
            )
        return seq

    def get_transcript(
        self, worker_id: str, *, since_seq: int = 0,
    ) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """SELECT seq, ts, role, content, meta FROM transcripts
                       WHERE worker_id = ? AND seq > ?
                       ORDER BY seq""",
                (worker_id, since_seq),
            ).fetchall()
        return [
            {
                "seq": r["seq"], "ts": r["ts"], "role": r["role"],
                "content": r["content"], "meta": _loads(r["meta"]),
            }
            for r in rows
        ]

    # ---------------------------- events ----------------------------------

    def emit_event(
        self, worker_id: str, kind: str, payload: dict | None = None,
    ) -> int:
        now = time.time()
        body = json.dumps(payload or {})
        with self._write() as con:
            cur = con.execute(
                "INSERT INTO events (worker_id, ts, kind, payload) "
                "VALUES (?, ?, ?, ?) RETURNING id",
                (worker_id, now, kind, body),
            )
            return cur.fetchone()[0]

    def events_since(
        self,
        *,
        worker_id: str | None = None,
        after_id: int = 0,
        limit: int = 200,
    ) -> list[Event]:
        clauses, params = ["id > ?"], [after_id]
        if worker_id:
            clauses.append("worker_id = ?")
            params.append(worker_id)
        sql = (
            f"SELECT * FROM events WHERE {' AND '.join(clauses)} "
            f"ORDER BY id LIMIT ?"
        )
        params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [
            Event(
                id=r["id"], worker_id=r["worker_id"], ts=r["ts"],
                kind=r["kind"], payload=_loads(r["payload"]),
            )
            for r in rows
        ]

    # ---------------------------- scheduled tasks -------------------------

    def upsert_task(
        self,
        name: str,
        prompt: str,
        *,
        cron: str | None = None,
        interval_seconds: int | None = None,
        enabled: bool = True,
        next_run: float | None = None,
        metadata: dict | None = None,
    ) -> ScheduledTask:
        if (cron is None) == (interval_seconds is None):
            raise ValueError(
                "scheduled task needs exactly one of `cron` or `interval_seconds`"
            )
        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self._write() as con:
            con.execute(
                """INSERT INTO scheduled_tasks
                    (name, prompt, cron, interval_seconds, enabled,
                     created_at, updated_at, next_run, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                    prompt = excluded.prompt,
                    cron = excluded.cron,
                    interval_seconds = excluded.interval_seconds,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at,
                    next_run = excluded.next_run,
                    metadata = excluded.metadata""",
                (name, prompt, cron, interval_seconds, int(enabled),
                 now, now, next_run, meta_json),
            )
        t = self.get_task(name)
        assert t is not None
        return t

    def get_task(self, name: str) -> ScheduledTask | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM scheduled_tasks WHERE name = ?", (name,)
            ).fetchone()
        return _row_to_task(row) if row else None

    def list_tasks(self) -> list[ScheduledTask]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM scheduled_tasks ORDER BY name"
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def delete_task(self, name: str) -> bool:
        with self._write() as con:
            cur = con.execute(
                "DELETE FROM scheduled_tasks WHERE name = ?", (name,)
            )
        return cur.rowcount > 0

    def mark_task_run(
        self, name: str, *, when: float, next_run: float | None,
    ) -> None:
        with self._write() as con:
            con.execute(
                """UPDATE scheduled_tasks
                    SET last_run = ?, next_run = ?, run_count = run_count + 1,
                        updated_at = ?
                    WHERE name = ?""",
                (when, next_run, time.time(), name),
            )

    def due_tasks(self, *, now: float | None = None) -> list[ScheduledTask]:
        now = time.time() if now is None else now
        with self._connect() as con:
            rows = con.execute(
                """SELECT * FROM scheduled_tasks
                    WHERE enabled = 1
                      AND next_run IS NOT NULL
                      AND next_run <= ?
                    ORDER BY next_run""",
                (now,),
            ).fetchall()
        return [_row_to_task(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loads(s: str | None) -> dict:
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def _row_to_worker(row: sqlite3.Row) -> Worker:
    return Worker(
        id=row["id"],
        parent_id=row["parent_id"],
        spawned_by=row["spawned_by"],
        prompt=row["prompt"],
        state=row["state"],
        pid=row["pid"],
        model=row["model"],
        max_iterations=row["max_iterations"],
        iterations=row["iterations"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        result=row["result"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        metadata=_loads(row["metadata"]),
    )


def _row_to_task(row: sqlite3.Row) -> ScheduledTask:
    return ScheduledTask(
        name=row["name"],
        prompt=row["prompt"],
        cron=row["cron"],
        interval_seconds=row["interval_seconds"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_run=row["last_run"],
        next_run=row["next_run"],
        run_count=row["run_count"],
        metadata=_loads(row["metadata"]),
    )
