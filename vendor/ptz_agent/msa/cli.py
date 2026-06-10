"""msa/cli.py — Top-level CLI: chat REPL, worker subcommands, helpers.

Single binary (``python -m msa`` or the installed ``msa`` script) that
covers the whole user surface:

    msa                       # opens chat (auto-starts supervisor)
    msa chat                  # same
    msa workers               # list workers
    msa logs <id>             # tail a worker's transcript
    msa tail <id>             # follow a worker's events live
    msa cancel <id>           # cancel a running worker
    msa schedule list         # list scheduled tasks
    msa schedule rm <name>    # delete a schedule
    msa task <prompt>         # one-off worker via supervisor
    msa supervisor [--fg]     # start the supervisor (for systemd)
    msa webui [--port N]      # serve the dashboard

The chat REPL is a master agent: each user message becomes one master
"worker" run that has access to meta tools (schedule, spawn, etc.) plus
the regular tool registry. The master worker's output is rendered live
to the terminal as tool calls fire.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import readline  # noqa: F401  (enables line editing on stdin)
import shlex
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import ipc, supervisor
from .config import load_config
from .store import (
    EVT_FAILED,
    EVT_FINISHED,
    EVT_LOG,
    EVT_PROGRESS,
    EVT_RESPONSE,
    EVT_STARTED,
    EVT_TOOL_CALL,
    EVT_TOOL_RESULT,
    Store,
)

logger = logging.getLogger("msa.cli")


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

if sys.stdout.isatty():
    BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
    GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"
else:
    BOLD = DIM = RESET = GREEN = YELLOW = RED = BLUE = MAGENTA = CYAN = ""


def _say(role: str, msg: str) -> None:
    """Format role: text. Multi-line indents under the role label."""
    label = {
        "you":      f"{BOLD}{CYAN}you>{RESET} ",
        "agent":    f"{BOLD}{GREEN}agent>{RESET} ",
        "tool":     f"{DIM}{MAGENTA}  ↳ {RESET}",
        "result":   f"{DIM}    {RESET}",
        "warn":     f"{YELLOW}  !  {RESET}",
        "error":    f"{RED}  ✗  {RESET}",
        "info":     f"{DIM}  ·  {RESET}",
    }.get(role, role + ": ")
    for i, line in enumerate(msg.splitlines() or [""]):
        print(f"{label if i == 0 else '    '}{line}")


class _Spinner:
    """In-place braille spinner that shares a line with status text.

    Heartbeats from the worker arrive every 5 seconds, but we tick at
    ~10 Hz so the user sees something *moving* — a silent terminal
    looks indistinguishable from a hang. Writes use ``\\r`` plus the
    ANSI clear-to-EOL escape so each tick overwrites the previous,
    keeping the scrollback clean during long ``ptz_scan`` /
    ``ptz_calibrate`` calls.

    When stdout is not a TTY (CI, ``msa chat | tee log``) the spinner
    is a no-op; callers should still print one line per heartbeat
    explicitly so log capture isn't empty.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty()
        self._msg = ""
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, msg: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            self._msg = msg
        if self._thread is not None:
            return
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._spin, name="cli-spinner", daemon=True,
        )
        self._thread.start()

    def set_message(self, msg: str) -> None:
        with self._lock:
            self._msg = msg

    def stop(self) -> None:
        if not self.enabled or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.3)
        self._thread = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    @contextmanager
    def paused(self):
        """Suppress the spinner so the caller can print on its own line."""
        if not self.enabled or self._thread is None:
            yield
            return
        self._paused.set()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        try:
            yield
        finally:
            self._paused.clear()

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            if not self._paused.is_set():
                with self._lock:
                    msg = self._msg
                frame = self.FRAMES[i % len(self.FRAMES)]
                sys.stdout.write(f"\r\033[K  {DIM}{frame}{RESET}  {msg}")
                sys.stdout.flush()
            i += 1
            if self._stop.wait(0.1):
                break


# ---------------------------------------------------------------------------
# Chat REPL
# ---------------------------------------------------------------------------

class Chat:
    """Master-agent chat REPL.

    Each user turn enqueues a master worker via the supervisor (so token
    counts, transcripts, etc. are unified with all other workers in
    SQLite) and tails the worker's events live, rendering tool calls and
    the final response inline.
    """

    def __init__(self, *, model_label: str | None = None):
        self.client = ipc.Client()
        self.store = Store()
        self.parent_id: str | None = None
        self.model_label = model_label
        self.spinner = _Spinner()

    def _emit(self, role: str, msg: str) -> None:
        """Print via _say with the spinner paused so the line stays clean."""
        with self.spinner.paused():
            _say(role, msg)

    # ---- entry -----------------------------------------------------------

    def run(self) -> int:
        # Banner.
        try:
            ping = self.client.ping()
        except ipc.IPCError as exc:
            _say("error", f"supervisor unreachable: {exc}")
            return 1
        cfg = load_config("config/config.yaml")
        model = self.model_label or cfg.get("model", {}).get("model", "?")
        backend = cfg.get("model", {}).get("backend", "?")
        running = ping.get("running", 0)
        print(f"{BOLD}MSA chat{RESET} — {model} on {backend}; "
              f"{running} worker(s) running. Ctrl+D to exit. "
              f"Slash-commands: /workers /log <id> /tail <id> /cancel <id> "
              "/tasks /help")

        try:
            while True:
                try:
                    line = input(f"{BOLD}{CYAN}you>{RESET} ").rstrip()
                except EOFError:
                    print()
                    return 0
                except KeyboardInterrupt:
                    print()
                    continue
                if not line:
                    continue
                if line.startswith("/"):
                    self._slash(line)
                    continue
                self._turn(line)
        except KeyboardInterrupt:
            print()
            return 0

    # ---- one turn --------------------------------------------------------

    def _turn(self, user_text: str) -> None:
        # Each user turn becomes a master worker run. parent_id chains
        # turns together so the web UI can show a conversation tree, AND
        # so we can reconstruct prior turns as context here. Without the
        # context preamble, a follow-up like "where was the person" would
        # spawn a fresh master with zero memory of the worker that just
        # answered the previous turn.
        prompt = self._with_session_context(user_text)
        try:
            w = self.client.run_task_now(
                prompt=prompt,
                parent_id=self.parent_id,
                spawned_by="chat",
                is_master=True,
            )
        except ipc.IPCError as exc:
            _say("error", str(exc))
            return
        self.parent_id = w["id"]
        self._tail_until_finished(w["id"])

    def _with_session_context(self, user_text: str,
                                 max_turns: int = 5,
                                 max_children_per_turn: int = 4) -> str:
        """Prepend a recap of recent chat turns + their spawned workers.

        Walks the parent_id chain backwards from the most recent master
        worker in this session (capped at ``max_turns`` for prompt size)
        and assembles a compact "[Earlier in this chat session]" block.
        Each entry shows the user's question, the master's `respond`
        message, and any background workers it spawned with their final
        results — exactly the missing puzzle pieces the master needs to
        answer follow-up questions without re-running expensive tools.

        Falls back to ``user_text`` unchanged on the very first turn or
        if the store query fails (we never want a context-build error to
        block the chat).
        """
        if not self.parent_id:
            return user_text
        try:
            masters = []
            cur: str | None = self.parent_id
            while cur and len(masters) < max_turns:
                w = self.store.get_worker(cur)
                if w is None:
                    break
                masters.insert(0, w)  # oldest first
                cur = w.parent_id
        except Exception as exc:  # noqa: BLE001
            logger.debug("session-context build failed: %s", exc)
            return user_text

        if not masters:
            return user_text

        lines = ["[Earlier in this chat session — for context, do NOT re-run "
                 "tools that already produced these answers]"]
        for m in masters:
            user_msg = (m.prompt or "").strip().splitlines()[0][:240]
            agent_msg = ((m.result or "").strip().splitlines()[0]
                         if m.result else f"({m.state})")
            agent_msg = agent_msg[:400]
            lines.append(f"- you: {user_msg}")
            lines.append(f"  agent: {agent_msg}")
            try:
                children = self.store.list_workers(
                    parent_id=m.id, limit=max_children_per_turn,
                )
            except Exception:  # noqa: BLE001
                children = []
            for c in children:
                cresult = (c.result or "").strip().splitlines()[0][:240] \
                          if c.result else f"({c.state})"
                lines.append(
                    f"    spawned worker {c.id} [{c.state}]: "
                    f"prompt={(c.prompt or '')[:80]!r} → {cresult}"
                )
        lines.append("[End earlier context]")
        lines.append("")
        lines.append(f"User now asks: {user_text}")
        return "\n".join(lines)

    def _tail_until_finished(self, worker_id: str) -> None:
        """Block until the worker terminates, rendering events as they fire.

        A single spinner shares the bottom line with the heartbeat
        message ("iter 2: running ptz_scan… 25s"); on every other
        event we pause the spinner, print on its own line, and let
        the spinner resume on the next line. Non-TTY callers (piped
        output) still get one line per 5 s heartbeat for log capture.
        """
        cur_iter = 0
        cur_phase = ""
        try:
            for ev in self.client.stream_events(worker_id=worker_id):
                kind = ev.get("kind")
                payload = ev.get("payload") or {}

                if kind == EVT_STARTED:
                    self._emit("info", f"worker {worker_id} started "
                                       f"({payload.get('model')})")
                    self.spinner.start("starting…")

                elif kind == EVT_PROGRESS:
                    it = payload.get("iteration", 0)
                    phase = payload.get("phase") or ""
                    elapsed = payload.get("elapsed_s") or 0
                    if phase.startswith("tool:"):
                        label = f"running {phase.split(':', 1)[1]}"
                    elif phase == "tool":
                        label = "tool"
                    elif phase == "thinking":
                        label = "thinking"
                    else:
                        label = phase or "working"
                    if it != cur_iter:
                        cur_iter, cur_phase = it, phase
                    elif phase != cur_phase:
                        cur_phase = phase
                    msg = f"iter {it}: {label}…"
                    if elapsed:
                        msg += f" {elapsed:.0f}s"
                    self.spinner.start(msg)
                    self.spinner.set_message(msg)
                    if not self.spinner.enabled and elapsed >= 5:
                        _say("info",
                             f"iter {it}: {label}… {elapsed:.0f}s elapsed")

                elif kind == EVT_TOOL_CALL:
                    args_preview = json.dumps(payload.get("args") or {})[:120]
                    prefix = f"[iter {cur_iter}] " if cur_iter else ""
                    self._emit(
                        "tool",
                        f"{prefix}{payload.get('tool')}({args_preview})",
                    )

                elif kind == EVT_TOOL_RESULT:
                    preview = (payload.get("preview") or "").strip()
                    if preview:
                        self._emit("result", _shorten(preview, 240))

                elif kind == EVT_LOG:
                    level = payload.get("level", "info")
                    say = {"warning": "warn",
                           "error": "error"}.get(level, "info")
                    self._emit(say, payload.get("message") or "")

                elif kind == EVT_RESPONSE:
                    self.spinner.stop()
                    _say("agent", payload.get("message") or "")

                elif kind == EVT_FINISHED:
                    self.spinner.stop()
                    return

                elif kind == EVT_FAILED:
                    self.spinner.stop()
                    _say("error", payload.get("reason") or "worker failed")
                    return
        except ipc.IPCError as exc:
            self.spinner.stop()
            _say("error", f"event stream lost: {exc}")
        finally:
            self.spinner.stop()

    # ---- slash commands --------------------------------------------------

    def _slash(self, line: str) -> None:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            _say("error", str(exc)); return
        cmd = parts[0]
        args = parts[1:]

        if cmd in ("/q", "/quit", "/exit"):
            raise EOFError

        if cmd in ("/?", "/help"):
            print("Slash-commands:")
            print("  /workers              list recent workers")
            print("  /log <id>             show transcript")
            print("  /tail <id>            follow events live (Ctrl+C to stop)")
            print("  /cancel <id>          SIGTERM a running worker")
            print("  /tasks                list scheduled tasks")
            print("  /rmtask <name>        delete a scheduled task")
            print("  /quit                 exit")
            return

        if cmd == "/workers":
            ws = self.client.list_workers(limit=20)
            if not ws:
                _say("info", "no workers yet"); return
            print(f"{BOLD}{'ID':12} {'STATE':10} {'ITER':>5} {'TOK':>6} {'RUNTIME':>9}  PROMPT{RESET}")
            for w in ws:
                rt = w.get("runtime_seconds")
                rt_s = f"{rt:.1f}s" if rt is not None else "—"
                print(f"{w['id']:12} {w['state']:10} "
                       f"{w.get('iterations', 0):>5} "
                       f"{w.get('total_tokens', 0):>6} "
                       f"{rt_s:>9}  "
                       f"{(w.get('prompt') or '')[:60]}")
            return

        if cmd == "/log":
            if not args:
                _say("error", "usage: /log <worker_id>"); return
            entries = self.client.get_transcript(args[0])
            for e in entries:
                role = e.get("role", "?")
                content = e.get("content", "")
                _say("info", f"[{role}] {_shorten(content, 600)}")
            return

        if cmd == "/tail":
            if not args:
                _say("error", "usage: /tail <worker_id>"); return
            try:
                self._tail_until_finished(args[0])
            except KeyboardInterrupt:
                print()
            return

        if cmd == "/cancel":
            if not args:
                _say("error", "usage: /cancel <worker_id>"); return
            try:
                r = self.client.cancel_worker(args[0])
                _say("info", json.dumps(r))
            except ipc.IPCError as exc:
                _say("error", str(exc))
            return

        if cmd == "/tasks":
            tasks = self.client.list_tasks()
            if not tasks:
                _say("info", "no scheduled tasks"); return
            for t in tasks:
                sched = (
                    f"every {t['interval_seconds']}s"
                    if t.get("interval_seconds")
                    else f"cron {t.get('cron')}"
                )
                print(f"  {t['name']}  [{sched}]  next_run={t.get('next_run')}  "
                       f"runs={t.get('run_count')}")
            return

        if cmd == "/rmtask":
            if not args:
                _say("error", "usage: /rmtask <name>"); return
            try:
                r = self.client.delete_task(args[0])
                _say("info", json.dumps(r))
            except ipc.IPCError as exc:
                _say("error", str(exc))
            return

        _say("error", f"unknown command: {cmd}. /help for list.")


def _shorten(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "…"


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_chat(_args) -> int:
    if not supervisor.ensure_running():
        _say("error", "supervisor failed to start; check ~/.msa/logs/supervisor.log")
        return 1
    return Chat().run()


def cmd_workers(args) -> int:
    if not supervisor.ensure_running():
        return 1
    ws = ipc.Client().list_workers(limit=args.limit)
    if not ws:
        print("(no workers)")
        return 0
    print(f"{'ID':12} {'STATE':10} {'ITER':>5} {'TOK':>6} {'RUNTIME':>9}  PROMPT")
    for w in ws:
        rt = w.get("runtime_seconds")
        rt_s = f"{rt:.1f}s" if rt is not None else "—"
        print(f"{w['id']:12} {w['state']:10} "
               f"{w.get('iterations', 0):>5} "
               f"{w.get('total_tokens', 0):>6} "
               f"{rt_s:>9}  "
               f"{(w.get('prompt') or '')[:60]}")
    return 0


def cmd_logs(args) -> int:
    if not supervisor.ensure_running():
        return 1
    entries = ipc.Client().get_transcript(args.worker_id)
    for e in entries:
        print(f"[{e.get('role')}] {e.get('content')}")
        if e.get("meta"):
            print(f"   meta={json.dumps(e['meta'])}")
    return 0


def cmd_tail(args) -> int:
    if not supervisor.ensure_running():
        return 1
    client = ipc.Client()
    try:
        for ev in client.stream_events(worker_id=args.worker_id):
            ts = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
            print(f"{ts} [{ev['kind']}] {json.dumps(ev.get('payload') or {})[:200]}")
            if ev["kind"] in (EVT_FINISHED, EVT_FAILED):
                return 0
    except KeyboardInterrupt:
        return 0


def cmd_cancel(args) -> int:
    if not supervisor.ensure_running():
        return 1
    print(json.dumps(ipc.Client().cancel_worker(args.worker_id)))
    return 0


def cmd_task(args) -> int:
    if not supervisor.ensure_running():
        return 1
    w = ipc.Client().run_task_now(
        prompt=args.prompt, name=args.name, spawned_by="cli",
    )
    print(f"started worker {w['id']}")
    if args.wait:
        # tail until done
        try:
            for ev in ipc.Client().stream_events(worker_id=w["id"]):
                if ev["kind"] == EVT_RESPONSE:
                    print(ev["payload"].get("message", ""))
                if ev["kind"] in (EVT_FINISHED, EVT_FAILED):
                    return 0 if ev["kind"] == EVT_FINISHED else 1
        except KeyboardInterrupt:
            return 130
    return 0


def cmd_schedule_list(_args) -> int:
    if not supervisor.ensure_running():
        return 1
    for t in ipc.Client().list_tasks():
        sched = (f"every {t['interval_seconds']}s"
                  if t.get("interval_seconds") else f"cron {t.get('cron')}")
        print(f"{t['name']}  {sched}  next_run={t.get('next_run')}  "
               f"runs={t.get('run_count')}  enabled={t.get('enabled')}")
    return 0


def cmd_schedule_rm(args) -> int:
    if not supervisor.ensure_running():
        return 1
    print(json.dumps(ipc.Client().delete_task(args.name)))
    return 0


def cmd_supervisor(args) -> int:
    return supervisor.main(["--config", args.config] +
                              ([] if args.fg else ["--daemonize"]))


def cmd_webui(args) -> int:
    if not supervisor.ensure_running():
        return 1
    from . import webui
    webui.serve(host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="msa")
    p.add_argument("--config", default="config/config.yaml")
    sp = p.add_subparsers(dest="cmd")

    sp_chat = sp.add_parser("chat", help="open the chat REPL")
    sp_chat.set_defaults(func=cmd_chat)

    sp_w = sp.add_parser("workers", help="list workers")
    sp_w.add_argument("--limit", type=int, default=30)
    sp_w.set_defaults(func=cmd_workers)

    sp_l = sp.add_parser("logs", help="show a worker's transcript")
    sp_l.add_argument("worker_id")
    sp_l.set_defaults(func=cmd_logs)

    sp_t = sp.add_parser("tail", help="follow a worker's events live")
    sp_t.add_argument("worker_id")
    sp_t.set_defaults(func=cmd_tail)

    sp_c = sp.add_parser("cancel", help="SIGTERM a running worker")
    sp_c.add_argument("worker_id")
    sp_c.set_defaults(func=cmd_cancel)

    sp_task = sp.add_parser("task", help="run a one-off task via supervisor")
    sp_task.add_argument("prompt")
    sp_task.add_argument("--name", default=None)
    sp_task.add_argument("--wait", action="store_true",
                          help="block until the worker finishes")
    sp_task.set_defaults(func=cmd_task)

    sp_sched = sp.add_parser("schedule", help="manage scheduled tasks")
    sched_sub = sp_sched.add_subparsers(dest="schedule_cmd")
    sched_ls = sched_sub.add_parser("list")
    sched_ls.set_defaults(func=cmd_schedule_list)
    sched_rm = sched_sub.add_parser("rm")
    sched_rm.add_argument("name")
    sched_rm.set_defaults(func=cmd_schedule_rm)

    sp_sup = sp.add_parser("supervisor", help="start the supervisor")
    sp_sup.add_argument("--fg", action="store_true",
                          help="foreground (don't daemonize)")
    sp_sup.set_defaults(func=cmd_supervisor)

    sp_web = sp.add_parser("webui", help="serve the worker dashboard")
    sp_web.add_argument("--host", default="127.0.0.1")
    sp_web.add_argument("--port", type=int, default=8765)
    sp_web.set_defaults(func=cmd_webui)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        return cmd_chat(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
