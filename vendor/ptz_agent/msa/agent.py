"""msa/agent.py — Backwards-compat shim over the new supervisor + worker stack.

The legacy CLI:

    python -m msa.agent --once
    python -m msa.agent --task "do X"
    python -m msa.agent --schedule
    python -m msa.agent --status
    python -m msa.agent --sensors
    python -m msa.agent --read-sensor NAME
    python -m msa.agent --tasks

is preserved here so existing scripts and docs keep working. Behaviour:

  * ``--task PROMPT``   enqueue a worker via the supervisor and tail it
                         until it finishes (auto-starts the supervisor).
  * ``--once``           run the most-recent pending task, or print a
                         friendly message if there isn't one.
  * ``--schedule``       no-op (the supervisor already runs scheduled
                         tasks when it's up); prints how to start it.
  * ``--status``         dump the active scratchpad.
  * ``--sensors``        list registered sensors.
  * ``--read-sensor``    read one sensor.
  * ``--tasks``          list scheduled tasks.

For new code: use ``msa.cli`` / ``msa.supervisor`` / ``msa.worker``
directly. The cycle/scratchpad concepts are gone; each worker is a
single, scoped run.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import ipc, supervisor
from .config import load_config
from .sensors import SensorRegistry
from .store import EVT_FAILED, EVT_FINISHED, EVT_RESPONSE, Store


def cmd_status() -> int:
    store = Store()
    workers = store.list_workers(limit=10)
    if not workers:
        print("(no workers; run `msa task \"...\"` or `msa chat`)")
        return 0
    print(f"{'ID':12} {'STATE':10} {'ITER':>5} {'TOK':>6}  PROMPT")
    for w in workers:
        print(f"{w.id:12} {w.state:10} "
               f"{w.iterations:>5} {w.total_tokens:>6}  "
               f"{(w.prompt or '')[:70]}")
    return 0


def cmd_sensors() -> int:
    cfg = load_config("config/config.yaml")
    registry = SensorRegistry(cfg)
    print(registry.describe() or "(no sensors)")
    return 0


def cmd_read_sensor(name: str) -> int:
    cfg = load_config("config/config.yaml")
    registry = SensorRegistry(cfg)
    print(json.dumps(registry.read(name), indent=2, default=str))
    return 0


def cmd_tasks() -> int:
    if not supervisor.ensure_running():
        return 1
    tasks = ipc.Client().list_tasks()
    if not tasks:
        print("(no scheduled tasks)")
        return 0
    print(json.dumps(tasks, indent=2, default=str))
    return 0


def cmd_task(prompt: str) -> int:
    if not supervisor.ensure_running():
        return 1
    client = ipc.Client()
    w = client.run_task_now(prompt=prompt, spawned_by="cli")
    print(f"started worker {w['id']}")
    try:
        for ev in client.stream_events(worker_id=w["id"]):
            if ev["kind"] == EVT_RESPONSE:
                print((ev["payload"] or {}).get("message", ""))
            if ev["kind"] in (EVT_FINISHED, EVT_FAILED):
                return 0 if ev["kind"] == EVT_FINISHED else 1
    except KeyboardInterrupt:
        return 130


def cmd_once() -> int:
    """Legacy --once: run the oldest pending worker, or noop."""
    if not supervisor.ensure_running():
        return 1
    store = Store()
    pending = store.list_workers(states=["pending"], limit=1)
    if not pending:
        print("(no pending workers; pass --task '...' or use `msa chat`)")
        return 0
    # The supervisor will pick it up automatically; just tail it.
    return cmd_task_tail(pending[0].id)


def cmd_task_tail(worker_id: str) -> int:
    try:
        for ev in ipc.Client().stream_events(worker_id=worker_id):
            if ev["kind"] == EVT_RESPONSE:
                print((ev["payload"] or {}).get("message", ""))
            if ev["kind"] in (EVT_FINISHED, EVT_FAILED):
                return 0 if ev["kind"] == EVT_FINISHED else 1
    except KeyboardInterrupt:
        return 130


def cmd_schedule() -> int:
    print(
        "Scheduled tasks run via the supervisor. Make sure it's up:\n"
        "    msa supervisor               # foreground (--fg)\n"
        "    msa supervisor               # already daemonised by `msa chat`\n"
        "Schedule tasks via `msa chat` or `msa schedule`. "
        "Inspect with `msa schedule list`."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m msa.agent",
                                  description="Legacy MSA CLI shim")
    p.add_argument("--once", action="store_true")
    p.add_argument("--schedule", action="store_true")
    p.add_argument("--task", type=str, metavar="PROMPT")
    p.add_argument("--status", action="store_true")
    p.add_argument("--sensors", action="store_true")
    p.add_argument("--read-sensor", type=str, metavar="NAME")
    p.add_argument("--tasks", action="store_true")
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args(argv)

    if args.status:
        return cmd_status()
    if args.sensors:
        return cmd_sensors()
    if args.read_sensor:
        return cmd_read_sensor(args.read_sensor)
    if args.tasks:
        return cmd_tasks()
    if args.task is not None:
        return cmd_task(args.task)
    if args.once:
        return cmd_once()
    if args.schedule:
        return cmd_schedule()
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
