"""Self-diagnosis & self-healing skill.

Two responsibilities:
  * **Snapshot** the system/OS state on a schedule (every N hours) so we can see
    what changed over time, with strict size/retention bounds.
  * **Heal** on demand: take an incident (or the latest recorded error) and run
    the LLM healer in the configured mode (notify_only / approval / auto).

Run modes (via :meth:`run` ``args['action']``):
  * ``snapshot`` (default) — capture one snapshot if due (or ``force=true``).
  * ``watch``              — loop forever, snapshotting every interval.
  * ``heal``               — diagnose+heal an incident from ``args`` or latest.
  * ``status``             — report snapshot history + pending heals.

CLI: ``ptz-node skill run self_diagnosis --args '{"action":"snapshot","force":true}'``
or the conveniences ``ptz-node snapshot`` / ``ptz-node heal``.
"""

from __future__ import annotations

import time
from typing import Any

from ptz_node.skills.base import BaseSkill, SkillContext, SkillResult


class SelfDiagnosisSkill(BaseSkill):
    name = "self_diagnosis"
    description = (
        "Periodically snapshots system/OS state (bounded, rotated) and uses a strong "
        "API LLM to diagnose and heal crashes/anomalies, notifying via Slack/email or "
        "waiting for human approval per self_healing.mode."
    )
    default_interval_hours = 12.0
    agent_callable = True

    def run(self, ctx: SkillContext) -> SkillResult:
        action = str(ctx.args.get("action", "snapshot")).lower()
        if action == "snapshot":
            return self._snapshot(ctx, force=bool(ctx.args.get("force", False)))
        if action == "watch":
            return self._watch(ctx)
        if action == "heal":
            return self._heal(ctx)
        if action == "status":
            return self._status(ctx)
        return SkillResult(ok=False, skill=self.name,
                           summary=f"unknown action {action!r}; "
                                   "use snapshot|watch|heal|status")

    # -- snapshot ----------------------------------------------------------

    def _snapshotter(self, ctx: SkillContext):
        from ptz_node.self_healing.snapshot import SystemSnapshotter

        return SystemSnapshotter(ctx.config)

    def _snapshot(self, ctx: SkillContext, *, force: bool) -> SkillResult:
        snap = self._snapshotter(ctx)
        if not force and not snap.is_due():
            since = snap.seconds_since_last()
            return SkillResult(
                ok=True, skill=self.name,
                summary=f"snapshot not due ({since/3600:.1f}h since last; "
                        f"interval {snap.interval_hours}h). Use force=true to override.",
                data={"due": False, "seconds_since_last": since},
            )
        path = snap.write(reason="forced" if force else "scheduled")
        idx = (snap._snapshot_files())
        return SkillResult(
            ok=True, skill=self.name,
            summary=f"captured snapshot → {path.name} (history: {len(idx)} files)",
            data={"due": True, "kept": len(idx),
                  "interval_hours": snap.interval_hours,
                  "max_total_mb": snap.max_total_mb},
            artifacts=[str(path)],
        )

    def _watch(self, ctx: SkillContext) -> SkillResult:
        snap = self._snapshotter(ctx)
        interval = float(ctx.args.get("interval_hours") or snap.interval_hours)
        period = max(60.0, interval * 3600)
        count = 0
        try:
            while True:
                if snap.is_due():
                    snap.write(reason="scheduled")
                    count += 1
                time.sleep(min(period, 3600))  # re-check at most hourly
        except KeyboardInterrupt:
            return SkillResult(ok=True, skill=self.name,
                               summary=f"watch stopped after {count} snapshot(s)",
                               data={"snapshots": count})

    # -- heal --------------------------------------------------------------

    def _heal(self, ctx: SkillContext) -> SkillResult:
        from ptz_node.self_healing.healer import Healer, Incident

        healer = Healer(ctx.config)
        a = ctx.args
        if a.get("message") or a.get("error_type"):
            incident = Incident(
                component=str(a.get("component", "manual")),
                error_type=str(a.get("error_type", "Anomaly")),
                message=str(a.get("message", "")),
                traceback=str(a.get("traceback", "")),
                context=a.get("context") or {},
            )
        else:
            latest = self._latest_error(ctx)
            if latest is None:
                return SkillResult(ok=False, skill=self.name,
                                   summary="no incident provided and no recent error "
                                           "run found; pass message/error_type/traceback")
            incident = Incident(**latest)
        record = healer.handle(incident)
        return SkillResult(
            ok=record.get("status") not in ("apply_failed", "diagnosis_failed"),
            skill=self.name,
            summary=f"heal {record.get('id')} → {record.get('status')} (mode={healer.mode})",
            data={"status": record.get("status"), "id": record.get("id"),
                  "proposed_changes": record.get("proposed_changes")},
        )

    def _latest_error(self, ctx: SkillContext) -> dict[str, Any] | None:
        """Reconstruct an incident from the most recent failed run trace."""
        import json

        from ptz_node.paths import debug_dir

        ptr = debug_dir() / "latest_run.json"
        if not ptr.is_file():
            return None
        try:
            meta = json.loads(ptr.read_text(encoding="utf-8"))
            if meta.get("exit_code", 0) == 0:
                return None
            trace_path = meta.get("trace_json")
            errors = ""
            if trace_path:
                from pathlib import Path

                tp = Path(trace_path)
                if tp.is_file():
                    errors = "\n".join(json.loads(tp.read_text()).get("errors", []))
            return {"component": "agent_run", "error_type": "RunFailure",
                    "message": (errors.splitlines() or ["run failed"])[-1][:300],
                    "traceback": errors, "context": {"run_id": meta.get("run_id")}}
        except Exception:
            return None

    # -- status ------------------------------------------------------------

    def _status(self, ctx: SkillContext) -> SkillResult:
        snap = self._snapshotter(ctx)
        from ptz_node.self_healing.healer import Healer

        files = snap._snapshot_files()
        total_mb = sum(f.stat().st_size for f in files if f.exists()) / (1024 * 1024)
        pending = Healer(ctx.config).list_pending()
        return SkillResult(
            ok=True, skill=self.name,
            summary=f"{len(files)} snapshot(s), {total_mb:.2f} MB used; "
                    f"{len(pending)} heal(s) awaiting approval",
            data={
                "snapshots": len(files),
                "snapshots_mb": round(total_mb, 3),
                "interval_hours": snap.interval_hours,
                "is_due": snap.is_due(),
                "pending_heals": pending,
            },
        )
