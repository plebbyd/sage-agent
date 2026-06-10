"""LLM-driven diagnosis and self-healing.

On an incident (crash / unexpected output / edge case) the healer:
  1. gathers bounded context — traceback, the source files named in it, the
     latest snapshot's doctor summary, recent run pointer;
  2. asks a *strong* API LLM for a structured diagnosis + full-file fix;
  3. acts per ``self_healing.mode``:
        off          → record only
        notify_only  → record + notify, no code change
        approval     → record + save a pending proposal + notify, wait for a human
                       to run ``ptz-node heal --approve <id>``
        auto         → apply the patch (with backups) + notify

All file writes are constrained to the repo root, with timestamped backups so any
applied change is reversible via ``heal --revert <id>``.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import traceback as _tb
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ptz_node.paths import (
    heal_backups_dir,
    heal_dir,
    heal_pending_dir,
    repo_root,
    write_json,
)
from ptz_node.self_healing.notify import Notifier

_VALID_MODES = {"off", "notify_only", "approval", "auto"}
_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')
_MAX_FILE_BYTES = 60_000


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class Incident:
    component: str
    error_type: str
    message: str
    traceback: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    detected_at: str = field(default_factory=_utc)

    @classmethod
    def from_exception(cls, exc: BaseException, *, component: str,
                       context: dict[str, Any] | None = None) -> "Incident":
        return cls(
            component=component,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback="".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
            context=context or {},
        )


def _in_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root().resolve())
        return True
    except ValueError:
        return False


def _is_editable(rel_or_abs: str) -> tuple[bool, Path, str]:
    """Resolve a proposed change path and confirm it is safe to write."""
    root = repo_root().resolve()
    p = Path(rel_or_abs)
    target = (p if p.is_absolute() else root / p).resolve()
    if not _in_repo(target):
        return False, target, "outside repo root"
    parts = set(target.relative_to(root).parts)
    if parts & {".git", ".venv", "venv", ".local", "__pycache__"}:
        return False, target, "protected directory"
    if target.suffix not in {".py", ".yaml", ".yml", ".txt", ".md", ".toml", ".cfg",
                             ".json", ".sh", ".example"}:
        return False, target, f"refusing to edit {target.suffix or 'extensionless'} file"
    return True, target, ""


class Healer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.sh = self.config.get("self_healing") or {}
        mode = str(self.sh.get("mode", "approval")).lower()
        self.mode = mode if mode in _VALID_MODES else "approval"
        self.notifier = Notifier(self.config)

    def enabled(self) -> bool:
        return self.mode != "off"

    # ------------------------------------------------------------------ #
    # public entry points
    # ------------------------------------------------------------------ #

    def handle_exception(self, exc: BaseException, *, component: str,
                         context: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.handle(Incident.from_exception(exc, component=component,
                                                   context=context))

    def report_anomaly(self, component: str, message: str,
                       context: dict[str, Any] | None = None) -> dict[str, Any]:
        """For unexpected/undefined output that isn't a raised exception."""
        return self.handle(Incident(component=component, error_type="Anomaly",
                                    message=message, context=context or {}))

    def handle(self, incident: Incident) -> dict[str, Any]:
        record = {
            "id": f"heal_{_stamp()}",
            "mode": self.mode,
            "incident": asdict(incident),
            "created_at": _utc(),
        }
        if not self.enabled():
            record["status"] = "disabled"
            self._write_record(record)
            return record

        gathered = self._gather_context(incident)
        record["gathered"] = {k: v for k, v in gathered.items() if k != "file_sources"}

        diagnosis = self._diagnose(incident, gathered)
        record["diagnosis"] = diagnosis

        if not diagnosis.get("ok"):
            record["status"] = "diagnosis_failed"
            self._notify_incident(record, applied=False)
            self._write_record(record)
            return record

        proposal = diagnosis.get("proposal") or {}
        changes = self._sanitize_changes(proposal.get("changes") or [])
        record["proposed_changes"] = [
            {k: v for k, v in c.items() if k != "new_content"} for c in changes
        ]

        if self.mode == "notify_only" or not changes:
            record["status"] = "reported"
            self._notify_incident(record, applied=False, proposal=proposal)
            self._write_record(record)
            return record

        if self.mode == "approval":
            record["status"] = "pending_approval"
            record["changes_full"] = changes
            self._save_pending(record)
            self._notify_incident(record, applied=False, proposal=proposal)
            self._write_record(record)
            return record

        # mode == auto
        applied = self.apply_changes(record["id"], changes)
        record["status"] = "applied" if applied.get("ok") else "apply_failed"
        record["apply_result"] = applied
        self._notify_incident(record, applied=applied.get("ok", False),
                              proposal=proposal)
        self._write_record(record)
        return record

    # ------------------------------------------------------------------ #
    # context gathering
    # ------------------------------------------------------------------ #

    def _gather_context(self, incident: Incident) -> dict[str, Any]:
        files: dict[str, str] = {}
        seen: set[str] = set()
        for m in _FILE_RE.finditer(incident.traceback or ""):
            fpath = m.group(1)
            if fpath in seen:
                continue
            seen.add(fpath)
            p = Path(fpath)
            if _in_repo(p) and p.is_file():
                try:
                    txt = p.read_text(encoding="utf-8")
                    files[str(p.resolve().relative_to(repo_root().resolve()))] = (
                        txt[:_MAX_FILE_BYTES]
                    )
                except Exception:
                    pass
            if len(files) >= 6:
                break

        doctor: dict[str, Any] = {}
        try:
            from ptz_node.self_healing.snapshot import SystemSnapshotter

            snap = SystemSnapshotter(self.config).latest() or {}
            doctor = snap.get("doctor", {})
        except Exception:
            pass

        return {"file_sources": files,
                "edited_files": sorted(files),
                "doctor": doctor}

    # ------------------------------------------------------------------ #
    # LLM diagnosis
    # ------------------------------------------------------------------ #

    def _healer_model_section(self) -> dict[str, Any]:
        section = dict(self.sh.get("model") or {})
        if section.get("provider"):
            return section
        # fall back to the agent's resolved model
        from ptz_node.config_loader import resolve_model_config

        return resolve_model_config(self.config)

    def _diagnose(self, incident: Incident, gathered: dict[str, Any]) -> dict[str, Any]:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from ptz_node.llm_factory import chat_model_from_config

            llm = chat_model_from_config(self._healer_model_section())
        except Exception as exc:
            return {"ok": False, "error": f"healer LLM unavailable: {exc}"}

        sys_prompt = (
            "You are an expert Python site-reliability engineer embedded in an edge "
            "AI agent (LangGraph + a sensor gateway) running on a Jetson node. You "
            "diagnose a runtime incident and propose a minimal, safe fix. "
            "Only edit files inside the repository. Prefer the smallest change that "
            "fixes the root cause without altering unrelated behavior. "
            "Return STRICT JSON only (no markdown), with this schema:\n"
            "{\n"
            '  "root_cause": str,\n'
            '  "explanation": str,\n'
            '  "confidence": float (0-1),\n'
            '  "risk": "low"|"medium"|"high",\n'
            '  "changes": [{"path": str, "action": "replace_file"|"create",\n'
            '               "new_content": str, "reason": str}],\n'
            '  "needs_human": bool\n'
            "}\n"
            "If you cannot safely fix it, return empty changes and needs_human=true."
        )
        src_blocks = "\n\n".join(
            f"### FILE: {path}\n```python\n{content}\n```"
            for path, content in (gathered.get("file_sources") or {}).items()
        ) or "(no in-repo source files identified from the traceback)"

        human = (
            f"COMPONENT: {incident.component}\n"
            f"ERROR: {incident.error_type}: {incident.message}\n\n"
            f"TRACEBACK:\n{(incident.traceback or '(none)')[:6000]}\n\n"
            f"EXTRA CONTEXT:\n{json.dumps(incident.context, default=str)[:2000]}\n\n"
            f"DOCTOR SUMMARY:\n{json.dumps(gathered.get('doctor', {}), default=str)[:2000]}\n\n"
            f"REPO SOURCE FILES (edit these by returning full replacement content):\n"
            f"{src_blocks}"
        )

        try:
            resp = llm.invoke([SystemMessage(content=sys_prompt),
                               HumanMessage(content=human)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            proposal = _parse_json(text)
            if proposal is None:
                return {"ok": False, "error": "LLM returned non-JSON",
                        "raw": text[:1500]}
            return {"ok": True, "proposal": proposal}
        except Exception as exc:
            return {"ok": False, "error": f"LLM call failed: {exc}"}

    def _sanitize_changes(self, changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean: list[dict[str, Any]] = []
        for ch in changes:
            if not isinstance(ch, dict):
                continue
            path = str(ch.get("path", "")).strip()
            content = ch.get("new_content")
            if not path or content is None:
                continue
            ok, target, why = _is_editable(path)
            if not ok:
                clean.append({"path": path, "action": "rejected", "reason": why,
                              "new_content": ""})
                continue
            clean.append({
                "path": str(target.relative_to(repo_root().resolve())),
                "abs_path": str(target),
                "action": str(ch.get("action", "replace_file")),
                "reason": str(ch.get("reason", "")),
                "new_content": str(content),
            })
        return clean

    # ------------------------------------------------------------------ #
    # apply / approve / revert
    # ------------------------------------------------------------------ #

    def apply_changes(self, heal_id: str,
                      changes: list[dict[str, Any]]) -> dict[str, Any]:
        applicable = [c for c in changes if c.get("action") in ("replace_file", "create")
                      and c.get("abs_path")]
        if not applicable:
            return {"ok": False, "error": "no applicable changes"}
        backup_root = heal_backups_dir() / heal_id
        results = []
        for ch in applicable:
            target = Path(ch["abs_path"])
            ok, _, why = _is_editable(target)
            if not ok:
                results.append({"path": ch["path"], "ok": False, "error": why})
                continue
            try:
                if target.exists():
                    rel = target.resolve().relative_to(repo_root().resolve())
                    bpath = backup_root / rel
                    bpath.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, bpath)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(ch["new_content"], encoding="utf-8")
                results.append({"path": ch["path"], "ok": True})
            except Exception as exc:
                results.append({"path": ch["path"], "ok": False, "error": str(exc)})
        ok = all(r["ok"] for r in results)
        return {"ok": ok, "backup_dir": str(backup_root), "results": results}

    def approve(self, heal_id: str) -> dict[str, Any]:
        pending = heal_pending_dir() / f"{heal_id}.json"
        if not pending.is_file():
            return {"ok": False, "error": f"no pending proposal {heal_id!r}"}
        record = json.loads(pending.read_text(encoding="utf-8"))
        changes = record.get("changes_full") or []
        applied = self.apply_changes(heal_id, changes)
        record["status"] = "applied" if applied.get("ok") else "apply_failed"
        record["apply_result"] = applied
        record["approved_at"] = _utc()
        self._write_record(record)
        pending.unlink(missing_ok=True)
        self.notifier.send(
            f"[self-heal] applied {heal_id}" if applied.get("ok")
            else f"[self-heal] apply FAILED {heal_id}",
            json.dumps(applied, indent=2)[:1500],
        )
        return {"ok": applied.get("ok"), "record": record}

    def reject(self, heal_id: str) -> dict[str, Any]:
        pending = heal_pending_dir() / f"{heal_id}.json"
        if not pending.is_file():
            return {"ok": False, "error": f"no pending proposal {heal_id!r}"}
        record = json.loads(pending.read_text(encoding="utf-8"))
        record["status"] = "rejected"
        record["rejected_at"] = _utc()
        self._write_record(record)
        pending.unlink(missing_ok=True)
        return {"ok": True, "id": heal_id, "status": "rejected"}

    def revert(self, heal_id: str) -> dict[str, Any]:
        backup_root = heal_backups_dir() / heal_id
        if not backup_root.is_dir():
            return {"ok": False, "error": f"no backups for {heal_id!r}"}
        restored = []
        for bfile in backup_root.rglob("*"):
            if bfile.is_file():
                rel = bfile.relative_to(backup_root)
                target = repo_root() / rel
                ok, _, why = _is_editable(target)
                if not ok:
                    restored.append({"path": str(rel), "ok": False, "error": why})
                    continue
                shutil.copy2(bfile, target)
                restored.append({"path": str(rel), "ok": True})
        return {"ok": True, "restored": restored}

    def list_pending(self) -> list[dict[str, Any]]:
        out = []
        for f in sorted(heal_pending_dir().glob("heal_*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                out.append({
                    "id": rec.get("id"),
                    "status": rec.get("status"),
                    "component": rec.get("incident", {}).get("component"),
                    "error": rec.get("incident", {}).get("error_type"),
                    "files": [c.get("path") for c in rec.get("changes_full", [])],
                    "confidence": rec.get("diagnosis", {}).get("proposal", {}).get("confidence"),
                })
            except Exception:
                continue
        return out

    def show(self, heal_id: str) -> dict[str, Any]:
        for d in (heal_pending_dir(), heal_dir() / "records"):
            f = d / f"{heal_id}.json"
            if f.is_file():
                return json.loads(f.read_text(encoding="utf-8"))
        return {"ok": False, "error": f"unknown heal id {heal_id!r}"}

    # ------------------------------------------------------------------ #
    # persistence + notify
    # ------------------------------------------------------------------ #

    def _save_pending(self, record: dict[str, Any]) -> None:
        write_json(heal_pending_dir() / f"{record['id']}.json", record)

    def _write_record(self, record: dict[str, Any]) -> None:
        write_json(heal_dir() / "records" / f"{record['id']}.json", record)
        write_json(heal_dir() / "latest.json", record)

    def _notify_incident(self, record: dict[str, Any], *, applied: bool,
                         proposal: dict[str, Any] | None = None) -> None:
        if not self.notifier.enabled():
            return
        inc = record["incident"]
        prop = proposal or {}
        status = record.get("status")
        subject = f"[self-heal:{status}] {inc['error_type']} in {inc['component']}"
        lines = [
            f"node incident — mode={self.mode}, id={record['id']}",
            f"error: {inc['error_type']}: {inc['message']}",
        ]
        if prop:
            lines += [
                f"root cause: {prop.get('root_cause', '?')}",
                f"confidence: {prop.get('confidence')}  risk: {prop.get('risk')}",
                f"files: {[c.get('path') for c in record.get('proposed_changes', [])]}",
            ]
        if record.get("status") == "pending_approval":
            lines.append(f"APPROVE: ptz-node heal --approve {record['id']}")
            lines.append(f"REJECT : ptz-node heal --reject {record['id']}")
        if applied:
            lines.append("a fix was applied automatically; review and revert if needed:")
            lines.append(f"REVERT : ptz-node heal --revert {record['id']}")
        self.notifier.send(subject, "\n".join(lines))


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None
