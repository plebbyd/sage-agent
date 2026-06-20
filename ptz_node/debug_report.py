"""Preflight checks, run traces, and Cursor-readable debug summaries."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ptz_node.paths import debug_dir, local_data_root, runs_dir, stamp, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_import(label: str, module: str) -> dict[str, Any]:
    try:
        __import__(module)
        return {"name": label, "ok": True}
    except Exception as exc:
        return {"name": label, "ok": False, "error": str(exc)}


def _http_ok(url: str, timeout: float = 3.0, api_key: str = "") -> dict[str, Any]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"url": url, "ok": True, "status": resp.status}
    except urllib.error.HTTPError as exc:
        # 401/403 still means the proxy is reachable
        return {
            "url": url,
            "ok": exc.code < 500,
            "status": exc.code,
            "error": str(exc),
        }
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc)}


def _ollama_models(tags_url: str, timeout: float = 4.0) -> list[str]:
    """Return the list of model names installed in Ollama (``/api/tags``)."""
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))
    except Exception:
        return []


def _model_present(want: str, installed: list[str]) -> bool:
    """True if the configured model is pulled. Ollama defaults a bare name to
    ``:latest``, so ``gemma4`` matches an installed ``gemma4:latest``."""
    if not want:
        return False
    want_full = want if ":" in want else f"{want}:latest"
    names = set(installed)
    return want in names or want_full in names


@dataclass
class RunRecorder:
    prompt: str
    config_path: str | None = None
    run_id: str = field(default_factory=lambda: f"run_{stamp()}")
    started_at: str = field(default_factory=_utc_now)
    trace: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    exit_code: int = 0
    final_reply: str = ""
    duration_s: float = 0.0

    @property
    def run_dir(self) -> Path:
        return runs_dir() / self.run_id

    def add_trace(self, entry: dict[str, Any]) -> None:
        self.trace.append(entry)

    def finish_success(self, messages: list, *, duration_s: float) -> Path:
        from ptz_node.graph_runner import extract_trace, summarize_messages

        self.duration_s = duration_s
        self.trace = extract_trace(messages)
        self.final_reply = summarize_messages(messages)
        self.exit_code = 0
        return self.persist()

    def finish_error(self, exc: BaseException, *, duration_s: float) -> Path:
        self.duration_s = duration_s
        self.exit_code = 1
        self.errors.append(str(exc))
        self.errors.append(traceback.format_exc())
        return self.persist()

    def persist(self) -> Path:
        d = self.run_dir
        d.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "prompt": self.prompt,
            "config_path": self.config_path,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "final_reply_preview": (self.final_reply or "")[:800],
            "cursor_hint": (
                f"Read {d / 'summary.txt'} for a human trace; "
                f"{d / 'trace.json'} for structured tool calls; "
                f"{debug_dir() / 'latest_run.json'} for the newest run pointer."
            ),
        }
        write_json(d / "meta.json", meta)
        write_json(d / "trace.json", {"steps": self.trace, "errors": self.errors})

        summary_lines = [
            f"# Run {self.run_id}",
            f"exit_code: {self.exit_code}",
            f"duration_s: {self.duration_s:.2f}",
            "",
            "## Prompt",
            self.prompt,
            "",
            "## Final reply",
            self.final_reply or "(empty)",
            "",
            "## Tool trace",
        ]
        for i, step in enumerate(self.trace, 1):
            summary_lines.append(f"{i}. {json.dumps(step, default=str)[:1200]}")
        if self.errors:
            summary_lines.extend(["", "## Errors", *self.errors])
        (d / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

        pointer = {
            "run_id": self.run_id,
            "run_dir": str(d),
            "summary_txt": str(d / "summary.txt"),
            "trace_json": str(d / "trace.json"),
            "meta_json": str(d / "meta.json"),
            "exit_code": self.exit_code,
            "updated_at": _utc_now(),
        }
        write_json(debug_dir() / "latest_run.json", pointer)
        write_json(local_data_root() / "latest_run.json", pointer)
        return d


def _probe_detector_details() -> dict[str, dict[str, Any]]:
    """Explain why each PTZ vision backend is or is not available."""
    details: dict[str, dict[str, Any]] = {}

    try:
        from ultralytics import YOLO  # noqa: F401

        details["yolo"] = {"ok": True}
    except ImportError as exc:
        details["yolo"] = {
            "ok": False,
            "error": str(exc),
            "fix": "pip install -r requirements-vision.txt  (or: pip install ultralytics)",
        }

    bio_missing: list[str] = []
    for mod, label in (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
    ):
        try:
            __import__(mod)
        except ImportError:
            bio_missing.append(label)
    if not bio_missing:
        try:
            import importlib.util

            if importlib.util.find_spec("open_clip") is None:
                bio_missing.append("open_clip_torch")
        except Exception as exc:
            bio_missing.append(f"open_clip ({exc})")
    if bio_missing:
        details["bioclip"] = {
            "ok": False,
            "missing": bio_missing,
            "fix": "pip install -r requirements-vision.txt",
        }
    else:
        details["bioclip"] = {"ok": True}

    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    probe = _http_ok(f"{ollama_url}/api/tags")
    if probe.get("ok"):
        details["gemma4"] = {"ok": True, "ollama": ollama_url}
    else:
        details["gemma4"] = {
            "ok": False,
            "error": probe.get("error", "Ollama not reachable"),
            "fix": (
                "Start Ollama (macOS: open Ollama app, or `ollama serve`), then "
                "`ollama pull gemma4:31b` for the agent (local reasoning) and "
                "`ollama pull gemma4:e2b` for vision detect/caption."
            ),
        }

    return details


def run_doctor(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    from ptz_node.config_loader import load_config, resolve_model_config

    cfg = cfg or load_config()
    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "venv": os.environ.get("VIRTUAL_ENV", ""),
        "checks": [],
        "detectors": {},
        "detector_details": {},
        "devices": [],
        "issues": [],
        "warnings": [],
        "gateway_ok": True,
        "vision_ok": True,
        "agent_ok": True,
        "ok": True,
        "cursor_hint": (
            f"Open {debug_dir() / 'doctor.json'} after `python -m ptz_node doctor`. "
            f"Fix checks with ok=false; gateway-smoke works without Ollama."
        ),
    }

    def add(name: str, ok: bool, *, critical: str = "gateway", **extra: Any) -> None:
        row = {"name": name, "ok": ok, "critical": critical, **extra}
        report["checks"].append(row)
        if not ok:
            msg = extra.get("error") or extra.get("fix") or extra.get("detail") or name
            if critical == "gateway":
                report["gateway_ok"] = False
                report["ok"] = False
                report["issues"].append(f"{name}: {msg}")
            elif critical == "agent":
                report["agent_ok"] = False
                report["ok"] = False
                report["issues"].append(f"{name}: {msg}")
            elif critical == "vision":
                report["vision_ok"] = False
                report["warnings"].append(f"{name}: {msg}")
            else:
                report["warnings"].append(f"{name}: {msg}")

    for pkg in (
        _check_import("import_langgraph", "langgraph"),
        _check_import("import_fastapi", "fastapi"),
        _check_import("import_pillow", "PIL"),
    ):
        add(pkg["name"], pkg["ok"], **{k: v for k, v in pkg.items() if k != "name" and k != "ok"})

    ptz_root: Path | None = None
    try:
        from ptz_node.bootstrap import bootstrap_ptz_agent_runtime

        ptz_root = bootstrap_ptz_agent_runtime(cfg.get("ptz_agent_root"))
        add("ptz_agent_root", True, critical="gateway", path=str(ptz_root))
    except Exception as exc:
        add("ptz_agent_root", False, critical="gateway", error=str(exc))

    panorama = ptz_root / "stitched.png" if ptz_root else None
    if panorama and panorama.is_file():
        add("sim_panorama", True, critical="gateway", path=str(panorama))
    else:
        add(
            "sim_panorama",
            False,
            critical="gateway",
            error="stitched.png missing in ptz-agent root (required for sim PTZ backend)",
            expected=str(panorama or "<unknown>"),
            fix="Copy a 360° panorama to ptz-agent/stitched.png or set MSA_PTZ_BACKEND=reolink",
        )

    model_cfg = resolve_model_config(cfg)
    provider = model_cfg.get("provider", "ollama")
    report["llm"] = {
        "provider": provider,
        "model": model_cfg.get("model"),
        "base_url": model_cfg.get("base_url"),
    }
    ap = cfg.get("argo_proxy") or {}
    if ap.get("enabled"):
        report["llm"]["argo_jump"] = ap.get("ssh_jump_host") or None
        report["llm"]["argo_port"] = ap.get("port")

    if provider == "ollama":
        base = (model_cfg.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        probe = _http_ok(f"{base}/api/tags")
        add("ollama_reachable", probe["ok"], critical="agent",
            **{k: v for k, v in probe.items() if k != "ok"})
        if not probe["ok"]:
            report["issues"].append(
                "Agent LLM needs Ollama: open the Ollama app (macOS) or run `ollama serve`, "
                "then `ollama pull gemma4:31b`."
            )
        else:
            # Reachable is not enough — the configured model must actually be pulled,
            # else the first `run` dies with a 404 (seen on a fresh Orin node).
            want = str(model_cfg.get("model") or "").strip()
            installed = _ollama_models(f"{base}/api/tags")
            present = _model_present(want, installed)
            extra = {"model": want, "installed": installed[:20]}
            if present:
                extra["detail"] = f"{want} present in ollama"
            else:
                extra["fix"] = f"ollama pull {want}" if want else "set model.model in config"
            add("ollama_model_pulled", present, critical="agent", **extra)
            if not present:
                report["issues"].append(
                    f"Configured Ollama model {want!r} is not pulled "
                    f"(have: {installed or 'none'}). Run: ollama pull {want}"
                )
    elif provider == "argo_proxy":
        base_v1 = (model_cfg.get("base_url") or "").rstrip("/")
        base = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
        jump = (ap.get("ssh_jump_host") or "").strip()
        api_key = (
            model_cfg.get("api_key")
            or ap.get("username")
            or os.environ.get("ARGO_PROXY_API_KEY")
            or ""
        ).strip()

        if not api_key:
            add("argo_proxy_username", False, critical="agent",
                fix="bash scripts/setup_argo_proxy.sh -u YOUR_ANL_USER --jump node-V010")
        else:
            add("argo_proxy_username", True, critical="agent")

        if base:
            health = _http_ok(f"{base}/health")
            add("argo_proxy_health", health["ok"], critical="agent",
                **{k: v for k, v in health.items() if k != "ok"})
            models_probe = _http_ok(f"{base_v1}/models", api_key=api_key, timeout=10.0)
            add("argo_proxy_models", models_probe["ok"], critical="agent",
                **{k: v for k, v in models_probe.items() if k != "ok"})
            if not health.get("ok"):
                if jump:
                    report["issues"].append(
                        f"argo-proxy not reachable at {base}. Steps: "
                        f"(1) `ssh {jump} hostname` "
                        f"(2) on {jump}: `bash scripts/setup_argo_proxy.sh -u USER -m MODEL` "
                        f"(3) on this Mac: `bash scripts/setup_argo_proxy.sh -u USER --jump {jump}`. "
                        f"See config/ssh_config.example"
                    )
                else:
                    report["issues"].append(
                        "Start argo-proxy on this ANL-network host: `argo-proxy serve` "
                        "(https://argo-proxy.readthedocs.io/en/latest/usage/running/)"
                    )
        else:
            health = {"ok": False}
            add("argo_proxy_health", False, critical="agent",
                error="argo_proxy enabled but base_url missing")

    if ptz_root:
        details = _probe_detector_details()
        report["detector_details"] = details
        report["detectors"] = {k: v.get("ok", False) for k, v in details.items()}
        for model, info in details.items():
            add(
                f"detector_{model}",
                bool(info.get("ok")),
                critical="vision",
                model=model,
                **{k: v for k, v in info.items() if k not in ("ok",)},
            )

        try:
            from ptz_node.sensor_gateway import SensorGateway

            devices = SensorGateway(cfg).list_devices()
            report["devices"] = devices
            kinds: dict[str, int] = {}
            for d in devices:
                kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
            report["device_kinds"] = kinds
            n_sensors = kinds.get("sensor", 0)
            add("gateway_devices", True, critical="gateway",
                count=len(devices), kinds=kinds,
                detail=f"{len(devices)} device(s): {kinds}")
            # Auto-discovered sensors are a platform feature, not a hard requirement.
            add("gateway_sensors", n_sensors > 0, critical="info",
                count=n_sensors,
                detail=(f"{n_sensors} ptz-agent sensor plugin(s) discovered"
                        if n_sensors else
                        "no sensor plugins found in ptz-agent/sensors/ (optional)"))
        except Exception as exc:
            add("gateway_devices", False, critical="gateway", error=str(exc))

    # Skills + self-healing visibility (informational).
    try:
        from ptz_node.skills import SkillRegistry

        skills = SkillRegistry(cfg).list_skills()
        report["skills"] = [s["name"] for s in skills]
        add("skills_discovered", len(skills) > 0, critical="info",
            count=len(skills), detail=f"{[s['name'] for s in skills]}")
    except Exception as exc:
        add("skills_discovered", False, critical="info", error=str(exc))

    sh = cfg.get("self_healing") or {}
    report["self_healing"] = {
        "enabled": bool(sh.get("enabled")),
        "mode": sh.get("mode"),
        "snapshot_interval_hours": (sh.get("snapshot") or {}).get("interval_hours"),
        "notify_channels": (sh.get("notify") or {}).get("channels"),
    }
    if sh.get("enabled"):
        healer_section = sh.get("model") or {}
        healer_provider = (healer_section.get("provider") or "agent-model").lower()
        _key_env = {
            "anthropic": ("ANTHROPIC_API_KEY",),
            "openrouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
            "openai": ("OPENAI_API_KEY",),
        }
        needed = _key_env.get(healer_provider, (
            "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"))
        has_key = bool(healer_section.get("api_key")
                       or any(os.environ.get(k) for k in needed))
        healer_model = healer_section.get("model") or "(agent model)"
        add("self_healing_llm_key", has_key, critical="info",
            detail=f"healer provider={healer_provider} model={healer_model}; "
                   + ("API key present" if has_key
                      else f"set one of {', '.join(needed)}"))
        chans = (sh.get("notify") or {}).get("channels") or []
        add("self_healing_notify", bool(chans), critical="info",
            detail=(f"channels={chans}; set SLACK_WEBHOOK_URL / SMTP_* env"
                    if chans else "no notify channels configured (optional)"))

    write_json(debug_dir() / "doctor.json", report)
    write_json(local_data_root() / "status.json", {
        "updated_at": _utc_now(),
        "doctor_ok": report["ok"],
        "gateway_ok": report["gateway_ok"],
        "agent_ok": report["agent_ok"],
        "vision_ok": report["vision_ok"],
        "detectors": report.get("detectors", {}),
        "issues": report.get("issues", []),
        "warnings": report.get("warnings", []),
        "paths": {
            "local_data": str(local_data_root()),
            "doctor_json": str(debug_dir() / "doctor.json"),
            "latest_run_json": str(debug_dir() / "latest_run.json"),
        },
        "cursor_hint": report["cursor_hint"],
    })
    return report


def format_doctor_text(report: dict[str, Any]) -> str:
    lines = [
        f"doctor ok={report.get('ok')}  gateway_ok={report.get('gateway_ok')}  "
        f"agent_ok={report.get('agent_ok')}  vision_ok={report.get('vision_ok')}  "
        f"platform={report.get('platform')}",
        "",
    ]
    if report.get("venv"):
        lines.append(f"  venv: {report['venv']}")
        lines.append("")
    for chk in report.get("checks", []):
        mark = "OK" if chk.get("ok") else "FAIL"
        detail = (
            chk.get("fix")
            or chk.get("error")
            or chk.get("path")
            or chk.get("detail")
            or ""
        )
        crit = chk.get("critical", "")
        lines.append(f"  [{mark}] {chk.get('name')} ({crit}) {detail}".rstrip())
    if report.get("llm"):
        lines.extend(["", "llm:", json.dumps(report["llm"], indent=2)])
    if report.get("detectors"):
        lines.extend(["", "detectors:", json.dumps(report["detectors"], indent=2)])
    if report.get("issues"):
        lines.extend(["", "issues (block agent runs):"])
        lines.extend(f"  - {i}" for i in report["issues"])
    if report.get("warnings"):
        lines.extend(["", "warnings (vision optional / gateway-smoke still works):"])
        lines.extend(f"  - {w}" for w in report["warnings"])
    lines.extend([
        "",
        "Quick fixes:",
        "  ptz-agent:   self-contained copy → bash scripts/vendor_ptz_agent.sh "
        "(or set PTZ_AGENT_ROOT=/path/to/ptz-agent)",
        "  No LLM:      python -m ptz_node gateway-smoke",
        "  Vision deps: bash scripts/local_dev.sh   (GPU PyTorch for YOLO/BioCLIP)",
        "  argo-proxy:  bash scripts/setup_argo_proxy.sh -u USER -m gpt-4o --jump node-V010",
        "  OpenRouter:  export OPENROUTER_API_KEY=... and set model.provider=openrouter",
        "",
        f"Wrote {debug_dir() / 'doctor.json'}",
    ])
    return "\n".join(lines)
