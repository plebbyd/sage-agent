"""Command-line entry for LangGraph + sensor-gateway runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ptz-node",
        description="LangGraph agent with PTZ/sensor gateway (hardware isolated).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    doc = sub.add_parser("doctor", help="preflight checks → .local/debug/doctor.json")
    doc.add_argument("--config", type=Path, default=None)
    doc.add_argument("--json", action="store_true")

    st = sub.add_parser("status", help="quick status from .local/status.json")
    st.add_argument("--config", type=Path, default=None)

    r = sub.add_parser("run", help="single user prompt → agent graph invoke")
    r.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path (else PTZ_GRAPH_CONFIG or config/default.yaml)",
    )
    r.add_argument("prompt", nargs=argparse.REMAINDER,
                   help='task text — use `"`quotes for multi-word')
    r.add_argument(
        "--limit",
        type=int,
        default=64,
        help="LangGraph recursion_limit (tool+model cycles)",
    )
    r.add_argument("--json-out", action="store_true",
                   help="print full result dict JSON instead of final reply")
    r.add_argument("--no-trace", action="store_true",
                   help="skip writing .local/runs trace artifacts")

    d = sub.add_parser("devices", help="print managed device catalog (all drivers)")
    d.add_argument("--config", type=Path, default=None)

    rd = sub.add_parser("read", help="read a sensor device (e.g. sensor:system_stats)")
    rd.add_argument("device_id")
    rd.add_argument("--config", type=Path, default=None)

    iv = sub.add_parser("invoke", help="invoke any device capability")
    iv.add_argument("device_id")
    iv.add_argument("capability")
    iv.add_argument("--params", default="{}", help='JSON object, e.g. \'{"pan": 30}\'')
    iv.add_argument("--config", type=Path, default=None)

    g = sub.add_parser(
        "gateway-smoke",
        help="exercise SensorGateway without LLM → .local/debug/gateway_smoke.json",
    )
    g.add_argument("--config", type=Path, default=None)

    t = sub.add_parser("test", help="run Sage-style agentic test cases")
    t.add_argument("--config", type=Path, default=None)
    t.add_argument("--list", action="store_true", help="list case ids")
    t.add_argument("--id", dest="case_id", default=None, help="run one case by id")
    t.add_argument("--all", action="store_true", help="run all cases")
    t.add_argument("--limit", type=int, default=80)

    sk = sub.add_parser("skill", help="list/run modular skills")
    sk.add_argument("--config", type=Path, default=None)
    sk_sub = sk.add_subparsers(dest="skill_cmd", required=True)
    sk_sub.add_parser("list", help="list discovered skills")
    sk_run = sk_sub.add_parser("run", help="run a skill once")
    sk_run.add_argument("name")
    sk_run.add_argument("--args", default="{}", help="JSON object of skill args")

    sn = sub.add_parser("snapshot", help="capture a system snapshot (self-healing)")
    sn.add_argument("--config", type=Path, default=None)
    sn.add_argument("--force", action="store_true", help="snapshot even if not due")
    sn.add_argument("--status", action="store_true", help="show snapshot/heal status")

    hl = sub.add_parser("heal", help="self-healing: diagnose / approve / revert fixes")
    hl.add_argument("--config", type=Path, default=None)
    hl.add_argument("--list", action="store_true", help="list proposals awaiting approval")
    hl.add_argument("--show", metavar="ID", default=None, help="print one proposal")
    hl.add_argument("--approve", metavar="ID", default=None, help="apply a pending fix")
    hl.add_argument("--reject", metavar="ID", default=None, help="discard a pending fix")
    hl.add_argument("--revert", metavar="ID", default=None, help="restore backups of a fix")
    hl.add_argument("--diagnose", action="store_true",
                    help="diagnose+heal the latest failed run")
    hl.add_argument("--message", default=None, help="diagnose an ad-hoc anomaly message")
    hl.add_argument("--component", default="manual")

    ap = sub.add_parser("argo", help="argo-proxy setup/test (wraps scripts/setup_argo_proxy.sh)")
    ap_sub = ap.add_subparsers(dest="argo_cmd", required=True)
    ap_setup = ap_sub.add_parser("setup", help="configure + tunnel/start proxy")
    ap_setup.add_argument("--username", "-u")
    ap_setup.add_argument("--model", "-m", default="")
    ap_setup.add_argument("--jump", "-j", dest="jump_host", default=None)
    ap_setup.add_argument("--port", "-p", type=int, default=0)
    ap_setup.add_argument("--skip-test", action="store_true")
    ap_sub.add_parser("test", help="test existing argo-proxy config")
    ap_sub.add_parser("disable", help="disable argo-proxy; fall back to ollama in config")

    argv = argv if argv is not None else sys.argv[1:]
    args = p.parse_args(argv)

    _apply_config_env(args)
    from ptz_node.config_loader import load_config
    from ptz_node.graph_runner import build_agent_executor, summarize_messages

    cfg = load_config(getattr(args, "config", None))

    if args.cmd == "doctor":
        from ptz_node.debug_report import format_doctor_text, run_doctor

        report = run_doctor(cfg)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(format_doctor_text(report))
        return 0 if report.get("ok") else 1

    if args.cmd == "status":
        from ptz_node.debug_report import run_doctor
        from ptz_node.paths import local_data_root

        run_doctor(cfg)
        status_path = local_data_root() / "status.json"
        if status_path.is_file():
            print(status_path.read_text(encoding="utf-8"))
        else:
            print("{}")
        return 0

    if args.cmd == "devices":
        from ptz_node.sensor_gateway import SensorGateway

        g = SensorGateway(cfg)
        print(json.dumps(g.list_devices(), indent=2))
        return 0

    if args.cmd == "read":
        from ptz_node.sensor_gateway import SensorGateway

        out = json.loads(SensorGateway(cfg).read_sensor(args.device_id))
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") else 1

    if args.cmd == "invoke":
        from ptz_node.sensor_gateway import SensorGateway

        try:
            params = json.loads(args.params) if args.params.strip() else {}
        except json.JSONDecodeError as exc:
            print(f"bad --params JSON: {exc}", file=sys.stderr)
            return 2
        out = json.loads(SensorGateway(cfg).invoke(args.device_id, args.capability, params))
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") else 1

    if args.cmd == "gateway-smoke":
        from ptz_node.paths import debug_dir, write_json
        from ptz_node.sensor_gateway import SensorGateway

        gw = SensorGateway(cfg)
        devices = gw.list_devices()
        sensor_reads = {}
        for dev in devices:
            if dev.get("kind") == "sensor":
                sensor_reads[dev["id"]] = json.loads(gw.read_sensor(dev["id"]))
        payload = {
            "devices": devices,
            "device_kinds": _kinds(devices),
            "detectors": json.loads(gw.detector_status()),
            "position": json.loads(gw.ptz_get_position()),
            "snapshot": json.loads(gw.ptz_snapshot(filename="gateway_smoke.jpg")),
            "sensor_reads": sensor_reads,
            "cursor_hint": f"Gateway-only smoke test; no LLM required. See {debug_dir() / 'gateway_smoke.json'}",
        }
        path = write_json(debug_dir() / "gateway_smoke.json", payload)
        print(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {path}", file=sys.stderr)
        return 0

    if args.cmd == "skill":
        from ptz_node.skills import SkillRegistry

        reg = SkillRegistry(cfg)
        if args.skill_cmd == "list":
            print(json.dumps(reg.list_skills(), indent=2, default=str))
            return 0
        if args.skill_cmd == "run":
            try:
                skill_args = json.loads(args.args) if args.args.strip() else {}
            except json.JSONDecodeError as exc:
                print(f"bad --args JSON: {exc}", file=sys.stderr)
                return 2
            if not reg.has(args.name):
                print(f"unknown skill {args.name!r}; known: {reg.names()}",
                      file=sys.stderr)
                return 2
            result = reg.run(args.name, skill_args)
            print(json.dumps(result.as_dict(), indent=2, default=str))
            return 0 if result.ok else 1
        return 2

    if args.cmd == "snapshot":
        from ptz_node.skills import SkillRegistry

        reg = SkillRegistry(cfg)
        action = "status" if args.status else "snapshot"
        result = reg.run("self_diagnosis", {"action": action, "force": args.force})
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0 if result.ok else 1

    if args.cmd == "heal":
        from ptz_node.self_healing.healer import Healer

        healer = Healer(cfg)
        if args.list:
            print(json.dumps(healer.list_pending(), indent=2, default=str))
            return 0
        if args.show:
            print(json.dumps(healer.show(args.show), indent=2, default=str))
            return 0
        if args.approve:
            out = healer.approve(args.approve)
            print(json.dumps(out, indent=2, default=str))
            return 0 if out.get("ok") else 1
        if args.reject:
            out = healer.reject(args.reject)
            print(json.dumps(out, indent=2, default=str))
            return 0 if out.get("ok") else 1
        if args.revert:
            out = healer.revert(args.revert)
            print(json.dumps(out, indent=2, default=str))
            return 0 if out.get("ok") else 1
        if args.diagnose or args.message:
            from ptz_node.skills import SkillRegistry

            skill_args = {"action": "heal"}
            if args.message:
                skill_args.update(component=args.component, error_type="Anomaly",
                                  message=args.message)
            result = SkillRegistry(cfg).run("self_diagnosis", skill_args)
            print(json.dumps(result.as_dict(), indent=2, default=str))
            return 0 if result.ok else 1
        print("use --list/--show/--approve/--reject/--revert/--diagnose/--message",
              file=sys.stderr)
        return 2

    if args.cmd == "argo":
        script = Path(__file__).resolve().parents[1] / "scripts" / "setup_argo_proxy.sh"
        if args.argo_cmd == "setup":
            cmd = [str(script)]
            if args.username:
                cmd.extend(["--username", args.username])
            if args.model:
                cmd.extend(["--model", args.model])
            if args.jump_host is not None:
                cmd.extend(["--jump", args.jump_host])
            if args.port:
                cmd.extend(["--port", str(args.port)])
            if args.skip_test:
                cmd.append("--skip-test")
            return subprocess.call(cmd)
        if args.argo_cmd == "test":
            return subprocess.call([str(script), "test"])
        if args.argo_cmd == "disable":
            return subprocess.call([str(script), "disable"])
        return 2

    if args.cmd == "test":
        from ptz_node.test_runner import load_test_cases, run_all_tests, run_test_case

        if args.list:
            for c in load_test_cases():
                print(f"{c['id']:40}  {c.get('title', '')}")
            return 0
        if args.all:
            summary = run_all_tests(
                cfg=cfg,
                config_path=str(args.config) if args.config else None,
                limit=args.limit,
            )
            print(json.dumps(summary, indent=2, default=str))
            return 0 if summary.get("failed", 0) == 0 else 1
        if args.case_id:
            result = run_test_case(
                args.case_id,
                cfg=cfg,
                config_path=str(args.config) if args.config else None,
                limit=args.limit,
            )
            print(json.dumps(result, indent=2, default=str))
            if result.get("skipped"):
                return 2
            return 0 if result.get("passed") else 1
        print("pass --list, --id CASE, or --all", file=sys.stderr)
        return 2

    if args.cmd == "run":
        prompt_parts = list(args.prompt)
        prompt = (
            (" ".join(prompt_parts).strip())
            if prompt_parts
            else ""
        ).strip()
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            print("provide a PROMPT argument or stdin", file=sys.stderr)
            return 2

        from ptz_node.debug_report import RunRecorder

        _maybe_snapshot(cfg)
        graph, _gw = build_agent_executor(cfg)
        rec = RunRecorder(
            prompt=prompt,
            config_path=str(args.config) if args.config else os.environ.get("PTZ_GRAPH_CONFIG"),
        )
        t0 = time.time()
        try:
            out = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": int(args.limit)},
            )
            duration = time.time() - t0
            if not args.no_trace:
                rec.finish_success(out.get("messages") or [], duration_s=duration)
        except Exception as exc:
            duration = time.time() - t0
            if not args.no_trace:
                rec.finish_error(exc, duration_s=duration)
            print(f"agent error: {exc}", file=sys.stderr)
            if rec.run_dir.exists():
                print(f"trace: {rec.run_dir / 'summary.txt'}", file=sys.stderr)
            heal = _maybe_heal(cfg, exc, component="agent_run",
                               context={"prompt": prompt[:500]})
            if heal:
                print(f"self-heal: {heal.get('status')} (id={heal.get('id')})",
                      file=sys.stderr)
            return 1

        if args.json_out:
            try:
                from langchain_core.messages import BaseMessage

                def _ser(o):
                    if isinstance(o, BaseMessage):
                        return {
                            "type": o.__class__.__name__,
                            "content": o.content,
                        }
                    raise TypeError

                printable = {}
                for k, v in out.items():
                    if k == "messages":
                        printable[k] = [_ser(x) for x in v]
                    else:
                        printable[k] = repr(v)
                print(json.dumps(printable, indent=2, default=str))
            except Exception:
                print(json.dumps({"messages_repr": repr(out.get("messages"))}))
        else:
            print(summarize_messages(out.get("messages")))
            if not args.no_trace:
                print(f"\n# trace: {rec.run_dir / 'summary.txt'}", file=sys.stderr)
        return 0

    return 1


def _maybe_snapshot(cfg) -> None:
    """Snapshot system state before a run if self-healing is on and a snap is due."""
    sh = cfg.get("self_healing") or {}
    if not sh.get("enabled"):
        return
    try:
        from ptz_node.self_healing.snapshot import SystemSnapshotter

        snap = SystemSnapshotter(cfg)
        if snap.is_due():
            snap.write(reason="pre_run")
    except Exception:
        pass


def _maybe_heal(cfg, exc: BaseException, *, component: str,
                context: dict | None = None):
    sh = cfg.get("self_healing") or {}
    if not sh.get("enabled"):
        return None
    from ptz_node.self_healing.guard import trigger_heal

    return trigger_heal(exc, component=component, cfg=cfg, context=context)


def _kinds(devices: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in devices:
        k = d.get("kind", "?")
        out[k] = out.get(k, 0) + 1
    return out


def _apply_config_env(args) -> None:
    cfg_path = getattr(args, "config", None)
    if cfg_path is not None:
        os.environ.setdefault("PTZ_GRAPH_CONFIG", str(cfg_path.expanduser()))


if __name__ == "__main__":
    raise SystemExit(main())
