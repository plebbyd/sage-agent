#!/usr/bin/env python3
"""Configure, start, and test argo-proxy for jetson-ptz-agent-graph."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "argo_proxy.local.yaml"
ENV_FILE = Path(os.environ.get("MSA_ENV_FILE", str(Path.home() / ".msa.env")))
ARGO_PROXY_CFG = Path.home() / ".config" / "argoproxy" / "config.yaml"
MSA_STATE_DIR = Path.home() / ".msa"
PID_FILE = MSA_STATE_DIR / "argo-proxy.pid"
LOG_FILE = PROJECT_ROOT / "logs" / "argo_proxy.log"

sys.path.insert(0, str(PROJECT_ROOT))
from ptz_node.config_loader import load_config, resolve_model_config  # noqa: E402

DEFAULT_ARGO_PROXY = {
    "enabled": False,
    "username": "",
    "host": "127.0.0.1",
    "port": 44497,
    "model": "gpt-4o",
    "auto_start": True,
    "ssh_jump_host": "",
    "ssh_local_port": 44497,
    "argo_upstream": "https://apps.inside.anl.gov/argoapi",
}

ARGO_BASE_URLS = {
    "prod": "https://apps.inside.anl.gov/argoapi",
    "dev": "https://apps-dev.inside.anl.gov/argoapi",
    "test": "https://apps-test.inside.anl.gov/argoapi",
}


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_msa_config() -> dict:
    return load_config(CONFIG_PATH)


def save_local_config(cfg: dict) -> None:
    payload: dict = {"argo_proxy": cfg.get("argo_proxy") or {}}
    if cfg.get("argo_proxy", {}).get("enabled"):
        payload["model"] = resolve_model_config(cfg)
    _save_yaml(LOCAL_CONFIG_PATH, payload)
    try:
        LOCAL_CONFIG_PATH.chmod(0o600)
    except OSError:
        pass
    print(f"[argo] wrote {LOCAL_CONFIG_PATH}")


def sync_model_from_argo_proxy(cfg: dict, enable: bool | None = None) -> dict:
    ap = cfg.setdefault("argo_proxy", dict(DEFAULT_ARGO_PROXY))
    if enable is not None:
        ap["enabled"] = enable
    return cfg


def _http_get(url: str, api_key: str = "", timeout: float = 10.0) -> tuple[int, str]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 0, str(exc.reason)


def proxy_base_url(ap: dict) -> str:
    jump = (ap.get("ssh_jump_host") or "").strip()
    if jump:
        port = int(ap.get("ssh_local_port") or ap.get("port") or 44497)
        return f"http://127.0.0.1:{port}"
    host = ap.get("host") or "127.0.0.1"
    port = int(ap.get("port") or 44497)
    return f"http://{host}:{port}"


def is_proxy_up(ap: dict) -> bool:
    code, _ = _http_get(f"{proxy_base_url(ap)}/health", timeout=3.0)
    return code == 200


def ensure_argo_proxy_installed() -> None:
    if shutil.which("argo-proxy"):
        return
    py = sys.executable
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"argo-proxy requires Python 3.10+; this interpreter is "
            f"{sys.version.split()[0]} ({py}). "
            f"Re-run via: python3.11 {__file__} setup …"
        )
    print(f"[argo] argo-proxy not on PATH — installing with {py} …")
    subprocess.run([py, "-m", "pip", "install", "argo-proxy"], check=True)


def write_argo_proxy_server_config(
    username: str, port: int, upstream: str, *, skip_url_validation: bool = False,
) -> None:
    if ARGO_PROXY_CFG.exists():
        print(f"[argo] argo-proxy config already exists: {ARGO_PROXY_CFG}")
        return
    print(f"[argo] writing argo-proxy server config → {ARGO_PROXY_CFG}")
    _save_yaml(
        ARGO_PROXY_CFG,
        {
            "config_version": "3",
            "user": username,
            "host": "0.0.0.0",
            "port": port,
            "verbose": True,
            "argo_base_url": upstream,
            "skip_url_validation": skip_url_validation,
            "connection_test_timeout": 10,
        },
    )


def _tail_log(n: int = 40) -> str:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log file not found)"
    if not lines:
        return "(log file empty)"
    return "\n".join(lines[-n:])


def _startup_wait_seconds(ap: dict) -> float:
    return float(ap.get("startup_wait_seconds") or 120)


def preflight_upstream(ap: dict) -> bool:
    """Return True if this host can reach the ARGO API (required for argo-proxy)."""
    upstream = ap.get("argo_upstream") or ARGO_BASE_URLS["prod"]
    url = upstream.rstrip("/") + "/"
    host = socket.gethostname()
    print(f"[argo] preflight: host={host!r} — can we reach ARGO at {upstream}?")
    code, body = _http_get(url, timeout=10.0)
    if code == 0:
        print(f"  FAIL — no route to host ({body[:200]})")
        print(
            "  argo-proxy must run on a host with ANL internal network access.\n"
            "  Sage *field* nodes (hostname sb-core-*, sgt-node-*, etc.) often\n"
            "  cannot reach apps.inside.anl.gov without VPN or a firewall conduit.\n"
            "\n"
            "  Options:\n"
            "    A) Mac/laptop on ANL VPN — run proxy locally (no --jump):\n"
            "         bash scripts/setup_argo_proxy.sh -u USER -m gpt-4o\n"
            "    B) ANL lab/login node (V029/V030 / VPN-connected workstation):\n"
            "         ssh that-host && bash scripts/setup_argo_proxy.sh -u USER -m MODEL\n"
            "         Mac: bash scripts/setup_argo_proxy.sh -u USER --jump that-host\n"
            "    C) Debug only (proxy will likely fail API calls):\n"
            "         bash scripts/setup_argo_proxy.sh -u USER -m MODEL --skip-url-validation\n"
            "\n"
            "  Test manually: curl -v --connect-timeout 5 "
            f"{upstream}/"
        )
        return False
    print(f"  OK — HTTP {code} (upstream reachable)")
    return True


def cmd_check_network(_args: argparse.Namespace) -> int:
    cfg = load_msa_config()
    ap = cfg.get("argo_proxy") or DEFAULT_ARGO_PROXY
    ok = preflight_upstream(ap)
    print(f"\nhostname: {socket.gethostname()}")
    return 0 if ok else 1


def start_argo_proxy(ap: dict, *, skip_url_validation: bool = False) -> None:
    if (ap.get("ssh_jump_host") or "").strip():
        start_ssh_tunnel(ap)
        return
    if is_proxy_up(ap):
        print(f"[argo] proxy already responding at {proxy_base_url(ap)}")
        return

    if not preflight_upstream(ap) and not skip_url_validation:
        raise RuntimeError(
            "This node cannot reach the ARGO API. Start argo-proxy on an ANL node "
            "(node-V010 / V029 / V030) and re-run with --jump node-V010, or use "
            "--skip-url-validation only after fixing network/VPN."
        )

    ensure_argo_proxy_installed()
    username = (ap.get("username") or "").strip()
    port = int(ap.get("port") or 44497)
    upstream = ap.get("argo_upstream") or ARGO_BASE_URLS["prod"]
    if username:
        write_argo_proxy_server_config(
            username, port, upstream, skip_url_validation=skip_url_validation,
        )
    if not shutil.which("argo-proxy"):
        raise RuntimeError("argo-proxy binary not found after install")
    MSA_STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_FILE, "a", encoding="utf-8")
    print(f"[argo] starting `argo-proxy serve` (log: {LOG_FILE}) …")
    proc = subprocess.Popen(
        ["argo-proxy", "serve"],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    wait_s = _startup_wait_seconds(ap)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_proxy_up(ap):
            print(f"[argo] proxy up at {proxy_base_url(ap)} (pid {proc.pid})")
            return
        if proc.poll() is not None:
            tail = _tail_log()
            raise RuntimeError(
                f"argo-proxy exited early (code {proc.returncode}). "
                f"Last log lines:\n{tail}"
            )
        time.sleep(0.5)

    still = proc.poll() is None
    tail = _tail_log()
    hint = (
        f"Process pid {proc.pid} is still running — first start can be slow "
        f"(model list fetch). Try: curl {proxy_base_url(ap)}/health"
        if still
        else "Process exited during wait."
    )
    raise RuntimeError(
        f"argo-proxy did not respond within {wait_s:.0f}s. {hint}\n"
        f"Last log lines:\n{tail}"
    )


def start_ssh_tunnel(ap: dict) -> None:
    jump = ap["ssh_jump_host"].strip()
    remote_port = int(ap.get("port") or 44497)
    local_port = int(ap.get("ssh_local_port") or remote_port)
    tunnel_pid_file = MSA_STATE_DIR / "argo-proxy-tunnel.pid"
    if is_proxy_up(ap):
        print(f"[argo] SSH tunnel already forwarding :{local_port} → {jump}:{remote_port}")
        return
    if not shutil.which("ssh"):
        raise RuntimeError("ssh not found on PATH")
    MSA_STATE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[argo] opening SSH tunnel -L {local_port}:127.0.0.1:{remote_port} {jump} …")
    proc = subprocess.Popen(
        [
            "ssh", "-N",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-L", f"{local_port}:127.0.0.1:{remote_port}",
            jump,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    tunnel_pid_file.write_text(str(proc.pid))
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if is_proxy_up(ap):
            print(f"[argo] tunnel up — local {proxy_base_url(ap)} → {jump}")
            return
        if proc.poll() is not None:
            raise RuntimeError(f"SSH tunnel to {jump} exited (code {proc.returncode})")
        time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    raise RuntimeError(
        f"Tunnel opened but proxy not reachable at {proxy_base_url(ap)}. "
        f"On {jump}, run: bash scripts/setup_argo_proxy.sh --username … "
        f"See config/ssh_config.example for ~/.ssh/config"
    )


def update_env_file(ap: dict) -> None:
    username = (ap.get("username") or "").strip()
    base = proxy_base_url(ap)
    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()

    def _set(key: str, val: str) -> None:
        nonlocal lines
        prefix = f"export {key}="
        lines = [ln for ln in lines if not ln.strip().startswith(prefix)]
        lines.append(f'export {key}="{val}"')

    _set("ARGO_PROXY_BASE_URL", f"{base}/v1")
    if username:
        _set("ARGO_PROXY_API_KEY", username)
    if not any("argo proxy" in ln.lower() for ln in lines):
        lines.extend(["", "# >>> argo proxy (setup_argo_proxy.sh) >>>"])
    ENV_FILE.write_text("\n".join(lines).rstrip() + "\n")
    try:
        ENV_FILE.chmod(0o600)
    except OSError:
        pass
    print(f"[argo] updated {ENV_FILE}")


def run_tests(cfg: dict) -> bool:
    ap = cfg["argo_proxy"]
    username = (ap.get("username") or os.environ.get("ARGO_PROXY_API_KEY", "")).strip()
    base = proxy_base_url(ap)
    ok = True

    print(f"[test] GET {base}/health")
    code, body = _http_get(f"{base}/health")
    if code != 200:
        print(f"  FAIL HTTP {code}: {body[:300]}")
        ok = False
    else:
        print("  OK")

    print(f"[test] GET {base}/v1/models")
    code, body = _http_get(f"{base}/v1/models", api_key=username)
    if code != 200:
        print(f"  FAIL HTTP {code}: {body[:300]}")
        ok = False
    else:
        try:
            data = json.loads(body)
            ids = [m.get("id", "?") for m in data.get("data", [])[:5]]
            print(f"  OK — sample models: {', '.join(ids) or '(none listed)'}")
        except json.JSONDecodeError:
            print(f"  OK (non-JSON body): {body[:120]}…")

    model_cfg = resolve_model_config(cfg)
    model_name = model_cfg.get("model", "gpt-4o")
    print(f"[test] chat via LangChain ({model_name})")
    from ptz_node.llm_factory import chat_model_from_config

    try:
        llm = chat_model_from_config(model_cfg)
        resp = llm.invoke("Say OK only.")
        snippet = str(getattr(resp, "content", resp)).strip()[:120]
        if snippet:
            print(f"  OK — model replied: {snippet!r}")
        else:
            print("  FAIL — empty model response")
            ok = False
    except Exception as exc:
        print(f"  FAIL — {exc}")
        ok = False
    return ok


def cmd_setup(args: argparse.Namespace) -> int:
    cfg = load_msa_config()
    ap = cfg.setdefault("argo_proxy", dict(DEFAULT_ARGO_PROXY))
    if args.username:
        ap["username"] = args.username.strip()
    if args.model:
        ap["model"] = args.model.strip()
    if args.port:
        ap["port"] = int(args.port)
    if args.jump_host is not None:
        ap["ssh_jump_host"] = args.jump_host.strip()
    if args.upstream:
        ap["argo_upstream"] = ARGO_BASE_URLS.get(args.upstream, args.upstream)
    if not ap.get("username"):
        ap["username"] = input("ANL username (argo-proxy API key): ").strip()
    if not ap["username"]:
        print("ERROR: username is required", file=sys.stderr)
        return 1
    cfg = sync_model_from_argo_proxy(cfg, enable=not args.no_enable)
    save_local_config(cfg)
    update_env_file(ap)

    jump = (ap.get("ssh_jump_host") or "").strip()
    if jump:
        # Tunnel mode: never start argo-proxy locally on this host.
        if not args.no_tunnel:
            start_ssh_tunnel(ap)
    elif not args.no_auto_start and ap.get("auto_start", True):
        start_argo_proxy(ap, skip_url_validation=args.skip_url_validation)
    elif not is_proxy_up(ap):
        print("[argo] WARN: proxy not reachable (--no-auto-start)")
    if not args.skip_test:
        return 0 if run_tests(load_msa_config()) else 1
    return 0


def cmd_test(_args: argparse.Namespace) -> int:
    return 0 if run_tests(load_msa_config()) else 1


def cmd_disable(_args: argparse.Namespace) -> int:
    cfg = sync_model_from_argo_proxy(load_msa_config(), enable=False)
    save_local_config(cfg)
    print("[argo] disabled — model.provider falls back to ollama from default.yaml")
    return 0


def main() -> int:
    if sys.version_info < (3, 10):
        print(
            f"ERROR: Python 3.10+ required (argo-proxy). "
            f"You are running {sys.version.split()[0]} ({sys.executable}).\n"
            f"On node-V010 try: python3.11 -m venv .venv && source .venv/bin/activate\n"
            f"Then: bash scripts/setup_argo_proxy.sh -u USER -m gpt-4o",
            file=sys.stderr,
        )
        return 1
    p = argparse.ArgumentParser(description="Configure argo-proxy for jetson-ptz-agent-graph")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("setup")
    sp.add_argument("--username", "-u")
    sp.add_argument("--model", "-m", default="")
    sp.add_argument("--port", "-p", type=int, default=0)
    sp.add_argument("--jump-host", "--jump", "-j", default=None, dest="jump_host")
    sp.add_argument("--upstream", choices=list(ARGO_BASE_URLS.keys()))
    sp.add_argument("--no-enable", action="store_true")
    sp.add_argument("--no-auto-start", action="store_true",
                    help="Do not start local argo-proxy (ignored when --jump is set)")
    sp.add_argument("--no-tunnel", action="store_true",
                    help="With --jump: write config only, do not open SSH tunnel")
    sp.add_argument("--skip-test", action="store_true")
    sp.add_argument(
        "--skip-url-validation",
        action="store_true",
        help="Write skip_url_validation: true in argo-proxy config (debug only)",
    )
    sp.set_defaults(func=cmd_setup)
    sub.add_parser("test").set_defaults(func=cmd_test)
    sub.add_parser("disable").set_defaults(func=cmd_disable)
    sub.add_parser("check-network", help="test reachability of ARGO upstream").set_defaults(
        func=cmd_check_network
    )
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
