"""Discover and configure network sensors/cameras — bounded, never interactive.

Finds devices on the node's LAN (e.g. a PTZ camera on ethernet), identifies
them (name, IP, vendor, open protocols), and produces the exact configuration
steps to wire them into the gateway. Built for an autonomous edge agent:

  * every probe has a timeout, every phase has a retry cap and a global time
    budget — it can fail, but it cannot run forever;
  * it NEVER prompts for input mid-run. Anything only a human can supply
    (passwords, credentials, picking among candidates) is collected and returned
    in ``data.needs_user_input`` so the agent asks the user at the END.

Actions (``args['action']``):
  * ``scan``      (default) — discover candidate devices on the local subnet(s).
        args: max_retries=2, time_budget_s=120, subnet="192.168.1." (optional)
  * ``identify``  — deep-probe one host. args: ip (required)
  * ``configure`` — emit setup steps for one device. args: ip (required),
        backend ("reolink"|"auto"), username

Discovery uses only the standard library: ONVIF WS-Discovery + SSDP multicast,
the ARP/neighbor table, a small TCP port sweep, and HTTP/RTSP banner grabs.

CLI:    python -m ptz_node skill run sensor_discovery --args '{"action":"scan"}'
Agent:  run_skill("sensor_discovery", '{"action":"configure","ip":"192.168.1.108"}')
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
from typing import Any

from ptz_node.skills.base import BaseSkill, SkillContext, SkillResult

# Ports we sweep, with the protocol each implies for a camera/sensor.
_PORT_HINTS: dict[int, str] = {
    80: "http", 443: "https", 554: "rtsp", 8000: "http-alt",
    8554: "rtsp-alt", 9000: "onvif-alt", 2020: "onvif", 22: "ssh", 23: "telnet",
}
_VENDOR_OUI = {
    "ec:71:db": "Reolink", "9c:8e:cd": "Reolink",
    "00:1a:07": "Costar", "00:0f:7c": "ACTi", "00:18:ae": "Hikvision",
    "00:12:12": "Axis", "00:40:8c": "Axis", "ac:cc:8e": "Axis",
    "bc:ad:28": "Hikvision", "44:19:b6": "Hikvision",
    "e0:50:8b": "Dahua", "3c:ef:8c": "Dahua",
    "00:09:18": "Hanwha", "00:16:6c": "Hanwha", "e4:30:22": "Hanwha",
    "00:03:c5": "Mobotix",
}

# Vendor fingerprints found in HTTP banners / landing pages. MAC OUI is often
# unavailable across an L3 hop (no ARP entry), so the banner is the primary
# signal in practice — see node-dev deployment where the AXIS camera had no MAC.
_VENDOR_BANNER = (
    ("axis", "Axis"),
    ("hanwha", "Hanwha"), ("wisenet", "Hanwha"), ("samsung techwin", "Hanwha"),
    ("mobotix", "Mobotix"),
    ("hikvision", "Hikvision"), ("dahua", "Dahua"),
    ("reolink", "Reolink"), ("bosch", "Bosch"),
    ("vivotek", "Vivotek"), ("amcrest", "Amcrest"),
)

# Correct main-stream RTSP path per vendor (USER/PASS/IP filled in by configure).
# These differ per vendor — emitting the Reolink path for an Axis camera is wrong.
_RTSP_TEMPLATES = {
    "axis": "rtsp://{user}:{pw}@{ip}:554/axis-media/media.amp",
    "hanwha": "rtsp://{user}:{pw}@{ip}:554/profile1/media.smp",
    "mobotix": "rtsp://{user}:{pw}@{ip}:554/mobotix.mjpeg",
    "hikvision": "rtsp://{user}:{pw}@{ip}:554/Streaming/Channels/101",
    "dahua": "rtsp://{user}:{pw}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "amcrest": "rtsp://{user}:{pw}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "reolink": "rtsp://{user}:{pw}@{ip}:554/h264Preview_01_main",
    "bosch": "rtsp://{user}:{pw}@{ip}:554/rtsp_tunnel",
    "vivotek": "rtsp://{user}:{pw}@{ip}:554/live.sdp",
    "_default": "rtsp://{user}:{pw}@{ip}:554/  (vendor-specific — check ONVIF GetStreamUri)",
}

# Vendors the gateway can actually DRIVE (PTZ control) today. Everything else is
# discoverable + viewable over RTSP, but has no control driver yet (ONVIF TODO).
_DRIVABLE_BACKENDS = {"reolink"}
_PTZ_DEFAULT_PORTS = {554, 8000, 80}


def _progress(msg: str) -> None:
    print(f"[discover] {msg}", file=sys.stderr, flush=True)


class SensorDiscoverySkill(BaseSkill):
    name = "sensor_discovery"
    description = (
        "Discover and configure network sensors/cameras (e.g. a PTZ camera on "
        "ethernet). Finds device IP/name/vendor/open-protocols, then emits exact "
        "gateway setup steps. Bounded: per-probe timeouts, capped retries, and a "
        "global time budget so it can give up cleanly instead of running forever. "
        "Never prompts mid-run — anything only a human can provide (passwords, "
        "credentials, choosing a candidate) is returned in data.needs_user_input for "
        "you to ASK THE USER AT THE END. Actions: scan (default) | identify (args: ip) "
        "| configure (args: ip, backend, username)."
    )
    agent_callable = True

    def run(self, ctx: SkillContext) -> SkillResult:
        action = str(ctx.args.get("action", "scan")).lower()
        try:
            if action == "scan":
                return self._scan(ctx)
            if action == "identify":
                return self._identify(ctx)
            if action == "configure":
                return self._configure(ctx)
        except Exception as exc:  # discovery must never crash the loop
            return SkillResult(ok=False, skill=self.name,
                               summary=f"{action} failed: {type(exc).__name__}: {exc}")
        return SkillResult(ok=False, skill=self.name,
                           summary=f"unknown action {action!r}; use scan|identify|configure")

    # ------------------------------------------------------------------ #
    # scan
    # ------------------------------------------------------------------ #

    def _scan(self, ctx: SkillContext) -> SkillResult:
        a = ctx.args
        max_retries = int(a.get("max_retries", 2))
        budget = float(a.get("time_budget_s", 120))
        t0 = time.time()
        attempts: list[dict[str, Any]] = []
        hosts: dict[str, dict[str, Any]] = {}

        for attempt in range(1, max_retries + 1):
            if time.time() - t0 > budget:
                attempts.append({"attempt": attempt, "skipped": "time budget"})
                break
            _progress(f"scan attempt {attempt}/{max_retries} "
                      f"[{time.time() - t0:.0f}s/{budget:.0f}s]")
            found: dict[str, dict[str, Any]] = {}

            # 1) active discovery protocols (cameras answer these directly)
            for ip in self._ws_discovery(timeout=4.0):
                found.setdefault(ip, {"ip": ip, "sources": []})["sources"].append("onvif")
            for ip in self._ssdp_discovery(timeout=3.0):
                found.setdefault(ip, {"ip": ip, "sources": []})["sources"].append("ssdp")
            # 2) passive: whatever the OS already knows about the L2 neighborhood
            for ip, mac in self._arp_table().items():
                e = found.setdefault(ip, {"ip": ip, "sources": []})
                e["mac"] = mac
                e["vendor"] = self._vendor_from_mac(mac)
                e["sources"].append("arp")

            # 3) confirm liveness + protocols with a tiny port sweep
            for ip, e in found.items():
                if time.time() - t0 > budget:
                    break
                e["open_ports"] = self._scan_ports(ip, timeout=0.6)
                e["likely_camera"] = self._looks_like_camera(e)

            attempts.append({"attempt": attempt, "candidates": len(found),
                             "cameras": sum(1 for e in found.values()
                                            if e.get("likely_camera"))})
            for ip, e in found.items():
                hosts.setdefault(ip, e).update(e)
            if any(e.get("likely_camera") for e in hosts.values()):
                break  # got at least one camera — stop retrying

        cameras = [e for e in hosts.values() if e.get("likely_camera")]
        others = [e for e in hosts.values() if not e.get("likely_camera")]
        elapsed = round(time.time() - t0, 1)

        needs: list[dict[str, Any]] = []
        if not hosts:
            summary = (f"no devices found after {len(attempts)} attempt(s) in "
                       f"{elapsed}s — check the ethernet link / that the camera is "
                       f"powered and on this subnet")
            needs.append({"field": "manual_ip",
                          "prompt": "Discovery found nothing. If you know the camera's "
                                    "IP, provide it so I can probe it directly.",
                          "secret": False})
        else:
            summary = (f"found {len(hosts)} device(s) "
                       f"({len(cameras)} likely camera(s)) in {elapsed}s over "
                       f"{len(attempts)} attempt(s)")
            if len(cameras) > 1:
                needs.append({
                    "field": "selected_ip",
                    "prompt": "Multiple cameras found — which IP should I configure? "
                              + ", ".join(c["ip"] for c in cameras),
                    "options": [c["ip"] for c in cameras], "secret": False})
            elif not cameras:
                opts = [e["ip"] for e in others]
                needs.append({
                    "field": "selected_ip",
                    "prompt": ("No device clearly identified as a camera, but these "
                               "hosts are reachable: " + ", ".join(opts) +
                               ". Which one is the camera (or give another IP)? "
                               "I'll deep-probe it with action=identify."),
                    "options": opts, "secret": False})

        return SkillResult(
            ok=bool(hosts), skill=self.name, summary=summary,
            data={"cameras": cameras, "other_devices": others,
                  "attempts": attempts, "elapsed_s": elapsed,
                  "needs_user_input": needs,
                  "next": "run action=identify with a chosen ip, then action=configure"},
        )

    # ------------------------------------------------------------------ #
    # identify
    # ------------------------------------------------------------------ #

    def _identify(self, ctx: SkillContext) -> SkillResult:
        ip = str(ctx.args.get("ip", "")).strip()
        if not ip:
            return SkillResult(ok=False, skill=self.name,
                               summary="identify requires args.ip",
                               data={"needs_user_input": [
                                   {"field": "ip", "prompt": "Which IP should I probe?",
                                    "secret": False}]})
        _progress(f"identifying {ip}")
        ports = self._scan_ports(ip, timeout=1.0)
        mac = self._arp_table().get(ip, "")
        banner = self._http_banner(ip) if (80 in ports or 8000 in ports) else ""
        # Prefer MAC OUI when present; otherwise (the common L3 case) read the
        # vendor straight out of the HTTP banner / landing page.
        vendor = self._vendor_from_mac(mac) or self._vendor_from_banner(banner)
        info: dict[str, Any] = {
            "ip": ip, "mac": mac, "vendor": vendor,
            "hostname": self._reverse_dns(ip), "open_ports": ports,
            "http_banner": banner,
            "model": self._model_from_banner(banner),
            "rtsp": 554 in ports or 8554 in ports,
            "onvif": 2020 in ports or 9000 in ports,
        }
        info["likely_camera"] = self._looks_like_camera(info)
        reachable = bool(ports)
        return SkillResult(
            ok=reachable, skill=self.name,
            summary=(f"{ip}: {info['vendor'] or 'unknown vendor'}, "
                     f"{len(ports)} open port(s)"
                     + (" — looks like a camera" if info["likely_camera"]
                        else "" if reachable else " — UNREACHABLE")),
            data={**info, "next": "run action=configure with this ip"},
        )

    # ------------------------------------------------------------------ #
    # configure
    # ------------------------------------------------------------------ #

    def _configure(self, ctx: SkillContext) -> SkillResult:
        ip = str(ctx.args.get("ip", "")).strip()
        if not ip:
            return SkillResult(ok=False, skill=self.name,
                               summary="configure requires args.ip",
                               data={"needs_user_input": [
                                   {"field": "ip", "prompt": "Which camera IP?",
                                    "secret": False}]})
        requested_backend = str(ctx.args.get("backend", "auto")).lower()
        username = str(ctx.args.get("username", "admin"))
        ident = self._identify(SkillContext(config=ctx.config, args={"ip": ip}))
        vendor_name = (ident.data.get("vendor") or "")
        vendor = vendor_name.lower()

        # Pick a control backend from the detected vendor. Only vendors in
        # _DRIVABLE_BACKENDS have a real PTZ-control driver today; the rest are
        # discoverable + viewable over RTSP but NOT controllable until an ONVIF /
        # vendor driver lands. Be honest about that instead of pretending Reolink.
        if requested_backend not in ("auto", ""):
            backend = requested_backend
        elif "reolink" in vendor:
            backend = "reolink"
        else:
            backend = "onvif"  # placeholder kind: discoverable, not yet drivable

        drivable = backend in _DRIVABLE_BACKENDS
        rtsp_url = _RTSP_TEMPLATES.get(vendor, _RTSP_TEMPLATES["_default"]).format(
            user=username or "USER", pw="PASSWORD", ip=ip)

        needs = [{
            "field": "CAMERA_PASSWORD",
            "prompt": f"Enter the password for camera {ip} (user '{username}'). "
                      "I will not store it; pass it via env var, not YAML/logs.",
            "secret": True,
        }]
        if username == "admin":
            needs.append({
                "field": "CAMERA_USER",
                "prompt": "Confirm the camera username (default 'admin' assumed).",
                "secret": False, "default": "admin"})

        if drivable:
            # ptz-agent Reolink driver: creds from env at connect time, selected
            # via MSA_PTZ_BACKEND=reolink.
            env_steps = {
                "MSA_PTZ_BACKEND": "reolink",
                "REOLINK_IP": ip,
                "REOLINK_USER": username,
                "REOLINK_PASSWORD": "<ASK_USER>",
            }
            shell = "\n".join(f"export {k}={v}" for k, v in env_steps.items())
            verify = [
                "source the env vars above (REOLINK_PASSWORD must be set)",
                "python -m ptz_node devices        # ptz_primary should show backend=reolink",
                "python -m ptz_node invoke ptz_primary get_position",
            ]
            notes = ("Reolink creds are read from the environment at connect time — "
                     "never commit the password. Live RTSP: " + rtsp_url)
            summary = (f"configuration plan for {ip} "
                       f"(vendor={vendor_name or 'unknown'}, backend=reolink — drivable); "
                       f"{len(needs)} item(s) need the user (password required)")
        else:
            # No control driver for this vendor yet. Don't emit Reolink env that
            # would silently fail to drive an Axis/Hanwha/Mobotix PTZ.
            env_steps = {}
            shell = ""
            verify = [
                "Viewing only (no PTZ control driver yet for this vendor):",
                f"ffplay '{rtsp_url}'   # replace PASSWORD; confirm the stream",
                "To control PTZ, a generic ONVIF driver is needed — see "
                "ptz_node/sensor_gateway/drivers/ (TODO: onvif_camera.py).",
            ]
            notes = (
                f"'{vendor_name or 'this vendor'}' has NO PTZ-control driver in the "
                "gateway yet — only sim + Reolink are drivable. The camera is reachable "
                "and viewable over RTSP at the URL below, but move_to/snapshot via the "
                "gateway will not work until an ONVIF (or vendor-specific) driver is added. "
                "Live RTSP: " + rtsp_url)

        return SkillResult(
            ok=True, skill=self.name,
            summary=summary if drivable else (
                f"{ip} identified as {vendor_name or 'unknown vendor'} — reachable but "
                f"NOT drivable yet (no {vendor or 'onvif'} control driver); RTSP viewing only"),
            data={
                "ip": ip, "backend": backend, "vendor": vendor_name,
                "model": ident.data.get("model", ""),
                "drivable": drivable,
                "rtsp_url": rtsp_url,
                "env_exports": env_steps,
                "shell": shell,
                "verify_steps": verify,
                "notes": notes,
                "needs_user_input": needs,
            },
        )

    # ------------------------------------------------------------------ #
    # discovery primitives (stdlib only, all timeout-bounded)
    # ------------------------------------------------------------------ #

    def _ws_discovery(self, timeout: float) -> list[str]:
        """ONVIF WS-Discovery probe (SOAP over UDP multicast 239.255.255.250:3702)."""
        msg = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
            'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
            'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
            '<e:Header><w:MessageID>uuid:sage-discovery</w:MessageID>'
            '<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
            '<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
            '</e:Header><e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types>'
            '</d:Probe></e:Body></e:Envelope>'
        )
        return self._udp_multicast("239.255.255.250", 3702, msg.encode(), timeout)

    def _ssdp_discovery(self, timeout: float) -> list[str]:
        """SSDP M-SEARCH (UPnP); many cameras/NVRs answer."""
        msg = ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\n"
               'MAN: "ssdp:discover"\r\nMX: 2\r\nST: ssdp:all\r\n\r\n')
        return self._udp_multicast("239.255.255.250", 1900, msg.encode(), timeout)

    def _udp_multicast(self, group: str, port: int, payload: bytes,
                       timeout: float) -> list[str]:
        ips: set[str] = set()
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.settimeout(timeout)
            sock.sendto(payload, (group, port))
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    _data, addr = sock.recvfrom(65535)
                    ips.add(addr[0])
                except socket.timeout:
                    break
                except OSError:
                    break
        except OSError:
            pass
        finally:
            if sock is not None:
                sock.close()
        return sorted(ips)

    def _arp_table(self) -> dict[str, str]:
        """Harvest the OS neighbor table (Linux `ip neigh`, fallback `arp -a`)."""
        out: dict[str, str] = {}
        for cmd in (["ip", "neigh"], ["arp", "-a"]):
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            if res.returncode != 0 or not res.stdout:
                continue
            for line in res.stdout.splitlines():
                ipm = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", line)
                macm = re.search(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})", line)
                if ipm and macm:
                    out[ipm.group(1)] = macm.group(1).lower()
            if out:
                break
        return out

    def _scan_ports(self, ip: str, timeout: float) -> list[int]:
        open_ports: list[int] = []
        for port in _PORT_HINTS:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    if s.connect_ex((ip, port)) == 0:
                        open_ports.append(port)
            except OSError:
                continue
        return open_ports

    def _http_banner(self, ip: str) -> str:
        for port in (80, 8000):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.5)
                    if s.connect_ex((ip, port)) != 0:
                        continue
                    s.sendall(b"GET / HTTP/1.0\r\nHost: %b\r\n\r\n" % ip.encode())
                    return s.recv(2048).decode("latin-1", "replace")[:1000]
            except OSError:
                continue
        return ""

    def _reverse_dns(self, ip: str) -> str:
        try:
            socket.setdefaulttimeout(2.0)
            return socket.gethostbyaddr(ip)[0]
        except OSError:
            return ""
        finally:
            socket.setdefaulttimeout(None)

    def _vendor_from_mac(self, mac: str) -> str:
        if not mac:
            return ""
        return _VENDOR_OUI.get(mac.lower()[:8], "")

    def _vendor_from_banner(self, banner: str) -> str:
        """Read a camera vendor out of an HTTP banner / landing page body.

        Across an L3 hop there is usually no ARP/MAC entry, so the banner is the
        only vendor signal — an Axis camera literally serves ``Axis Communications``
        and ``<title>AXIS</title>``. Without this, identify reports 'unknown'.
        """
        if not banner:
            return ""
        low = banner.lower()
        for needle, name in _VENDOR_BANNER:
            if needle in low:
                return name
        return ""

    _MODEL_RE = re.compile(
        r"\b((?:Q\d{4}|XN[PV]-\d{4,}|M\d{2}|DS-[A-Z0-9-]+)[A-Z0-9-]*)\b")

    def _model_from_banner(self, banner: str) -> str:
        """Best-effort model token (Axis Q6055, Hanwha XNP-6400RW, Mobotix M16…)."""
        if not banner:
            return ""
        m = self._MODEL_RE.search(banner)
        return m.group(1) if m else ""

    def _looks_like_camera(self, entry: dict[str, Any]) -> bool:
        ports = set(entry.get("open_ports") or [])
        vendor = (entry.get("vendor") or "").lower()
        if any(v in vendor for v in ("reolink", "hikvision", "dahua", "axis", "acti",
                                     "hanwha", "mobotix", "bosch", "vivotek", "amcrest")):
            return True
        if "onvif" in (entry.get("sources") or []):
            return True
        return bool(ports & _PTZ_DEFAULT_PORTS) and (554 in ports or 8554 in ports)
