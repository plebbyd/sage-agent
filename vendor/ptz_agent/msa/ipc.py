"""msa/ipc.py — Unix-domain-socket JSON-RPC for chat ↔ supervisor.

Why a custom mini-RPC instead of pulling in jsonrpcserver/grpc?

  * Zero deps; one file to audit.
  * The contract is small (≈10 methods).
  * We need both blocking (rpc-call) and streaming (event subscribe)
    semantics; a one-message-per-line framing handles both.

Wire format
-----------
Each message is a single JSON object terminated by ``\n``. Requests:

    {"id": "<uuid>", "method": "list_workers", "params": {...}}

Responses:

    {"id": "<uuid>", "result": <any>}        # success
    {"id": "<uuid>", "error":  {"code": "Err", "message": "..."}}

For streaming (events), the server sends a ``result: {"stream": "<id>"}``
ack, then 0..N notifications:

    {"stream": "<id>", "event": {...}}

and finally:

    {"stream": "<id>", "done": true}
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Socket location
# ---------------------------------------------------------------------------

def default_socket_path() -> Path:
    """Resolve the supervisor's listening socket.

    Honours $MSA_SOCKET; otherwise puts it next to the SQLite DB so a
    single $HOME/.msa/ directory holds everything.
    """
    env = os.environ.get("MSA_SOCKET")
    if env:
        return Path(env)
    home = Path.home() / ".msa"
    home.mkdir(parents=True, exist_ok=True)
    return home / "supervisord.sock"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class IPCError(RuntimeError):
    """Raised by Client when the supervisor returns an error response."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Client (used by the chat REPL, web UI, and the test harness)
# ---------------------------------------------------------------------------

class Client:
    """Synchronous client; opens one connection per call.

    The supervisor's worker thread pool is small but each call is short
    (tens of ms typical), so per-call connections keep the protocol
    state machine simple. For the events stream we keep a connection
    open in ``stream_events``.
    """

    def __init__(self, socket_path: str | Path | None = None, timeout: float = 30.0):
        self.socket_path = Path(socket_path) if socket_path else default_socket_path()
        self.timeout = timeout

    # ---- single-shot RPC --------------------------------------------------

    def call(self, method: str, **params) -> object:
        req = {"id": uuid.uuid4().hex, "method": method, "params": params}
        with _connect(self.socket_path, self.timeout) as sock:
            _send(sock, req)
            for resp in _recv_iter(sock):
                if resp.get("id") != req["id"]:
                    continue
                if "error" in resp:
                    err = resp["error"] or {}
                    raise IPCError(
                        err.get("code", "RPCError"),
                        err.get("message", str(err)),
                    )
                return resp.get("result")
        raise IPCError("Disconnected", "supervisor closed connection")

    # ---- ergonomic helpers ------------------------------------------------

    def ping(self) -> dict:
        return self.call("ping")  # type: ignore[return-value]

    def schedule_task(
        self,
        name: str,
        prompt: str,
        *,
        cron: str | None = None,
        interval_seconds: int | None = None,
        enabled: bool = True,
    ) -> dict:
        return self.call(  # type: ignore[return-value]
            "schedule_task",
            name=name, prompt=prompt, cron=cron,
            interval_seconds=interval_seconds, enabled=enabled,
        )

    def run_task_now(
        self,
        prompt: str,
        *,
        name: str | None = None,
        parent_id: str | None = None,
        spawned_by: str = "chat",
        is_master: bool = False,
    ) -> dict:
        return self.call(  # type: ignore[return-value]
            "run_task_now",
            prompt=prompt, name=name, parent_id=parent_id,
            spawned_by=spawned_by, is_master=is_master,
        )

    def list_workers(self, *, states: Iterable[str] | None = None,
                     limit: int = 50) -> list[dict]:
        return self.call("list_workers", states=list(states) if states else None,
                          limit=limit)  # type: ignore[return-value]

    def get_worker(self, worker_id: str) -> dict | None:
        return self.call("get_worker", worker_id=worker_id)  # type: ignore[return-value]

    def cancel_worker(self, worker_id: str) -> dict:
        return self.call("cancel_worker", worker_id=worker_id)  # type: ignore[return-value]

    def list_tasks(self) -> list[dict]:
        return self.call("list_tasks")  # type: ignore[return-value]

    def delete_task(self, name: str) -> dict:
        return self.call("delete_task", name=name)  # type: ignore[return-value]

    def get_transcript(
        self, worker_id: str, *, since_seq: int = 0,
    ) -> list[dict]:
        return self.call(  # type: ignore[return-value]
            "get_transcript", worker_id=worker_id, since_seq=since_seq,
        )

    # ---- streaming events -------------------------------------------------

    def stream_events(
        self, *, worker_id: str | None = None, after_id: int = 0,
    ) -> Iterator[dict]:
        """Yield events as they arrive. Connection stays open."""
        req = {
            "id": uuid.uuid4().hex,
            "method": "subscribe_events",
            "params": {"worker_id": worker_id, "after_id": after_id},
        }
        with _connect(self.socket_path, timeout=None) as sock:
            _send(sock, req)
            for msg in _recv_iter(sock):
                if msg.get("done"):
                    return
                if "event" in msg:
                    yield msg["event"]


# ---------------------------------------------------------------------------
# Server (used by the supervisor)
# ---------------------------------------------------------------------------

class Server:
    """Threaded UDS server.

    Methods are registered by name. Subscriptions are special-cased so a
    single client connection can stream many messages — the handler is a
    generator that yields events.
    """

    def __init__(self, socket_path: str | Path | None = None):
        self.socket_path = Path(socket_path) if socket_path else default_socket_path()
        self._methods: dict[str, Callable] = {}
        self._subscriptions: dict[str, Callable[..., Iterator[dict]]] = {}
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None

    # ---- registration -----------------------------------------------------

    def method(self, name: str | None = None):
        def deco(fn: Callable):
            self._methods[name or fn.__name__] = fn
            return fn
        return deco

    def subscription(self, name: str | None = None):
        def deco(fn: Callable[..., Iterator[dict]]):
            self._subscriptions[name or fn.__name__] = fn
            return fn
        return deco

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        # remove stale socket if any (only safe if no live server is using it).
        if self.socket_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as test:
                    test.settimeout(0.2)
                    test.connect(str(self.socket_path))
                raise RuntimeError(
                    f"another supervisor already owns {self.socket_path}"
                )
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                self.socket_path.unlink(missing_ok=True)

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        srv = _ThreadingUnixServer(str(self.socket_path), _Handler, self)
        os.chmod(self.socket_path, 0o600)
        self._server = srv
        self._thread = threading.Thread(target=srv.serve_forever,
                                         name="msa-ipc", daemon=True)
        self._thread.start()
        logger.info("IPC server listening on %s", self.socket_path)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None


class _ThreadingUnixServer(socketserver.ThreadingMixIn,
                            socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, sockpath: str, handler_cls, app: Server):
        super().__init__(sockpath, handler_cls)
        self.app = app


class _Handler(socketserver.BaseRequestHandler):
    server: _ThreadingUnixServer  # type: ignore[assignment]

    def handle(self) -> None:
        try:
            for msg in _recv_iter(self.request):
                self._dispatch(msg)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            logger.exception("IPC handler error")

    def _dispatch(self, msg: dict) -> None:
        app = self.server.app
        rid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method in app._methods:
            try:
                result = app._methods[method](**params)
            except Exception as exc:  # noqa: BLE001
                logger.exception("RPC %s failed", method)
                _send(self.request, {
                    "id": rid,
                    "error": {"code": type(exc).__name__, "message": str(exc)},
                })
                return
            _send(self.request, {"id": rid, "result": result})
            return

        if method in app._subscriptions:
            sid = uuid.uuid4().hex
            _send(self.request, {"id": rid, "result": {"stream": sid}})
            try:
                for event in app._subscriptions[method](**params):
                    _send(self.request, {"stream": sid, "event": event})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("subscription %s failed", method)
                _send(self.request, {
                    "stream": sid,
                    "error": {"code": type(exc).__name__, "message": str(exc)},
                })
            _send(self.request, {"stream": sid, "done": True})
            return

        _send(self.request, {
            "id": rid,
            "error": {"code": "MethodNotFound",
                      "message": f"unknown method: {method!r}"},
        })


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------

def _send(sock: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(data)


def _recv_iter(sock: socket.socket) -> Iterator[dict]:
    """Yield decoded JSON objects, one per newline-delimited frame."""
    buf = b""
    while True:
        try:
            chunk = sock.recv(65536)
        except (BrokenPipeError, ConnectionResetError):
            return
        if not chunk:
            return
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line, buf = buf[:nl], buf[nl + 1:]
            if not line.strip():
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                logger.warning("IPC: dropped malformed frame: %r", line[:200])


@contextmanager
def _connect(path: Path, timeout: float | None):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if timeout is not None:
        sock.settimeout(timeout)
    try:
        sock.connect(str(path))
    except FileNotFoundError as exc:
        sock.close()
        raise IPCError(
            "SupervisorDown",
            f"no supervisor at {path}; start it with `msa supervisor`",
        ) from exc
    except ConnectionRefusedError as exc:
        sock.close()
        raise IPCError(
            "SupervisorDown",
            f"supervisor at {path} refused connection",
        ) from exc
    try:
        yield sock
    finally:
        try:
            sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Module convenience
# ---------------------------------------------------------------------------

def is_supervisor_running(socket_path: str | Path | None = None) -> bool:
    path = Path(socket_path) if socket_path else default_socket_path()
    if not path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(str(path))
        return True
    except OSError:
        return False


def wait_for_supervisor(
    socket_path: str | Path | None = None, *, timeout: float = 10.0,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_supervisor_running(socket_path):
            return True
        time.sleep(0.2)
    return False
