"""msa/webui.py — Stdlib-only worker dashboard.

Why stdlib only?

  * Edge devices don't always have Flask available, and we already
    require ``requests`` for Ollama; pulling in another web framework
    for a 200-LOC dashboard is overkill.
  * The HTML/JS is intentionally crude (no build step, no React) so
    you can read every byte the browser receives.

What it shows
-------------
* Worker tree: parent → children, grouped by chat session.
* Live updates (poll every 2 s) of state, iteration count, tokens, runtime.
* Per-worker: full transcript, event timeline, cancel button.
* Scheduled tasks list.

Reads go straight to the SQLite store (fast, lock-free under WAL).
Actions (``cancel_worker``, ``delete_task``) go through the supervisor
IPC so we keep one source of truth for state transitions.
"""

from __future__ import annotations

import http.server
import json
import logging
import socketserver
import sys
import urllib.parse
from dataclasses import asdict
from pathlib import Path

from . import ipc
from .store import Store

logger = logging.getLogger("msa.webui")


# ---------------------------------------------------------------------------
# HTML + JS (single-page app, no build step)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>MSA — workers</title>
<style>
  :root { color-scheme: dark; }
  body {
    font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #0e1116; color: #d8d8d8;
    margin: 0; padding: 16px;
  }
  h1 { font-size: 18px; margin: 0 0 8px; color: #fff; }
  h2 { font-size: 14px; margin: 16px 0 8px; color: #b0b8c8; text-transform: uppercase; letter-spacing: 0.05em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #161a22; border: 1px solid #232936; border-radius: 6px; padding: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #232936; vertical-align: top; }
  th { font-weight: 500; color: #8088a0; }
  tr.row { cursor: pointer; }
  tr.row:hover { background: #1d2230; }
  tr.row.selected { background: #243047; }
  .pill {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
  }
  .state-pending  { background: #3a3b00; color: #ffeb3b; }
  .state-running  { background: #1f3a5b; color: #57b0ff; }
  .state-completed{ background: #14401a; color: #5cdf6e; }
  .state-failed   { background: #3d1414; color: #ff5757; }
  .state-cancelled{ background: #2a2a2a; color: #aaaaaa; }
  .id { font-family: ui-monospace, Menlo, monospace; color: #b0d4ff; }
  .small { color: #6c7488; font-size: 12px; }
  .prompt-cell { max-width: 480px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .transcript {
    font-family: ui-monospace, Menlo, monospace; font-size: 12px;
    background: #0a0d12; padding: 10px; border-radius: 4px;
    white-space: pre-wrap; max-height: 60vh; overflow: auto;
  }
  .role-system { color: #8088a0; }
  .role-user { color: #57b0ff; }
  .role-assistant { color: #5cdf6e; }
  .role-tool { color: #c576ff; }
  .role-event { color: #ff9e57; }
  button {
    background: #2a3346; color: #d8d8d8; border: 1px solid #3a4459;
    border-radius: 4px; padding: 4px 10px; cursor: pointer; font-size: 12px;
  }
  button:hover { background: #3a4459; }
  button.danger { background: #3d1414; border-color: #5a1f1f; color: #ff8080; }
  button.danger:hover { background: #5a1f1f; }
  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
  .indent { padding-left: 16px; }
  .tree-line { color: #4a5266; }
</style>

<h1>MSA — workers</h1>
<div class="small" id="ping">supervisor: …</div>

<div class="grid" style="margin-top: 12px;">
  <div class="card">
    <div class="header">
      <h2 style="margin: 0;">Workers</h2>
      <span class="small" id="updated">updated …</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>id</th>
          <th>state</th>
          <th>iter</th>
          <th>tokens</th>
          <th>runtime</th>
          <th>prompt</th>
          <th></th>
        </tr>
      </thead>
      <tbody id="workers"></tbody>
    </table>
    <h2 style="margin-top: 16px;">Scheduled tasks</h2>
    <table>
      <thead>
        <tr><th>name</th><th>schedule</th><th>next run</th><th>runs</th><th></th></tr>
      </thead>
      <tbody id="tasks"></tbody>
    </table>
  </div>

  <div class="card">
    <div class="header">
      <h2 style="margin: 0;">Detail</h2>
      <span id="detail-actions"></span>
    </div>
    <div id="detail">Click a worker to inspect.</div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const fmt = (s) => s == null ? "—" : s;
const seconds = n => n == null ? "—" : `${(+n).toFixed(1)}s`;
const localTs = ts => ts ? new Date(ts * 1000).toLocaleTimeString() : "";
let selected = null;
let workersById = {};

async function fetchJSON(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function indentFor(w, depth) {
  if (depth <= 0) return "";
  return `<span class="tree-line">${"  ".repeat(depth - 1)}└─ </span>`;
}

function buildTree(workers) {
  // Group children under parents; root = no parent_id.
  const children = {};
  for (const w of workers) {
    const k = w.parent_id || "_root";
    (children[k] ||= []).push(w);
  }
  const lines = [];
  function walk(id, depth) {
    const list = children[id] || [];
    for (const w of list) {
      lines.push({w, depth});
      walk(w.id, depth + 1);
    }
  }
  walk("_root", 0);
  return lines;
}

function renderWorkers(workers) {
  workersById = Object.fromEntries(workers.map(w => [w.id, w]));
  const lines = buildTree(workers);
  const tbody = $("workers");
  tbody.innerHTML = lines.map(({w, depth}) => `
    <tr class="row ${selected === w.id ? "selected" : ""}" data-id="${w.id}">
      <td class="id">${indentFor(w, depth)}${w.id}</td>
      <td><span class="pill state-${w.state}">${w.state}</span></td>
      <td>${w.iterations}/${w.max_iterations}</td>
      <td>${w.total_tokens ?? "?"}</td>
      <td>${seconds(w.runtime_seconds)}</td>
      <td class="prompt-cell" title="${escapeHtml(w.prompt || "")}">${escapeHtml(w.prompt || "")}</td>
      <td>${w.state === "running" || w.state === "pending"
            ? `<button class="danger" data-cancel="${w.id}">cancel</button>` : ""}</td>
    </tr>
  `).join("");
  for (const tr of tbody.querySelectorAll("tr.row")) {
    tr.addEventListener("click", e => {
      if (e.target.closest("[data-cancel]")) return;
      selectWorker(tr.dataset.id);
    });
  }
  for (const btn of tbody.querySelectorAll("[data-cancel]")) {
    btn.addEventListener("click", async e => {
      e.stopPropagation();
      const id = btn.dataset.cancel;
      try {
        await fetchJSON(`/api/cancel/${id}`, {method: "POST"});
      } catch (e) { alert(e.message); }
    });
  }
  $("updated").textContent = "updated " + new Date().toLocaleTimeString();
}

function renderTasks(tasks) {
  $("tasks").innerHTML = (tasks || []).map(t => `
    <tr>
      <td>${escapeHtml(t.name)}</td>
      <td>${t.interval_seconds ? `every ${t.interval_seconds}s`
                                : `cron ${escapeHtml(t.cron || "")}`}</td>
      <td>${t.next_run ? new Date(t.next_run * 1000).toLocaleString() : "—"}</td>
      <td>${t.run_count}</td>
      <td><button class="danger" data-rmtask="${escapeAttr(t.name)}">delete</button></td>
    </tr>
  `).join("");
  for (const btn of $("tasks").querySelectorAll("[data-rmtask]")) {
    btn.addEventListener("click", async () => {
      try {
        await fetchJSON(`/api/task/${encodeURIComponent(btn.dataset.rmtask)}`,
                          {method: "DELETE"});
      } catch (e) { alert(e.message); }
    });
  }
}

async function selectWorker(id) {
  selected = id;
  for (const tr of document.querySelectorAll("tr.row")) {
    tr.classList.toggle("selected", tr.dataset.id === id);
  }
  await renderDetail(id);
}

async function renderDetail(id) {
  const w = workersById[id];
  if (!w) { $("detail").textContent = "(no worker)"; return; }
  let entries = [];
  try { entries = await fetchJSON(`/api/transcript/${id}`); } catch (e) {}
  $("detail-actions").innerHTML = (w.state === "running" || w.state === "pending")
      ? `<button class="danger" data-cancel="${id}">cancel</button>` : "";
  for (const btn of $("detail-actions").querySelectorAll("[data-cancel]")) {
    btn.addEventListener("click", async () => {
      try { await fetchJSON(`/api/cancel/${id}`, {method: "POST"}); } catch(e) { alert(e.message); }
    });
  }
  const meta = w.metadata || {};
  const info = `
    <div><span class="small">id</span> <span class="id">${id}</span></div>
    <div><span class="small">state</span> <span class="pill state-${w.state}">${w.state}</span></div>
    <div><span class="small">spawned by</span> ${escapeHtml(w.spawned_by || "")}</div>
    <div><span class="small">parent</span> <span class="id">${w.parent_id || "—"}</span></div>
    <div><span class="small">model</span> ${escapeHtml(w.model || "?")}</div>
    <div><span class="small">tokens</span> in=${w.prompt_tokens ?? "?"} out=${w.completion_tokens ?? "?"}</div>
    <div><span class="small">runtime</span> ${seconds(w.runtime_seconds)}</div>
    ${w.error ? `<div style="color:#ff8080"><span class="small">error</span> ${escapeHtml(w.error)}</div>` : ""}
    ${w.result ? `<div style="margin-top:6px"><span class="small">final</span> <pre class="transcript" style="margin:4px 0">${escapeHtml(w.result)}</pre></div>` : ""}
  `;
  const transcript = entries.length === 0
      ? "(empty)"
      : entries.map(e => `<span class="role-${e.role}">[${e.role}] ${escapeHtml(e.content)}</span>`).join("\n\n");
  $("detail").innerHTML = info + `<h2>Transcript</h2><div class="transcript">${transcript}</div>`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}
function escapeAttr(s) { return escapeHtml(s).replace(/"/g, "&quot;"); }

async function tick() {
  try {
    const ping = await fetchJSON("/api/ping");
    $("ping").textContent = `supervisor: running (${ping.running} active, cap ${ping.concurrency})`;
  } catch (e) {
    $("ping").innerHTML = `<span style="color:#ff8080">supervisor: down</span>`;
  }
  try {
    const ws = await fetchJSON("/api/workers");
    renderWorkers(ws);
    if (selected) renderDetail(selected);
  } catch (e) {}
  try {
    const ts = await fetchJSON("/api/tasks");
    renderTasks(ts);
  } catch (e) {}
}

tick();
setInterval(tick, 2000);
</script>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    store: Store
    client: ipc.Client

    # --- routes ----------------------------------------------------------

    def _ok(self, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, msg: str) -> None:
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                                # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path == "/" or path == "/index.html":
            return self._ok(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/ping":
            try:
                return self._ok(json.dumps(self.client.ping()).encode())
            except ipc.IPCError as exc:
                return self._err(503, str(exc))
        if path == "/api/workers":
            ws = [self._serialise_worker(w) for w in
                   self.store.list_workers(limit=200)]
            return self._ok(json.dumps(ws).encode())
        if path == "/api/tasks":
            ts = [asdict(t) for t in self.store.list_tasks()]
            return self._ok(json.dumps(ts).encode())
        if path.startswith("/api/transcript/"):
            wid = path.rsplit("/", 1)[-1]
            entries = self.store.get_transcript(wid)
            return self._ok(json.dumps(entries).encode())
        if path.startswith("/api/worker/"):
            wid = path.rsplit("/", 1)[-1]
            w = self.store.get_worker(wid)
            if w is None:
                return self._err(404, f"no worker {wid}")
            return self._ok(json.dumps(self._serialise_worker(w)).encode())
        return self._err(404, f"not found: {path}")

    def do_POST(self) -> None:                                # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/cancel/"):
            wid = path.rsplit("/", 1)[-1]
            try:
                r = self.client.cancel_worker(wid)
                return self._ok(json.dumps(r).encode())
            except ipc.IPCError as exc:
                return self._err(502, str(exc))
        return self._err(404, f"not found: {path}")

    def do_DELETE(self) -> None:                              # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/task/"):
            name = urllib.parse.unquote(path.rsplit("/", 1)[-1])
            try:
                r = self.client.delete_task(name)
                return self._ok(json.dumps(r).encode())
            except ipc.IPCError as exc:
                return self._err(502, str(exc))
        return self._err(404, f"not found: {path}")

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _serialise_worker(w) -> dict:
        d = asdict(w)
        d["total_tokens"] = w.total_tokens
        d["runtime_seconds"] = w.runtime_seconds
        return d

    def log_message(self, fmt: str, *args) -> None:           # noqa: ARG002
        # Quieter than the default; route through our logger.
        logger.debug("%s - %s", self.address_string(), fmt % args)


class _ThreadingHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    _Handler.store = Store()
    _Handler.client = ipc.Client()
    srv = _ThreadingHTTP((host, port), _Handler)
    print(f"MSA web UI on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    serve(host=host, port=port)
