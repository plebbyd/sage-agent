"""msa/worker.py — Per-task agentic loop running as its own subprocess.

A worker is given:
    * a row in the ``workers`` table (the supervisor created it)
    * a free-form ``prompt`` describing what to do
    * full access to the tool registry

Its job is to drive a ReAct loop until it emits ``respond`` (terminate
with a final answer to the user) or hits ``max_iterations`` /
``circuit_breaker_limit``.

Lifecycle:
    1. ``msa.worker --id w-XXXX`` is exec'd by the supervisor as a fresh
       Python process. Each worker thus has its own Ollama HTTP client
       and conversation history — perfectly isolated.
    2. The worker reads its prompt + master rules from the store + config.
    3. It loops:
         model.chat(history) → JSON action → dispatcher → record → repeat
    4. Every iteration appends rows to ``transcripts`` and ``events`` so
       the master and the web UI can render progress live.
    5. On ``respond``, it transitions the worker to ``completed`` and
       exits 0. On uncaught error, ``failed``. On SIGTERM/SIGINT,
       ``cancelled``.

Subprocess entry point:

    python -m msa.worker --id w-1234abcd
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .config import load_config, resolve_model_config
from .dispatcher import (
    CYCLE_RESULT_LIMIT,
    Dispatcher,
    DispatchResult,
    NOTES_RESULT_LIMIT,
    TOOL_DONE,
    TOOL_RESPOND,
    TOOL_SCRATCHPAD,
    clip,
)
from .model import ModelClient, ModelResponse
from .store import (
    EVT_CANCELLED,
    EVT_FAILED,
    EVT_FINISHED,
    EVT_LOG,
    EVT_PROGRESS,
    EVT_RESPONSE,
    EVT_STARTED,
    EVT_TOOL_CALL,
    EVT_TOOL_RESULT,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUNNING,
    Store,
)
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


CIRCUIT_BREAKER_LIMIT = 3   # consecutive parse / unknown-tool failures

# When a tool returns a result containing ``"skipped": true`` (currently
# only the ptz_move guards do this), the model is being told "stop".
# Some small models still ignore that and keep calling the tool, often
# interleaving harmless ptz_position reads to dodge a "consecutive"
# check. So we count TOTAL skips per worker run; after this many we
# synthesise a ``respond`` from the last meaningful tool result.
SKIP_BREAKER_LIMIT = 2


# ---------------------------------------------------------------------------
# Public entry — called by msa/__main__ and the subprocess
# ---------------------------------------------------------------------------

def run(worker_id: str,
        config_path: str = "config/config.yaml",
        is_master: bool = False) -> int:
    cfg = load_config(config_path)
    store = Store()
    record = store.get_worker(worker_id)
    if record is None:
        logger.error("worker %s not found in store; refusing to start",
                      worker_id)
        return 2
    if record.state in ("completed", "failed", "cancelled"):
        logger.info("worker %s already in terminal state %s; nothing to do",
                     worker_id, record.state)
        return 0

    _setup_logging(worker_id)
    return _Worker(worker_id, cfg, store, is_master=is_master).run()


# ---------------------------------------------------------------------------
# Logging — file per worker so the web UI / chat can `tail` cleanly
# ---------------------------------------------------------------------------

def _setup_logging(worker_id: str) -> None:
    logs_dir = Path.home() / ".msa" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / f"{worker_id}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    # also stream to stderr for systemd / supervisor capture
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    root.addHandler(sh)


# ---------------------------------------------------------------------------
# Worker implementation
# ---------------------------------------------------------------------------

class _Worker:
    def __init__(self, worker_id: str, cfg: dict, store: Store, *,
                  is_master: bool):
        self.id = worker_id
        self.cfg = cfg
        self.store = store
        self.is_master = is_master
        self.cancelled = False

        self.model = ModelClient(resolve_model_config(cfg))
        # Tool registry: master gets meta tools too. Pass `is_master`
        # via env so the tool plugins can opt in.
        os.environ["MSA_AGENT_ROLE"] = "master" if is_master else "worker"
        self.tools = ToolRegistry(cfg.get("tools", {}))
        self.dispatcher = Dispatcher(self.tools)

        self.system_prompt = self._load_system_prompt()
        self.max_iterations = int(
            self.store.get_worker(worker_id).max_iterations or
            cfg.get("max_iterations", 12)
        )

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    # ---- signal handler --------------------------------------------------

    def _handle_signal(self, signum, _frame) -> None:
        logger.info("worker %s received signal %s; cancelling", self.id, signum)
        self.cancelled = True

    @contextmanager
    def _heartbeat(self, *, iteration: int, phase: str, interval: float = 5.0):
        """Emit EVT_PROGRESS every `interval` seconds while the body runs.

        ``phase`` may be mutated by the body via ``state.phase`` so a
        tool-name swap (set in the on_tool_start callback) shows up in
        subsequent heartbeats. Without this the chat REPL just sits
        silent or shows a generic "tool" label, leaving the user with
        no idea what's actually running during a long call.
        """
        stop = threading.Event()
        start = time.time()
        state = type("HBState", (), {"phase": phase})()

        def loop():
            while not stop.wait(interval):
                self.store.emit_event(self.id, EVT_PROGRESS, {
                    "iteration": iteration,
                    "max_iterations": self.max_iterations,
                    "phase": state.phase,
                    "elapsed_s": round(time.time() - start, 1),
                })

        t = threading.Thread(target=loop, name=f"hb-{self.id}", daemon=True)
        t.start()
        try:
            yield state
        finally:
            stop.set()
            t.join(timeout=1.0)

    # ---- top-level loop --------------------------------------------------

    def run(self) -> int:
        worker = self.store.get_worker(self.id)
        assert worker is not None
        self.store.transition_worker(
            self.id, state=STATE_RUNNING, pid=os.getpid(),
        )
        self.store.append_transcript(self.id, "system", self.system_prompt)
        self.store.append_transcript(self.id, "user", worker.prompt)
        self.store.emit_event(self.id, EVT_STARTED, {
            "prompt": worker.prompt,
            "model": self.model.model,
            "backend": self.model.backend,
            "max_iterations": self.max_iterations,
            "spawned_by": worker.spawned_by,
            "parent_id": worker.parent_id,
            "is_master": self.is_master,
        })

        history: list[dict] = [{"role": "user", "content": worker.prompt}]
        consecutive_errors = 0
        total_skips = 0
        last_meaningful_result: str = ""  # last non-skipped tool result text
        scratchpad: dict = {"notes": "", "task": worker.prompt}
        cycle_actions: list[dict] = []
        last_response: str = ""

        try:
            for iteration in range(1, self.max_iterations + 1):
                if self.cancelled:
                    self._finish(STATE_CANCELLED,
                                  result=last_response or None,
                                  error="cancelled by signal")
                    self.store.emit_event(self.id, EVT_CANCELLED, {})
                    return 130

                self.store.increment_iterations(self.id)
                self.store.emit_event(self.id, EVT_PROGRESS, {
                    "iteration": iteration,
                    "max_iterations": self.max_iterations,
                })

                prompt = self._build_prompt(
                    iteration, worker.prompt, scratchpad, cycle_actions,
                )

                turn_history = history + [
                    {"role": "user", "content": prompt}
                ] if cycle_actions else history
                # On iteration 1 history already has the original prompt;
                # subsequent iterations add the synthesised prompt as the
                # latest user turn.
                if iteration > 1:
                    turn_history = history + [
                        {"role": "user", "content": prompt}
                    ]
                else:
                    turn_history = [
                        {"role": "user", "content": prompt}
                    ]

                with self._heartbeat(iteration=iteration, phase="thinking"):
                    resp = self._call_model(turn_history)
                if resp is None:
                    self._finish(STATE_FAILED, error="model unreachable")
                    self.store.emit_event(self.id, EVT_FAILED, {
                        "reason": "model unreachable",
                    })
                    return 1

                history.append({"role": "assistant", "content": resp.content})
                self.store.append_transcript(self.id, "assistant",
                                              resp.content, meta={
                                                  "prompt_tokens": resp.prompt_tokens,
                                                  "completion_tokens": resp.completion_tokens,
                                              })
                self.store.add_tokens(
                    self.id,
                    prompt_tokens=max(resp.prompt_tokens, 0),
                    completion_tokens=max(resp.completion_tokens, 0),
                )

                # Tool execution can block for many seconds (PTZ moves)
                # or minutes (ptz_scan w/ describe). Heartbeat through
                # it AND emit EVT_TOOL_CALL the moment we know which
                # tool is firing, so the user sees the tool name right
                # away rather than after it returns.
                with self._heartbeat(iteration=iteration, phase="tool") as hb:
                    def _on_tool_start(tool, args):
                        hb.phase = f"tool:{tool}"
                        self.store.emit_event(self.id, EVT_TOOL_CALL, {
                            "tool": tool, "args": args,
                        })
                    outcome = self.dispatcher.dispatch(
                        resp.content, on_tool_start=_on_tool_start,
                    )
                self._record_outcome(outcome, scratchpad, cycle_actions)

                if outcome.kind == "respond":
                    last_response = outcome.response_text or ""
                    self._finish(STATE_COMPLETED, result=last_response)
                    self.store.emit_event(self.id, EVT_FINISHED, {
                        "iterations": iteration,
                        "response": last_response,
                    })
                    return 0

                if outcome.kind in ("parse_error", "unknown_tool", "tool_error"):
                    consecutive_errors += 1
                    if consecutive_errors >= CIRCUIT_BREAKER_LIMIT:
                        msg = (f"circuit breaker tripped after "
                                f"{consecutive_errors} consecutive errors")
                        self._finish(STATE_FAILED, error=msg)
                        self.store.emit_event(self.id, EVT_FAILED, {
                            "reason": msg,
                        })
                        return 1
                else:
                    consecutive_errors = 0

                # Skip-loop circuit breaker: if any tool returns
                # ``"skipped": true`` more than SKIP_BREAKER_LIMIT times in
                # a single run (currently only the ptz_move guards do
                # this), the model is ignoring the "STOP" signal. End the
                # run with a synthesised respond from the last real
                # result. Skips don't have to be consecutive -- the model
                # can sneak harmless tool calls (ptz_position, etc) in
                # between, but the total cap still trips.
                tool_text = (outcome.result or "") if outcome.kind == "tool" else ""
                is_skip = bool(tool_text) and '"skipped": true' in tool_text
                if outcome.kind == "tool" and not is_skip:
                    last_meaningful_result = tool_text
                if is_skip:
                    total_skips += 1
                    if total_skips >= SKIP_BREAKER_LIMIT:
                        synth = (
                            "Stopped after repeated tool guard rejections. "
                            "Camera state from the last real tool call: "
                            + (last_meaningful_result[:600] or
                                "(no prior tool result captured)")
                        )
                        self._finish(STATE_COMPLETED, result=synth)
                        self.store.emit_event(self.id, EVT_RESPONSE, {
                            "response": synth,
                        })
                        self.store.emit_event(self.id, EVT_FINISHED, {
                            "iterations": iteration,
                            "response": synth,
                            "synthetic": True,
                        })
                        return 0

            # Out of iterations without `respond`.
            msg = "max_iterations reached without `respond`"
            last_response = (
                f"(worker exhausted {self.max_iterations} iterations)\n"
                + (scratchpad.get("notes") or "")[-2000:]
            )
            self._finish(STATE_FAILED, result=last_response, error=msg)
            self.store.emit_event(self.id, EVT_FAILED, {"reason": msg})
            return 1

        except Exception as exc:  # noqa: BLE001
            logger.exception("worker %s crashed", self.id)
            self._finish(STATE_FAILED, error=str(exc))
            self.store.emit_event(self.id, EVT_FAILED, {"reason": str(exc)})
            return 1

    # ---- helpers ---------------------------------------------------------

    def _call_model(self, messages: list[dict]) -> ModelResponse | None:
        for attempt in range(2):
            try:
                return self.model.chat(self.system_prompt, messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("model call failed (attempt %d/2): %s",
                                attempt + 1, exc)
                self.store.emit_event(self.id, EVT_LOG, {
                    "level": "warning",
                    "message": f"model call failed: {exc}",
                })
                time.sleep(2 * (attempt + 1))
        return None

    def _build_prompt(
        self, iteration: int, task: str,
        scratchpad: dict, cycle_actions: list[dict],
    ) -> str:
        parts: list[str] = []

        parts.append(f"Task: {task}")
        if scratchpad.get("notes"):
            parts.extend(["", "Notes so far:", scratchpad["notes"]])
        parts.extend(["", "Available tools:", self.tools.describe()])

        if cycle_actions:
            parts.extend(["", "Actions you have ALREADY taken this run:"])
            for i, a in enumerate(cycle_actions, 1):
                parts.append(
                    f"  {i}. {a['tool']}({a['args']}) →\n     {a['result']}"
                )
            parts.append("")
            parts.append(
                "Do NOT repeat any of the calls above. Take a DIFFERENT next "
                "action that advances the task, OR call `respond` if you "
                "have an answer."
            )
            # Synthesis trigger
            rich = [a for a in cycle_actions
                    if a["tool"] not in (TOOL_SCRATCHPAD, TOOL_RESPOND, TOOL_DONE)
                    and len(str(a.get("result") or "")) > 500]
            synthed = any(a["tool"] == TOOL_SCRATCHPAD for a in cycle_actions)
            if rich and not synthed:
                parts.extend([
                    "",
                    "SYNTHESIS REQUIRED: A previous tool call returned "
                    "substantive content. Your NEXT action must be "
                    "`update_scratchpad` writing a real synthesis (specific "
                    "things observed, distinct items, conclusions) into "
                    "notes. Then call `respond` with that synthesis as the "
                    "message.",
                ])
        else:
            parts.extend([
                "",
                "No actions taken yet. Pick the first concrete step.",
            ])

        parts.extend([
            "",
            f"Iteration {iteration}/{self.max_iterations}.",
            "Respond with a SINGLE JSON object: "
            '{"tool": "<name>", "args": {...}}',
        ])
        return "\n".join(parts)

    def _record_outcome(
        self, outcome: DispatchResult, scratchpad: dict,
        cycle_actions: list[dict],
    ) -> None:
        if outcome.kind == "parse_error":
            self.store.append_transcript(
                self.id, "event",
                f"[parse_error] {outcome.parse_error}",
                meta={"kind": "parse_error"},
            )
            self.store.emit_event(self.id, EVT_LOG, {
                "level": "warning",
                "message": "model output not parseable",
                "preview": outcome.parse_error,
            })
            # Surface the failure in cycle_actions so the next prompt
            # shows the model what went wrong. Without this, every
            # iteration sees an identical prompt and the model loops
            # on the same broken output until the circuit breaker fires.
            cycle_actions.append({
                "tool": "(none)",
                "args": {},
                "result": f"PARSE_ERROR: {outcome.parse_error}",
            })
            return

        if outcome.kind == "tool":
            self.store.append_transcript(
                self.id, "tool",
                clip(outcome.result or "", CYCLE_RESULT_LIMIT),
                meta={"name": outcome.tool, "args": outcome.args},
            )
            # EVT_TOOL_CALL was already emitted by on_tool_start before
            # the tool ran. Only emit the result here.
            self.store.emit_event(self.id, EVT_TOOL_RESULT, {
                "tool": outcome.tool,
                "preview": clip(outcome.result or "", 400),
                "length": len(outcome.result or ""),
            })
            cycle_actions.append({
                "tool": outcome.tool,
                "args": outcome.args,
                "result": clip(outcome.result or "", CYCLE_RESULT_LIMIT),
            })
            scratchpad["notes"] = (scratchpad.get("notes") or "") + (
                f"\n[tool] {outcome.tool}({outcome.args}) → "
                f"{clip(outcome.result or '', NOTES_RESULT_LIMIT)}"
            )
            return

        if outcome.kind == "scratchpad":
            for k, v in outcome.args.items():
                if k == "notes":
                    scratchpad["notes"] = str(v)
                else:
                    scratchpad[k] = v
            self.store.append_transcript(
                self.id, "event",
                f"[scratchpad] {outcome.args}",
                meta={"kind": "scratchpad"},
            )
            cycle_actions.append({
                "tool": TOOL_SCRATCHPAD,
                "args": outcome.args,
                "result": "ok",
            })
            return

        if outcome.kind == "tool_error":
            note = f"[tool_error] {outcome.tool}: {outcome.error}"
            self.store.append_transcript(self.id, "event", note,
                                          meta={"kind": "tool_error"})
            self.store.emit_event(self.id, EVT_LOG, {
                "level": "error", "message": note,
            })
            cycle_actions.append({
                "tool": outcome.tool, "args": outcome.args,
                "result": f"ERROR: {outcome.error}",
            })
            scratchpad["notes"] = (scratchpad.get("notes") or "") + f"\n{note}"
            return

        if outcome.kind == "unknown_tool":
            note = f"[unknown_tool] {outcome.tool}"
            self.store.append_transcript(self.id, "event", note,
                                          meta={"kind": "unknown_tool"})
            self.store.emit_event(self.id, EVT_LOG, {
                "level": "warning", "message": note,
            })
            cycle_actions.append({
                "tool": outcome.tool, "args": outcome.args,
                "result": f"ERROR: unknown tool {outcome.tool}",
            })
            return

        if outcome.kind == "respond":
            self.store.append_transcript(
                self.id, "assistant",
                outcome.response_text or "",
                meta={"kind": "respond"},
            )
            self.store.emit_event(self.id, EVT_RESPONSE, {
                "message": outcome.response_text or "",
            })

    def _finish(self, state: str, *, result: str | None = None,
                  error: str | None = None) -> None:
        self.store.transition_worker(
            self.id, state=state, result=result, error=error,
        )

    # ---- system prompt ---------------------------------------------------

    def _load_system_prompt(self) -> str:
        """Build the agent's system prompt.

        Only ``config/chat_rules.md`` (master) or ``config/worker_rules.md``
        (worker) is loaded by default. Per-plugin ``rules.md`` files in
        ``tools/`` and ``sensors/`` are NOT auto-included — they're
        legacy from the cycle-based agent and contain contradictory
        guidance ("signal done", tools that don't exist on this node).

        To opt back in for a specific plugin, set ``MSA_INCLUDE_PLUGIN_RULES=1``.
        """
        rules_dir = Path("config")
        which = "chat_rules.md" if self.is_master else "worker_rules.md"
        primary = rules_dir / which

        sections: list[str] = []
        if primary.exists():
            sections.append(primary.read_text())
        else:
            sections.append(
                "You are an MSA agent. Respond with a single JSON object "
                '{"tool": "...", "args": {...}}.'
            )

        if os.environ.get("MSA_INCLUDE_PLUGIN_RULES") == "1":
            for sub in ("tools", "sensors"):
                d = Path(sub)
                if d.is_dir():
                    for f in sorted(d.glob("*rules*.md")):
                        if f.name == which:
                            continue
                        try:
                            sections.append(
                                f"\n---\n[{f}]\n\n{f.read_text()}"
                            )
                        except OSError:
                            pass

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m msa.worker")
    p.add_argument("--id", required=True, help="worker id (must exist in store)")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--master", action="store_true",
                    help="run with master rules and meta tools enabled")
    args = p.parse_args(argv)
    return run(args.id, config_path=args.config, is_master=args.master)


if __name__ == "__main__":
    sys.exit(main())
