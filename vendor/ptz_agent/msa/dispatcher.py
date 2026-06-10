"""msa/dispatcher.py — Parses one model response and runs the action it asks for.

Each model response is one JSON object:

    {"tool": "tool_name", "args": {...}}

Special "tools" that don't live in the registry:

    "respond"          — final message to the user; ends the inner loop
    "done"             — legacy alias for respond (kept for backwards compat)
    "update_scratchpad"— mutates the worker's scratchpad/notes
    "schedule_task"    — handled by master-only meta tools (registered as a
                          regular tool by the master agent)

Robust JSON parsing handles markdown fences, prose around the JSON,
unescaped newlines inside string values, and nested ``{...}``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Tool-name constants (importable so callers don't reach into strings).
TOOL_RESPOND = "respond"
TOOL_DONE = "done"           # alias of respond
TOOL_SCRATCHPAD = "update_scratchpad"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Structured outcome of a single dispatch step.

    ``finished`` is True when the model emitted ``respond`` / ``done``,
    indicating the inner ReAct loop should terminate.

    ``response_text`` is set on a successful ``respond``. Workers / chat
    use this to surface a final answer to the user.

    ``parse_error`` is set when the model output couldn't be coerced
    into a JSON action even after every fallback. Callers may want to
    increment a circuit-breaker counter.
    """

    kind: str                                  # parse_error|tool|respond|scratchpad|unknown_tool|tool_error
    finished: bool = False
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    result: Optional[str] = None
    response_text: Optional[str] = None
    parse_error: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Truncation budgets for tool results echoed back to the model. Macro
# tools (``ptz_scan`` with N captioned stops) emit multi-KB JSON; the
# old 300/500 char limits starved the synthesis step.
NOTES_RESULT_LIMIT = 1200
CYCLE_RESULT_LIMIT = 8000


class Dispatcher:
    def __init__(self, tool_registry):
        self.tools = tool_registry

    def dispatch(
        self,
        response: str,
        *,
        on_tool_start=None,
    ) -> DispatchResult:
        """Parse a model response and execute its action.

        ``on_tool_start(tool_name, args)`` is fired immediately BEFORE
        a registry tool is invoked. Workers use this to surface the
        tool name to the chat REPL so the user knows what's running
        during long calls (PTZ moves, panorama captioning).
        """
        action = self._parse_response(response)
        if action is None:
            preview = response.strip()[:300] if response else ""
            return DispatchResult(
                kind="parse_error",
                parse_error=(
                    "Your last response was not valid JSON, or had no "
                    '"tool" field. Respond with EXACTLY one object like: '
                    '{"tool": "shell", "args": {"command": "date"}}. '
                    f'You sent: {preview}'
                ),
            )

        tool_name = action.get("tool", "")
        raw_args = action.get("args", {})

        # Common small-model failure: emitting `"args": "date"` instead of
        # `"args": {"command": "date"}`. Surface a precise correction
        # instead of looping on parse errors.
        if not isinstance(raw_args, dict):
            return DispatchResult(
                kind="tool_error",
                tool=tool_name,
                args={},
                error=(
                    f'`args` must be a JSON object, got '
                    f'{type(raw_args).__name__} ({raw_args!r}). Example: '
                    f'{{"tool": "{tool_name}", "args": {{"command": "..."}}}}'
                ),
            )
        args = raw_args

        # ---- respond / done: terminate the inner loop ------------------
        if tool_name in (TOOL_RESPOND, TOOL_DONE):
            text = (
                args.get("message")
                or args.get("text")
                or args.get("summary")
                or args.get("response")
                or ""
            )
            return DispatchResult(
                kind="respond",
                finished=True,
                tool=tool_name,
                args=args,
                response_text=str(text),
            )

        # ---- scratchpad: silent state update ---------------------------
        if tool_name == TOOL_SCRATCHPAD:
            return DispatchResult(
                kind="scratchpad",
                tool=tool_name,
                args=args,
            )

        # ---- registry tool ---------------------------------------------
        if self.tools.has(tool_name):
            if on_tool_start is not None:
                try:
                    on_tool_start(tool_name, args)
                except Exception:  # noqa: BLE001
                    logger.exception("on_tool_start callback failed")
            try:
                result = self.tools.call(tool_name, args)
            except Exception as exc:  # noqa: BLE001
                logger.exception("tool %s failed", tool_name)
                return DispatchResult(
                    kind="tool_error",
                    tool=tool_name,
                    args=args,
                    error=str(exc),
                )
            return DispatchResult(
                kind="tool",
                tool=tool_name,
                args=args,
                result=str(result),
            )

        return DispatchResult(
            kind="unknown_tool",
            tool=tool_name,
            args=args,
            error=f"unknown tool: {tool_name!r}",
        )

    # -----------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------

    def _parse_response(self, response: str) -> dict | None:
        """Permissive JSON extraction.

        Strategies, tried in order:
          1. ``json.loads`` raw
          2. strip ```json``` / ``` ``` / `` ` `` fences and retry
          3. greedy ``{.*}`` regex
          4. brace-balanced scan ignoring braces inside string literals
          5. last-ditch repair: escape literal newlines/tabs inside strings
        """
        if not response or not response.strip():
            return None

        candidates: list[str] = [response.strip()]
        stripped = response.strip()

        fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)\n?```",
                          stripped, re.DOTALL)
        if fence:
            candidates.append(fence.group(1).strip())
        if stripped.startswith("`") and stripped.endswith("`"):
            candidates.append(stripped.strip("`").strip())

        greedy = re.search(r"\{.*\}", stripped, re.DOTALL)
        if greedy:
            candidates.append(greedy.group())

        balanced = self._balanced_object(stripped)
        if balanced:
            candidates.append(balanced)

        seen: set[str] = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            obj = self._try_load(cand)
            if obj is None:
                repaired = self._repair_unescaped_strings(cand)
                if repaired and repaired != cand:
                    obj = self._try_load(repaired)
            if obj is None or not isinstance(obj, dict):
                continue
            if not isinstance(obj.get("tool"), str) or not obj["tool"]:
                continue
            # Note: args may be wrong-shaped here. We accept the action
            # and let dispatch() return a precise tool_error so the
            # model gets actionable feedback (rather than an opaque
            # parse error that it can't learn from).
            obj.setdefault("args", {})
            return obj
        return None

    @staticmethod
    def _try_load(s: str) -> Any:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _balanced_object(s: str) -> str | None:
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    @staticmethod
    def _repair_unescaped_strings(s: str) -> str:
        out: list[str] = []
        in_str = False
        escape = False
        for ch in s:
            if in_str:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    in_str = False
                    out.append(ch)
                    continue
                if ch == "\n":
                    out.append("\\n"); continue
                if ch == "\r":
                    out.append("\\r"); continue
                if ch == "\t":
                    out.append("\\t"); continue
                out.append(ch)
            else:
                if ch == '"':
                    in_str = True
                out.append(ch)
        return "".join(out)


# ---------------------------------------------------------------------------
# Truncation helper used by callers that build prompts.
# ---------------------------------------------------------------------------

def clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated, {len(s) - limit} chars omitted]"
