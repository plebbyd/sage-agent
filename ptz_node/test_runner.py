"""Run Sage-style agentic test cases with structured artifacts under .local/tests/."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage

from ptz_node.debug_report import RunRecorder, format_doctor_text, run_doctor
from ptz_node.graph_runner import build_agent_executor, extract_trace, summarize_messages
from ptz_node.paths import debug_dir, stamp, tests_dir, write_json


def _cases_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "agentic_test_cases.yaml"


def load_test_cases(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _cases_path()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cases = raw.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError("agentic_test_cases.yaml: cases must be a list")
    return cases


def _tool_names(trace: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for step in trace:
        if step.get("kind") == "tool_call":
            names.add(str(step.get("name", "")))
        if step.get("kind") == "tool_result" and step.get("name"):
            names.add(str(step["name"]))
    return names


def _trace_text(trace: list[dict[str, Any]]) -> str:
    return json.dumps(trace, default=str).lower()


def evaluate_case(case: dict[str, Any], trace: list[dict[str, Any]],
                  reply: str, doctor: dict[str, Any]) -> dict[str, Any]:
    tools_used = _tool_names(trace)
    text_blob = _trace_text(trace) + " " + (reply or "").lower()
    missing_tools = [
        t for t in (case.get("expect_tools") or []) if t not in tools_used
    ]
    missing_detectors = []
    for det in case.get("expect_detectors") or []:
        if not doctor.get("detectors", {}).get(det):
            missing_detectors.append(det)
    missing_keywords = [
        kw for kw in (case.get("expect_keywords_in_reply") or [])
        if kw.lower() not in (reply or "").lower()
        and kw.lower() not in text_blob
    ]
    passed = not missing_tools and not missing_detectors and not missing_keywords
    return {
        "passed": passed,
        "tools_used": sorted(tools_used),
        "missing_tools": missing_tools,
        "missing_detectors": missing_detectors,
        "missing_keywords": missing_keywords,
    }


def run_test_case(case_id: str, *, cfg: dict[str, Any],
                  config_path: str | None = None,
                  limit: int = 80) -> dict[str, Any]:
    cases = {c["id"]: c for c in load_test_cases()}
    if case_id not in cases:
        raise KeyError(f"unknown test case {case_id!r}; choices: {sorted(cases)}")
    case = cases[case_id]

    doctor = run_doctor(cfg)
    required = case.get("require_doctor_ok", False)
    if required and not doctor.get("ok"):
        result = {
            "case_id": case_id,
            "title": case.get("title"),
            "passed": False,
            "skipped": True,
            "reason": "doctor preflight failed",
            "doctor_ok": False,
        }
        out_dir = tests_dir() / f"{case_id}_{stamp()}"
        write_json(out_dir / "result.json", result)
        return result

    for det in case.get("expect_detectors") or []:
        if not doctor.get("detectors", {}).get(det):
            result = {
                "case_id": case_id,
                "title": case.get("title"),
                "passed": False,
                "skipped": True,
                "reason": f"detector {det!r} not available on this machine",
                "doctor_ok": doctor.get("ok"),
            }
            out_dir = tests_dir() / f"{case_id}_{stamp()}"
            write_json(out_dir / "result.json", result)
            return result

    prompt = (case.get("prompt") or "").strip()
    graph, _gw = build_agent_executor(cfg)
    rec = RunRecorder(prompt=prompt, config_path=config_path,
                      run_id=f"test_{case_id}_{stamp()}")
    t0 = time.time()
    try:
        out = graph.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": int(limit)},
        )
        duration = time.time() - t0
        messages = out.get("messages") or []
        rec.finish_success(messages, duration_s=duration)
        trace = extract_trace(messages)
        reply = summarize_messages(messages)
    except Exception as exc:
        duration = time.time() - t0
        rec.finish_error(exc, duration_s=duration)
        trace = rec.trace
        reply = ""

    evaluation = evaluate_case(case, trace, reply, doctor)
    result = {
        "case_id": case_id,
        "title": case.get("title"),
        "sage_domain": case.get("sage_domain"),
        "passed": evaluation["passed"],
        "skipped": False,
        "evaluation": evaluation,
        "run_dir": str(rec.run_dir),
        "summary_txt": str(rec.run_dir / "summary.txt"),
        "doctor_ok": doctor.get("ok"),
        "duration_s": round(duration, 3),
    }

    out_dir = tests_dir() / f"{case_id}_{stamp()}"
    write_json(out_dir / "result.json", result)
    write_json(out_dir / "evaluation.json", evaluation)
    write_json(debug_dir() / "latest_test.json", result)
    return result


def run_all_tests(*, cfg: dict[str, Any], config_path: str | None = None,
                  limit: int = 80) -> dict[str, Any]:
    doctor = run_doctor(cfg)
    results = []
    for case in load_test_cases():
        results.append(run_test_case(
            case["id"], cfg=cfg, config_path=config_path, limit=limit,
        ))
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "doctor_ok": doctor.get("ok"),
        "doctor_text": format_doctor_text(doctor),
        "total": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "failed": sum(
            1 for r in results if not r.get("passed") and not r.get("skipped")
        ),
        "results": results,
        "cursor_hint": f"See {tests_dir()} and {debug_dir() / 'latest_test.json'}",
    }
    write_json(tests_dir() / "latest_summary.json", summary)
    write_json(debug_dir() / "latest_test_summary.json", summary)
    return summary
