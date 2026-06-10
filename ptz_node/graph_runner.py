"""LangGraph prebuilt agent wiring."""

from __future__ import annotations

from typing import Any

from langgraph.prebuilt import create_react_agent

from ptz_node.langchain_tools import build_gateway_tools
from ptz_node.config_loader import resolve_model_config
from ptz_node.llm_factory import chat_model_from_config


def build_agent_executor(cfg: dict[str, Any]):
    llm = chat_model_from_config(resolve_model_config(cfg))
    from ptz_node.sensor_gateway import SensorGateway

    gateway = SensorGateway(cfg)
    tools = build_gateway_tools(gateway)
    system_prompt = (
        "You are a Sage edge-node agent (Jetson Thor class hardware). "
        "Collect and interpret sensor data for scientific users: wildfire monitoring, "
        "ecosystem/biodiversity surveys, agriculture, and urban/environmental science. "
        "Never touch hardware directly — only use the provided tools, which route "
        "through the node's sensor-management gateway. "
        "ALWAYS start with sensor_list_devices to see what is attached, then use "
        "sensor_capabilities/sensor_read for non-camera sensors and sensor_invoke for "
        "any capability that lacks a typed tool. "
        "For PTZ cameras prefer the typed tools and check detector_status when vision "
        "is involved. PTZ workflow: read position, snapshot or detect, move deliberately, "
        "then compare views. Vision backends: ptz_detect model=yolo (objects), "
        "model=bioclip (species/taxa via target_taxon), model=gemma4 (semantic scenes). "
        "Use ptz_caption for narrative scene descriptions. "
        "Summarize findings with counts, taxa, and confidence; note model limitations."
    )

    graph = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
        name="jetson_ptz_gateway_agent",
    )
    return graph, gateway


def summarize_messages(messages) -> str:
    from langchain_core.messages import AIMessage

    for m in reversed(messages or []):
        if isinstance(m, AIMessage):
            txt = getattr(m, "content", "")
            if isinstance(txt, list):
                parts = []
                for b in txt:
                    if isinstance(b, dict) and "text" in b:
                        parts.append(b["text"])
                    else:
                        parts.append(str(b))
                return "\n".join(parts).strip()
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    return ""


def extract_trace(messages) -> list[dict]:
    """Structured steps for .local/runs traces (Cursor-friendly)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    steps: list[dict] = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            steps.append({"kind": "human", "content": str(m.content)[:2000]})
        elif isinstance(m, AIMessage):
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    steps.append({
                        "kind": "tool_call",
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                        "id": tc.get("id"),
                    })
            content = m.content
            if content and not getattr(m, "tool_calls", None):
                steps.append({"kind": "assistant", "content": str(content)[:4000]})
        elif isinstance(m, ToolMessage):
            steps.append({
                "kind": "tool_result",
                "name": getattr(m, "name", None),
                "tool_call_id": getattr(m, "tool_call_id", None),
                "content": str(m.content)[:8000],
            })
    return steps

