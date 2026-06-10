"""
MSA — Minimal Synthetic Agent

Core modules:
    agent      — wake/sense/run/sleep loop and CLI
    model      — multi-backend LLM client
    dispatcher — JSON parsing and tool routing
    scratchpad — persistent YAML memory with introspection
    tools      — tool registry with plugin discovery
    sensors    — sensor registry with plugin discovery
    tasks      — cron/interval task scheduler
    plugins    — shared plugin discovery engine
    config     — YAML config loader with deep merge
    scheduler  — trigger modes (interval, file_watch, slack)
"""
