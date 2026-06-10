"""Resolve and prepend the canonical ``ptz-agent`` tree so ``tools.ptz_facade`` imports work."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def graph_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _vendored_ptz_agent() -> Path:
    """Self-contained copy created by ``scripts/vendor_ptz_agent.sh``."""
    return graph_repo_root() / "vendor" / "ptz_agent"


def _default_sibling_ptz_agent() -> Path:
    """When this repo sits next to ``ptz-agent`` under a common parent (e.g. MSA-main)."""
    return graph_repo_root().parent / "ptz-agent"


def _looks_like_ptz_agent(p: Path) -> bool:
    """A valid ptz-agent root must expose the camera facade we bridge to."""
    return (p / "tools" / "ptz_facade.py").is_file()


def resolved_ptz_agent_root(explicit: str | None = None) -> Path:
    # Priority: explicit/env overrides (for dev against a live checkout), then the
    # vendored self-contained copy, then a sibling ../ptz-agent.
    tried: list[tuple[str, Path, str]] = []
    candidates = (
        ("explicit/config", explicit),
        ("PTZ_AGENT_ROOT", os.environ.get("PTZ_AGENT_ROOT")),
        ("MSA_PTZ_AGENT_ROOT", os.environ.get("MSA_PTZ_AGENT_ROOT")),
        ("vendored vendor/ptz_agent", str(_vendored_ptz_agent())),
        ("sibling ../ptz-agent", str(_default_sibling_ptz_agent())),
    )
    for label, raw in candidates:
        if not (raw and str(raw).strip()):
            continue
        p = Path(raw).expanduser().resolve()
        if not p.is_dir():
            tried.append((label, p, "not a directory"))
            continue
        if not _looks_like_ptz_agent(p):
            tried.append((label, p, "missing tools/ptz_facade.py"))
            continue
        return p

    detail = "; ".join(f"{lbl}={p} ({why})" for lbl, p, why in tried) or "none provided"
    raise FileNotFoundError(
        "Could not locate a valid ptz-agent checkout (the folder containing "
        "`tools/ptz_facade.py`). Set PTZ_AGENT_ROOT or config ptz_agent_root to it. "
        f"Tried: {detail}"
    )


def bootstrap_ptz_agent_runtime(configured_root: str | None = None) -> Path:
    """Insert ``ptz-agent`` on ``sys.path`` so ``tools.*`` resolves.

    Returns the resolved PTZ-agent root ``Path``.
    """
    root = resolved_ptz_agent_root(configured_root)
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    return root
