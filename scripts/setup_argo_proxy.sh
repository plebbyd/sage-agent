#!/usr/bin/env bash
# setup_argo_proxy.sh — argo-proxy bring-up for jetson-ptz-agent-graph (agent_pjl).
#
# Usage (on the node, e.g. sgt-node-H00F):
#   bash scripts/setup_argo_proxy.sh --username YOUR_ANL_USER --model claude-opus-4.6
#
# Test only:
#   bash scripts/setup_argo_proxy.sh test
#
# Switch back to Ollama:
#   bash scripts/setup_argo_proxy.sh disable

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -t 1 ]]; then
    GREEN=$'\033[32m'; RED=$'\033[31m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    GREEN=""; RED=""; BLUE=""; RESET=""
fi
log() { printf "%s==>%s %s\n" "$BLUE" "$RESET" "$*"; }
ok()  { printf "%s✓%s  %s\n" "$GREEN" "$RESET" "$*"; }
err() { printf "%s✗%s  %s\n" "$RED" "$RESET" "$*" >&2; }

# argo-proxy requires Python 3.10+ (https://argo-proxy.readthedocs.io/en/latest/usage/installation/)
find_python310() {
    local cmd ver
    for cmd in python3.12 python3.11 python3.10 python3; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            continue
        fi
        if "$cmd" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

PYTHON=""
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    if "$PROJECT_ROOT/.venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
    fi
fi
if [[ -z "$PYTHON" ]]; then
    PYTHON="$(find_python310)" || true
fi
if [[ -z "$PYTHON" ]]; then
    err "Python 3.10+ is required (argo-proxy). On this host, python3 is:"
    python3 --version 2>&1 >&2 || err "  (python3 not found)"
    err ""
    err "This node only has Python 3.6 — bootstrap Python 3.11 with micromamba:"
    err "  bash scripts/bootstrap_python311.sh"
    err "  source .venv/bin/activate"
    err "  bash scripts/setup_argo_proxy.sh -u USER -m gpt-4o"
    exit 1
fi

log "Using $PYTHON ($($PYTHON --version 2>&1))"

# Ensure venv CLI tools (argo-proxy) are on PATH when using project .venv
if [[ "$PYTHON" == "$PROJECT_ROOT/.venv/bin/python" ]]; then
    export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
fi

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # Optional: not required because PYTHON/PATH are set above
    :
fi

if ! "$PYTHON" -c "import yaml" 2>/dev/null; then
    log "Installing PyYAML …"
    "$PYTHON" -m pip install -q pyyaml
fi

if [[ "${1:-}" == "test" ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/argo_proxy_setup.py" test
fi
if [[ "${1:-}" == "disable" ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/argo_proxy_setup.py" disable
fi
if [[ "${1:-}" == "check-network" ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/argo_proxy_setup.py" check-network
fi

log "jetson-ptz-agent-graph argo-proxy setup ($PROJECT_ROOT)"
"$PYTHON" "$SCRIPT_DIR/argo_proxy_setup.py" setup "$@"
ok "Done. Run: cd $PROJECT_ROOT && python3 -m ptz_node --help"
