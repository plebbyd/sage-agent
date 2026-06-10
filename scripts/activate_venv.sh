#!/usr/bin/env bash
# activate_venv.sh — use the project .venv (micromamba or std venv) in bash.
# Usage: source scripts/activate_venv.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_PROJECT_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
_VENV="$_PROJECT_ROOT/.venv"

if [[ ! -x "$_VENV/bin/python" ]]; then
    echo "No .venv found. Run: bash scripts/bootstrap_python311.sh" >&2
    return 1 2>/dev/null || exit 1
fi

export VIRTUAL_ENV="$_VENV"
export PATH="$_VENV/bin:${PATH:-}"
unset _SCRIPT_DIR _PROJECT_ROOT _VENV

echo "Activated $(python --version 2>&1) from $VIRTUAL_ENV"
