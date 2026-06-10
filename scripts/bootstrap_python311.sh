#!/usr/bin/env bash
# bootstrap_python311.sh — Python 3.11 via micromamba when system python3 is too old.
#
# Sage / edge nodes often ship python3.6; argo-proxy needs 3.10+.
# Creates PROJECT_ROOT/.venv with Python 3.11 (conda-forge, user-local).
#
# Usage:
#   bash scripts/bootstrap_python311.sh
#   source scripts/activate_venv.sh
#   bash scripts/setup_argo_proxy.sh -u USER -m gpt-4o
#
# Or skip activate — setup_argo_proxy.sh auto-uses .venv/bin/python:
#   bash scripts/setup_argo_proxy.sh -u USER -m gpt-4o

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    if "$PROJECT_ROOT/.venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "OK: .venv already has $($PROJECT_ROOT/.venv/bin/python --version 2>&1)"
        exit 0
    fi
    echo "Removing stale .venv (python < 3.10) …"
    rm -rf "$PROJECT_ROOT/.venv"
fi

for cmd in python3.12 python3.11 python3.10; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
            echo "Found system $cmd — creating venv …"
            "$cmd" -m venv "$PROJECT_ROOT/.venv"
            "$PROJECT_ROOT/.venv/bin/pip" install -q --upgrade pip pyyaml
            echo "Done. Run: source $PROJECT_ROOT/.venv/bin/activate"
            exit 0
        fi
    fi
done

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64) MAMBA_ARCH="linux-64" ;;
    aarch64|arm64) MAMBA_ARCH="linux-aarch64" ;;
    *)
        echo "Unsupported CPU arch: $ARCH (need x86_64 or aarch64)" >&2
        exit 1
        ;;
esac

MAMBA_BIN="$PROJECT_ROOT/.micromamba/bin/micromamba"
if [[ ! -x "$MAMBA_BIN" ]]; then
    echo "==> Downloading micromamba ($MAMBA_ARCH) …"
    mkdir -p "$PROJECT_ROOT/.micromamba/bin"
    curl -fsSL "https://micro.mamba.pm/api/micromamba/${MAMBA_ARCH}/latest" \
        | tar -xj -C "$PROJECT_ROOT/.micromamba" bin/micromamba
fi

export MAMBA_ROOT_PREFIX="$PROJECT_ROOT/.micromamba/root"
export MAMBA_EXE="$MAMBA_BIN"

echo "==> Creating .venv with Python 3.11 (this may take a few minutes) …"
"$MAMBA_BIN" create -y -p "$PROJECT_ROOT/.venv" \
    python=3.11 pip pyyaml -c conda-forge

# conda envs expose python under bin/ — ensure pip works
"$PROJECT_ROOT/.venv/bin/python" -m pip install -q --upgrade pip

# micromamba envs do not ship bin/activate — add a minimal one for bash
cat > "$PROJECT_ROOT/.venv/bin/activate" << 'ACTIVATE'
# Minimal activate for micromamba/conda .venv (source this file)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV="$_VENV"
PATH="$_VENV/bin:${PATH:-}"
export PATH
unset _VENV
ACTIVATE

echo ""
echo "OK: $($PROJECT_ROOT/.venv/bin/python --version 2>&1) at $PROJECT_ROOT/.venv"
echo "Next (pick one):"
echo "  source scripts/activate_venv.sh"
echo "  bash scripts/setup_argo_proxy.sh -u YOUR_ANL_USER -m gpt-4o   # no activate needed"
