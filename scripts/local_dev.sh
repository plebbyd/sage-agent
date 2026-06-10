#!/usr/bin/env bash
# Local Mac development — venv, doctor, argo-proxy tunnel, optional API + test.
#
#   bash scripts/local_dev.sh              # doctor (argo-proxy via node-V010)
#   bash scripts/local_dev.sh argo -u USER # configure + SSH tunnel to node-V010
#   bash scripts/local_dev.sh argo-test    # test argo-proxy LLM
#   bash scripts/local_dev.sh smoke        # PTZ gateway only (no LLM)
#   bash scripts/local_dev.sh api          # REST gateway :8848
#   bash scripts/local_dev.sh test         # flagship agentic test case

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export PTZ_GRAPH_CONFIG="${PTZ_GRAPH_CONFIG:-$ROOT/config/local.yaml}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MSA_PTZ_BACKEND="${MSA_PTZ_BACKEND:-sim}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt

if [[ "${INSTALL_VISION:-1}" != "0" ]]; then
  echo "==> Installing vision backends (YOLO, BioCLIP deps) …"
  pip install -q -r requirements-vision.txt
fi

if [[ "${INSTALL_ARGO:-1}" != "0" ]]; then
  echo "==> Installing argo-proxy CLI …"
  pip install -q -r requirements-argo.txt
fi

MODE="${1:-doctor}"
shift || true

case "$MODE" in
  doctor)
    python -m ptz_node doctor
    ;;
  argo)
    exec bash scripts/setup_argo_proxy.sh "$@"
    ;;
  argo-test)
    exec bash scripts/setup_argo_proxy.sh test
    ;;
  argo-disable)
    exec bash scripts/setup_argo_proxy.sh disable
    ;;
  run)
    python -m ptz_node doctor
    python -m ptz_node devices
    ;;
  smoke)
    python -m ptz_node gateway-smoke
    ;;
  api)
    python -m ptz_node doctor
    echo "Gateway API http://127.0.0.1:8848/v1/health"
    echo "Debug: http://127.0.0.1:8848/v1/debug/doctor"
    exec uvicorn ptz_node.api_server:app --host 127.0.0.1 --port 8848 --reload
    ;;
  test)
    python -m ptz_node doctor
    python -m ptz_node test --id ptz_multimodel_scientific_survey
    ;;
  test-all)
    python -m ptz_node doctor
    python -m ptz_node test --all
    ;;
  *)
    echo "Usage: $0 [doctor|argo|argo-test|argo-disable|run|smoke|api|test|test-all]" >&2
    echo "  argo: pass -u USER -m gpt-4o --jump node-V010 to setup script" >&2
    exit 2
    ;;
esac
