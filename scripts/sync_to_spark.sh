#!/usr/bin/env bash
# sync_to_spark.sh — copy this repo to the DGX Spark, skipping temp/data/pyc files.
#
# scp -r copies EVERYTHING (including .venv/, .local/, logs/, __pycache__/...).
# rsync can mirror the repo while honoring .gitignore, so junk never leaves your machine.
#
# Usage:
#   bash scripts/sync_to_spark.sh                 # sync to default host/path below
#   bash scripts/sync_to_spark.sh user@host:/path # override destination
#   DRY_RUN=1 bash scripts/sync_to_spark.sh        # preview what would transfer

set -euo pipefail

DEST="${1:-plebbyd@spark-7d68.local:/home/plebbyd/sage-agent}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

RSYNC_OPTS=(-az --delete --human-readable --progress)

# Honor .gitignore (per-directory rules, including negations) and never ship .git.
RSYNC_OPTS+=(--filter=":- .gitignore" --exclude='.git/')

# PROTECT node-side runtime state from --delete. These live ONLY on the node
# (created by bootstrap/runs), are absent from the source, and would otherwise be
# deleted on every re-sync — forcing a full venv reinstall each deploy. `P` =
# protect (receiver-side) and is stronger/clearer than relying on exclude rules.
RSYNC_OPTS+=(
    --filter='P .venv/'
    --filter='P .micromamba/'
    --filter='P .local/'
    --filter='P logs/'
    --filter='P scratchpads/'
    --filter='P config/argo_proxy.local.yaml'
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    RSYNC_OPTS+=(--dry-run --itemize-changes)
    echo "[dry-run] no files will be transferred"
fi

echo "Syncing $SRC_DIR/  ->  $DEST"
rsync "${RSYNC_OPTS[@]}" "$SRC_DIR/" "$DEST"
echo "Done."
