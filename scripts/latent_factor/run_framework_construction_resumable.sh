#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${script_dir}/../.." && pwd)"
RUN_LABEL="${1:-framework_construction_20260825}"
LOG_DIR="$REPO_ROOT/outputs/latent_factor_beyond_rmse/framework_construction/logs"
LOG_PATH="$LOG_DIR/$RUN_LABEL.log"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
exec >>"$LOG_PATH" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] framework construction confirmation started"
python -m src.latent_factor_study.framework_construction select
python -m src.latent_factor_study.framework_construction confirm
python -m src.latent_factor_study.framework_construction analyze
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] framework construction confirmation completed"
