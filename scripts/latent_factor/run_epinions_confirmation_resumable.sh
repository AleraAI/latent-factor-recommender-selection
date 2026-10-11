#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${script_dir}/../.." && pwd)"
RUN_LABEL="${1:-epinions_confirmation_20260825}"
LOG_DIR="$REPO_ROOT/outputs/latent_factor_beyond_rmse/prospective_epinions/logs"
MAIN_LOG="$LOG_DIR/$RUN_LABEL.log"
CONFIG="configs/latent_factor_beyond_rmse/epinions_prospective_confirmation.yaml"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
exec >>"$MAIN_LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] prospective confirmation started"
retentions=(r100 r075 r050 r025)
pids=()
for retention in "${retentions[@]}"; do
  retention_log="$LOG_DIR/${RUN_LABEL}_${retention}.log"
  python -m src.latent_factor_study.prospective_confirmation \
    --config "$CONFIG" \
    --retention "$retention" \
    >"$retention_log" 2>&1 &
  retention_pid="$!"
  pids+=("$retention_pid")
  echo "started $retention as PID $retention_pid"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${retentions[$index]}"
  else
    echo "failed ${retentions[$index]}"
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "prospective confirmation did not complete; inference not started"
  exit 1
fi

python -m src.latent_factor_study.prospective_inference --config "$CONFIG"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] prospective confirmation and inference completed"
