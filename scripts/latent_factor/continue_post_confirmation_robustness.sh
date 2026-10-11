#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 CURRENT_PID [RUN_LABEL]" >&2
  exit 2
fi

current_pid="$1"
run_label="${2:-post_confirmation_robustness_20260820}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while kill -0 "${current_pid}" 2>/dev/null; do
  sleep 30
done

exec "${script_dir}/run_post_confirmation_robustness.sh" "${run_label}"
