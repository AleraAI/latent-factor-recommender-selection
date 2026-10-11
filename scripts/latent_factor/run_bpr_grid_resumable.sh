#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study_root="$(cd "${script_dir}/../.." && pwd)"
workers="${BPR_WORKERS:-6}"
run_label="${1:-bpr_grid_20260817}"
log_dir="${study_root}/outputs/latent_factor_beyond_rmse/logs"
log_path="${log_dir}/${run_label}.log"

mkdir -p "${log_dir}"
cd "${study_root}"
exec > >(tee -a "${log_path}") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BPR grid start; workers=${workers}"
for dataset in ciaodvd yelp_tampa_restaurants movielens_1m; do
  for retention in r100 r075 r050 r025; do
    python -m src.latent_factor_study.bpr_grid \
      --dataset "${dataset}" \
      --retention "${retention}" \
      --workers "${workers}"
  done
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BPR grid complete"
