#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study_root="$(cd "${script_dir}/../.." && pwd)"
workers="${BASELINE_WORKERS:-3}"
run_label="${1:-baseline_grid_20260817}"
log_dir="${study_root}/outputs/latent_factor_beyond_rmse/logs"
log_path="${log_dir}/${run_label}.log"

mkdir -p "${log_dir}"
cd "${study_root}"
exec > >(tee -a "${log_path}") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Baseline grid start; workers=${workers}"
for dataset in ciaodvd yelp_tampa_restaurants movielens_1m; do
  for retention in r100 r075 r050 r025; do
    python -m src.latent_factor_study.baseline_grid \
      --dataset "${dataset}" \
      --retention "${retention}" \
      --workers "${workers}"
  done
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Baseline grid complete"
