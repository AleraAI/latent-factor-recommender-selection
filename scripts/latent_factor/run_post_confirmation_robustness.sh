#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study_root="$(cd "${script_dir}/../.." && pwd)"
workers="${ROBUSTNESS_WORKERS:-5}"
run_label="${1:-post_confirmation_robustness_20260820}"
log_dir="${study_root}/outputs/latent_factor_beyond_rmse/logs"
log_path="${log_dir}/${run_label}.log"

mkdir -p "${log_dir}"
cd "${study_root}"
exec > >(tee -a "${log_path}") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Robustness pipeline start; workers=${workers}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Verifying focused robustness contracts"
pytest -q \
  tests/test_latent_factor_study_robustness_explicit.py \
  tests/test_latent_factor_study_robustness_bpr.py \
  tests/test_latent_factor_study_robustness_explicit_confirmation.py \
  tests/test_latent_factor_study_robustness_analysis.py

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Building nested retention artifacts"
python -m src.latent_factor_study.robustness_explicit nested-retention

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Expanded explicit-MF surface and budgets"
python -m src.latent_factor_study.robustness_explicit expanded \
  --dataset all --retention all --workers "${workers}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Repeated nested-retention searches"
python -m src.latent_factor_study.robustness_explicit repeated \
  --retention-seed all --dataset all --retention all --workers "${workers}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Explicit-MF fresh-seed post-confirmation summaries"
python -m src.latent_factor_study.robustness_explicit_confirmation expanded \
  --dataset all --retention all --workers "${workers}"
python -m src.latent_factor_study.robustness_explicit_confirmation repeated \
  --retention-seed all --dataset all --retention all --workers "${workers}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Independent BPR coarse and refined tuning"
python -m src.latent_factor_study.robustness_bpr coarse \
  --dataset all --retention all --workers "${workers}"
python -m src.latent_factor_study.robustness_bpr refine \
  --dataset all --retention all --workers "${workers}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BPR negative-pool sensitivity and confirmation"
python -m src.latent_factor_study.robustness_bpr sensitivity \
  --dataset all --retention all --workers "${workers}"
python -m src.latent_factor_study.robustness_bpr confirmation \
  --dataset all --retention all --workers "${workers}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Robustness analysis and publication figures"
python -m src.latent_factor_study.robustness_analysis

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Robustness pipeline complete"
