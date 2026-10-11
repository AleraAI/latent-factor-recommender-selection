#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study_root="$(cd "${script_dir}/../.." && pwd)"
run_label="${1:-locked_followups_20260817}"
log_dir="${study_root}/outputs/latent_factor_beyond_rmse/logs"
log_path="${log_dir}/${run_label}.log"

mkdir -p "${log_dir}"
cd "${study_root}"
exec > >(tee -a "${log_path}") 2>&1

while true; do
  bpr_selections="$(find outputs/latent_factor_beyond_rmse/bpr_development -name selection.json | wc -l | tr -d ' ')"
  baseline_selections="$(find outputs/latent_factor_beyond_rmse/baseline_development -name selection.json | wc -l | tr -d ' ')"
  if [[ "${bpr_selections}" == "12" && "${baseline_selections}" == "12" ]]; then
    break
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Waiting for grids: BPR ${bpr_selections}/12; baselines ${baseline_selections}/12"
  sleep 30
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Building and verifying BPR lock"
python -m src.latent_factor_study.bpr_lock
python -m src.latent_factor_study.bpr_lock --verify

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Building and verifying baseline lock"
python -m src.latent_factor_study.baseline_lock
python -m src.latent_factor_study.baseline_lock --verify

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running BPR confirmation and inference"
python -m src.latent_factor_study.bpr_confirmation
python -m src.latent_factor_study.bpr_inference

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running baseline confirmation and inference"
python -m src.latent_factor_study.baseline_confirmation
python -m src.latent_factor_study.baseline_inference

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running five-star sensitivity and inference"
python -m src.latent_factor_study.five_star_sensitivity
python -m src.latent_factor_study.inference \
  --confirmation-config configs/latent_factor_beyond_rmse/five_star_sensitivity.yaml

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Locked follow-ups complete"
