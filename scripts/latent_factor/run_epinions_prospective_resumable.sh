#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study_root="$(cd "${script_dir}/../.." && pwd)"
workers="${EPINIONS_WORKERS:-6}"
run_label="${1:-epinions_prospective_20260825}"
log_dir="${study_root}/outputs/latent_factor_beyond_rmse/prospective_epinions/logs"
log_path="${log_dir}/${run_label}.log"
dataset_config="configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml"
modeling_config="configs/latent_factor_beyond_rmse/epinions_prospective_modeling.yaml"
protocol_lock="configs/latent_factor_beyond_rmse/epinions_prospective_protocol_lock.yaml"

mkdir -p "${log_dir}"
cd "${study_root}"
exec >>"${log_path}" 2>&1

python - <<'PY'
from pathlib import Path
from src.latent_factor_study.uncertainty_selector import verify_prospective_protocol_lock

verify_prospective_protocol_lock(
    protocol_lock_path=Path("configs/latent_factor_beyond_rmse/epinions_prospective_protocol_lock.yaml"),
    dataset_config_path=Path("configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml"),
    modeling_path=Path("configs/latent_factor_beyond_rmse/epinions_prospective_modeling.yaml"),
)
print("Prospective protocol lock verified; test evaluation remains prohibited.")
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Epinions prospective grid start; workers=${workers}"
for retention in r100 r075 r050 r025; do
  python -m src.latent_factor_study.grid \
    --config "${dataset_config}" \
    --modeling-config "${modeling_config}" \
    --dataset epinions \
    --retention "${retention}" \
    --workers "${workers}"
done

python -m src.latent_factor_study.uncertainty_selector \
  --config "${dataset_config}" \
  --modeling-config "${modeling_config}" \
  --protocol-lock "${protocol_lock}" \
  --dataset epinions \
  --retention all

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Epinions development and framework selection complete"
echo "Test confirmation is intentionally not started until the Epinions selection lock is written."
