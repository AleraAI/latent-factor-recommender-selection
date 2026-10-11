from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.latent_factor_study.bpr_lock import (
    build_bpr_selection_lock,
    verify_bpr_selection_lock,
)
from src.latent_factor_study.bpr_grid import DATASETS, RETENTIONS
from src.latent_factor_study.data import sha256_file


def test_bpr_lock_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    dataset_path = configs / "datasets.yaml"
    modeling_path = configs / "bpr.yaml"
    confirmation_path = configs / "confirmation.yaml"
    lock_path = configs / "lock.yaml"
    output_root = tmp_path / "outputs"
    dataset_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {name: {"output_dir": f"data/{name}"} for name in DATASETS},
                "controlled_sparsity": {"output_root": "data/sparsity"},
            }
        ),
        encoding="utf-8",
    )
    modeling_path.write_text(
        yaml.safe_dump(
            {
                "development_output_root": str(output_root),
                "bpr_mf": {
                    "dimensions": [1],
                    "regularizations": [0.1],
                    "selection_seeds": [1],
                    "confirmation_seeds": [2],
                },
            }
        ),
        encoding="utf-8",
    )
    confirmation_path.write_text("stage: locked_bpr_test_confirmation\n", encoding="utf-8")
    for dataset in DATASETS:
        for retention in RETENTIONS:
            condition = output_root / dataset / retention
            condition.mkdir(parents=True)
            (condition / "selection.json").write_text(
                json.dumps(
                    {
                        "stage": "bpr_development_validation_only",
                        "observed_runs": 1,
                        "expected_runs": 1,
                        "selection": {
                            "bpr_selected": {
                                "dimension": 1,
                                "regularization": 0.1,
                                "selection_seed_count": 1,
                                "mean_validation_ndcg@10": 0.2,
                            }
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    build_bpr_selection_lock(
        dataset_config_path=dataset_path,
        modeling_path=modeling_path,
        confirmation_path=confirmation_path,
        output_path=lock_path,
    )
    verified = verify_bpr_selection_lock(
        lock_path=lock_path,
        dataset_config_path=dataset_path,
        modeling_path=modeling_path,
        confirmation_path=confirmation_path,
    )
    assert verified["conditions_verified"] == 12
    assert verified["development_runs_verified"] == 12

    original = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    original["locked_input_sha256"]["datasets_yaml"] = "0" * len(
        sha256_file(dataset_path)
    )
    lock_path.write_text(yaml.safe_dump(original, sort_keys=False), encoding="utf-8")
    try:
        verify_bpr_selection_lock(
            lock_path=lock_path,
            dataset_config_path=dataset_path,
            modeling_path=modeling_path,
            confirmation_path=confirmation_path,
        )
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("Tampered BPR lock should fail verification.")
