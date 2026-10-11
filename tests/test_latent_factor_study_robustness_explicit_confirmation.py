from __future__ import annotations

import json

import pandas as pd

from src.latent_factor_study.robustness_explicit_confirmation import (
    _collapse_roles,
    _fit_and_evaluate,
)


def test_role_collapse_avoids_duplicate_confirmation_fits() -> None:
    collapsed = _collapse_roles(
        [
            {"role": "rmse_selected", "dimension": 4, "regularization": 0.1, "epoch": 20},
            {"role": "ndcg_selected", "dimension": 4, "regularization": 0.1, "epoch": 20},
        ]
    )
    assert len(collapsed) == 1
    assert sorted(collapsed[0]["roles"]) == ["ndcg_selected", "rmse_selected"]


def test_fixed_epoch_confirmation_writes_post_confirmation_evidence(tmp_path) -> None:
    sparse = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0},
            {"user_id": "u1", "item_id": "i2", "rating": 2.0},
            {"user_id": "u2", "item_id": "i2", "rating": 5.0},
            {"user_id": "u2", "item_id": "i3", "rating": 2.0},
            {"user_id": "u3", "item_id": "i3", "rating": 5.0},
            {"user_id": "u3", "item_id": "i1", "rating": 1.0},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i3", "rating": 5.0},
            {"user_id": "u2", "item_id": "i1", "rating": 5.0},
            {"user_id": "u3", "item_id": "i2", "rating": 5.0},
        ]
    )
    test = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i3", "rating": 4.0},
            {"user_id": "u2", "item_id": "i1", "rating": 4.0},
            {"user_id": "u3", "item_id": "i2", "rating": 4.0},
        ]
    )
    paths = {}
    for name, frame in (
        ("sparse", sparse),
        ("full", sparse),
        ("validation", validation),
        ("test", test),
    ):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    summary_path = tmp_path / "run" / "summary.json"
    per_user_path = tmp_path / "run" / "test_per_user.parquet"
    _fit_and_evaluate(
        {
            "summary_path": str(summary_path),
            "per_user_path": str(per_user_path),
            "signature": {"frozen": True},
            "dataset": "toy",
            "retention_label": "r050",
            "retention_seed": 17,
            "selection_source": "toy_validation",
            "selection_roles": ["rmse_selected"],
            "sparse_train_path": str(paths["sparse"]),
            "full_train_path": str(paths["full"]),
            "validation_path": str(paths["validation"]),
            "test_path": str(paths["test"]),
            "ranking_batch_size": 2,
            "relevance_threshold": 4.0,
            "run_config": {
                "dimension": 2,
                "regularization": 0.01,
                "learning_rate": 0.01,
                "batch_size": 4,
                "max_epochs": 2,
                "convergence_patience": 3,
                "convergence_relative_tolerance": 1e-5,
                "seed": 31,
                "device": "cpu",
                "torch_num_threads": 1,
            },
        }
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["test_status"] == "post_confirmation_robustness_not_a_new_locked_test"
    assert payload["training"]["epochs_completed"] == 2
    assert per_user_path.exists()

