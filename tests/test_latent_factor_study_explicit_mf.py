from __future__ import annotations

import numpy as np
import pandas as pd

from src.latent_factor_study.explicit_mf import (
    MFRunConfig,
    evaluate_stage,
    evaluate_validation,
    fit_explicit_mf,
    predict_frame,
)


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    validation_rows = []
    source = 0
    for user_index, user_id in enumerate(("u1", "u2", "u3", "u4")):
        for item_index, item_id in enumerate(("i1", "i2", "i3", "i4")):
            source += 1
            train_rows.append(
                {
                    "interaction_id": f"t{source}",
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": float(1 + ((user_index + item_index) % 5)),
                    "timestamp": source,
                    "source_row_id": source,
                }
            )
        validation_rows.append(
            {
                "interaction_id": f"v{user_id}",
                "user_id": user_id,
                "item_id": "i5" if user_id == "u1" else "i4",
                "rating": 5.0,
                "timestamp": 100 + user_index,
                "source_row_id": 100 + user_index,
            }
        )
    return pd.DataFrame(train_rows), pd.DataFrame(validation_rows)


def test_explicit_mf_is_reproducible_and_reports_validation_only() -> None:
    train, validation = _toy()
    config = MFRunConfig(
        dimension=2,
        regularization=0.01,
        learning_rate=0.02,
        batch_size=8,
        max_epochs=3,
        convergence_patience=10,
        convergence_relative_tolerance=0.0,
        seed=17,
        torch_num_threads=1,
    )
    first = fit_explicit_mf(train, config)
    second = fit_explicit_mf(train, config)

    np.testing.assert_allclose(predict_frame(first, train), predict_frame(second, train))
    cold = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "unknown", "rating": 3.0},
            {"user_id": "unknown", "item_id": "i1", "rating": 3.0},
        ]
    )
    cold_predictions = predict_frame(first, cold)
    assert np.isfinite(cold_predictions).all()
    assert cold_predictions[1] == float(first.model.global_mean)
    summary, per_user = evaluate_validation(
        first, sparse_train=train, full_train=train, validation=validation, ranking_batch_size=2
    )
    assert summary["prediction"]["num_rows"] == 4
    assert summary["prediction"]["item_cold_rows"] == 1
    assert summary["ranking"]["users_total"] == 4
    assert set(per_user.columns) == {
        "user_id",
        "num_candidates",
        "num_relevant",
        "num_cold_relevant",
        "ndcg@10",
        "recall@10",
        "ndcg@20",
        "recall@20",
        "num_prediction_rows",
        "prediction_rmse",
        "prediction_mae",
        "prediction_raw_rmse",
        "prediction_raw_mae",
    }

    alternate = validation.copy()
    alternate.loc[alternate["user_id"] == "u1", "rating"] = 4.0
    alternate_summary, alternate_users = evaluate_stage(
        first,
        full_train=train,
        evaluation=alternate,
        ranking_batch_size=2,
        relevance_threshold=5.0,
    )
    assert alternate_summary["ranking"]["relevance_threshold"] == 5.0
    assert alternate_summary["ranking"]["cold_relevant_events"] == 0
    assert alternate_users.loc[
        alternate_users["user_id"] == "u1", "num_cold_relevant"
    ].item() == 0
