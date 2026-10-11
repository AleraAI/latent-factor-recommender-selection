import pandas as pd

from src.latent_factor_study.robustness_explicit import (
    _train_explicit_checkpoints,
    nested_retention_family,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    source = 0
    for user in ("u1", "u2", "u3"):
        for item_index in range(8):
            source += 1
            rows.append(
                {
                    "interaction_id": f"x{source}",
                    "user_id": user,
                    "item_id": f"i{item_index + (8 if user == 'u3' else 0)}",
                    "rating": float(1 + (item_index % 5)),
                    "timestamp": source,
                    "source_row_id": source,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_nested_retention_family_is_deterministic_nested_and_catalog_preserving() -> None:
    train = _training_frame()
    first, audit = nested_retention_family(train, seed=17, minimum_per_user=2)
    second, _ = nested_retention_family(train, seed=17, minimum_per_user=2)
    ids = {
        label: set(frame["interaction_id"].astype(str))
        for label, frame in first.items()
    }
    assert ids["r025"] <= ids["r050"] <= ids["r075"] <= ids["r100"]
    full_catalog = set(train["item_id"].astype(str))
    assert all(set(frame["item_id"].astype(str)) == full_catalog for frame in first.values())
    assert all(first[label].equals(second[label]) for label in first)
    assert audit["family"]["nested"] is True


def test_explicit_checkpoint_runner_reports_each_requested_budget() -> None:
    train = _training_frame()
    validation = pd.DataFrame(
        [
            {
                "interaction_id": f"v{index}",
                "user_id": user,
                "item_id": "i7",
                "rating": 5.0,
                "timestamp": 100 + index,
                "source_row_id": 100 + index,
            }
            for index, user in enumerate(("u1", "u2", "u3"), start=1)
        ]
    )
    result = _train_explicit_checkpoints(
        sparse_train=train,
        full_train=train,
        validation=validation,
        dimension=2,
        regularization=0.01,
        learning_rate=0.01,
        batch_size=8,
        checkpoints=[1, 2],
        convergence_patience=2,
        convergence_relative_tolerance=1e-5,
        seed=11,
        ranking_batch_size=2,
        device_name="cpu",
        torch_num_threads=1,
    )
    assert set(result["validation_checkpoints"]) == {"1", "2"}
    assert result["training"]["epochs_completed"] == 2
    assert len(result["training"]["epoch_history"]) == 2
