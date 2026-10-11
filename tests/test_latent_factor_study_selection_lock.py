from src.latent_factor_study.selection_lock import _same_config


def test_same_config_normalizes_numeric_types() -> None:
    assert _same_config(
        {"dimension": "16", "regularization": "0.1"},
        {"dimension": 16, "regularization": 0.1},
    )
    assert not _same_config(
        {"dimension": 16, "regularization": 0.1},
        {"dimension": 32, "regularization": 0.1},
    )
