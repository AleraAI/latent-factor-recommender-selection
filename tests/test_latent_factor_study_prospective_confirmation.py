from __future__ import annotations

from src.latent_factor_study.prospective_confirmation import selected_config_groups


def test_three_roles_collapse_identical_configurations() -> None:
    groups = selected_config_groups(
        {
            "rmse_selected": {"dimension": 16, "regularization": 0.1},
            "ndcg_selected": {"dimension": 64, "regularization": 0.01},
            "framework_selected": {"dimension": 16, "regularization": 0.1},
        }
    )

    assert len(groups) == 2
    assert groups[0]["roles"] == ["rmse_selected", "framework_selected"]
    assert groups[1]["roles"] == ["ndcg_selected"]

