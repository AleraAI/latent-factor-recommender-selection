from src.latent_factor_study.confirmation import selected_config_groups


def test_identical_selections_are_trained_once() -> None:
    condition = {
        "rmse_selected": {"dimension": 1, "regularization": 0.1},
        "ndcg_selected": {"dimension": 1, "regularization": 0.1},
    }
    groups = selected_config_groups(condition)
    assert groups == [
        {
            "roles": ["rmse_selected", "ndcg_selected"],
            "dimension": 1,
            "regularization": 0.1,
        }
    ]


def test_distinct_selections_remain_separate() -> None:
    condition = {
        "rmse_selected": {"dimension": 1, "regularization": 0.1},
        "ndcg_selected": {"dimension": 64, "regularization": 0.0},
    }
    groups = selected_config_groups(condition)
    assert [group["roles"] for group in groups] == [["rmse_selected"], ["ndcg_selected"]]
