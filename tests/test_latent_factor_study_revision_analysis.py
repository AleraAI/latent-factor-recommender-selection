import numpy as np
import pandas as pd

from src.latent_factor_study.revision_analysis import (
    gini,
    pareto_mask,
    practical_evidence_class,
)


def test_practical_evidence_class_separates_point_and_interval_evidence() -> None:
    assert (
        practical_evidence_class(
            point=0.010, lower=0.006, upper=0.014, threshold=0.005
        )
        == "clearly above"
    )
    assert (
        practical_evidence_class(
            point=0.010, lower=0.004, upper=0.014, threshold=0.005
        )
        == "inconclusive"
    )
    assert (
        practical_evidence_class(
            point=0.003, lower=0.001, upper=0.004, threshold=0.005
        )
        == "clearly below"
    )
    assert (
        practical_evidence_class(
            point=0.0,
            lower=0.0,
            upper=0.0,
            threshold=0.005,
            identical_selection=True,
        )
        == "identical selection"
    )


def test_gini_handles_uniform_and_concentrated_counts() -> None:
    assert gini(np.array([1.0, 1.0, 1.0])) == 0.0
    assert np.isclose(gini(np.array([0.0, 0.0, 3.0])), 2.0 / 3.0)


def test_pareto_mask_requires_joint_domination() -> None:
    frame = pd.DataFrame(
        {
            "mean_validation_rmse": [0.9, 0.8, 0.85, 0.95],
            "mean_validation_ndcg_at_10": [0.10, 0.09, 0.11, 0.05],
            "mean_training_seconds": [1.0, 2.0, 3.0, 4.0],
        }
    )
    mask = pareto_mask(frame)
    assert mask.tolist() == [True, True, True, False]
