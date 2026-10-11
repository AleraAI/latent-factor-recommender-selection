import pandas as pd

from src.latent_factor_study.secondary_analysis import assign_activity_bin


def test_activity_bins_are_fixed_and_exhaustive() -> None:
    bins = [
        {"label": "3--9", "lower_inclusive": 3, "upper_inclusive": 9},
        {"label": "10--19", "lower_inclusive": 10, "upper_inclusive": 19},
        {"label": "20--49", "lower_inclusive": 20, "upper_inclusive": 49},
        {"label": "50+", "lower_inclusive": 50, "upper_inclusive": None},
    ]
    result = assign_activity_bin(pd.Series([3, 9, 10, 19, 20, 49, 50, 100]), bins)
    assert result.tolist() == [
        "3--9",
        "3--9",
        "10--19",
        "10--19",
        "20--49",
        "20--49",
        "50+",
        "50+",
    ]
