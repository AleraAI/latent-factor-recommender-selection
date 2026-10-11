from pathlib import Path

import pandas as pd

from src.latent_factor_study.reporting import _configuration, _integer, _percent, _timing


def test_latex_number_formatters() -> None:
    assert _integer(1000209) == "1,000,209"
    assert _percent(0.2511745, 2) == r"25.12\%"
    assert _configuration(64, 0.0001) == r"$(64,10^{-4})$"
    assert _timing(
        pd.Series(
            {
                "median_training_seconds": 2.345,
                "q25_training_seconds": 2.1,
                "q75_training_seconds": 2.6,
            }
        )
    ) == "2.35 [2.10--2.60]"
