from pathlib import Path

import pandas as pd

from src.latent_factor_study.plotting import (
    DATASETS,
    RETENTIONS,
    plot_epinions_validation_surface,
    plot_selection_regret,
)


def test_selection_regret_plot_writes_pdf_and_png(tmp_path: Path) -> None:
    rows = []
    for dataset, _ in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            value = 0.001 * (index + 1)
            rows.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "delta_rank_ndcg@10": value,
                    "delta_rank_ci_lower": value - 0.0005,
                    "delta_rank_ci_upper": value + 0.0005,
                }
            )
    paths = plot_selection_regret(pd.DataFrame(rows), tmp_path / "selection_regret")
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_epinions_validation_surface_writes_pdf_and_png(tmp_path: Path) -> None:
    selections = {}
    for retention in RETENTIONS:
        rows = []
        for index, regularization in enumerate((0.0, 0.1)):
            rows.append(
                {
                    "dimension": 64,
                    "regularization": regularization,
                    "mean_validation_rmse": 1.1 - index * 0.05,
                    "paired_regret_upper_bound": index * 0.001,
                }
            )
        selections[retention] = {
            "selection": {
                "configuration_table": rows,
                "ndcg_selected": {"dimension": 64, "regularization": 0.0},
                "framework_selected": {"dimension": 64, "regularization": 0.1},
            }
        }
    paths = plot_epinions_validation_surface(
        selections, tmp_path / "epinions_validation_surface"
    )
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
