"""Uncertainty-aware practical-equivalence selection on validation grids.

The framework is intentionally validation-only.  For each configuration, it
forms paired per-seed NDCG@10 regret relative to the ranking winner and admits
the configuration only when a one-sided Student-t upper bound is no greater
than the frozen practical-regret threshold.  It then minimizes validation RMSE
inside that admissible set, with declared deterministic tie-breakers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml
from scipy.stats import t as student_t

from .data import _repo_root, _resolve, sha256_file


RETENTIONS = ("r100", "r075", "r050", "r025")


def _same_config(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["dimension"]) == int(right["dimension"]) and float(
        left["regularization"]
    ) == float(right["regularization"])


def _choice(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dimension": int(row["dimension"]),
        "regularization": float(row["regularization"]),
        "selection_seed_count": int(row["seeds"]),
        "mean_validation_rmse": float(row["mean_validation_rmse"]),
        "mean_validation_ndcg@10": float(row["mean_validation_ndcg_at_10"]),
        "mean_training_seconds": float(row["mean_training_seconds"]),
    }


def select_uncertainty_aware_configuration(
    rows: pd.DataFrame,
    *,
    practical_regret: float = 0.005,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Select one configuration without using test observations or metrics."""
    required = {
        "dimension",
        "regularization",
        "seed",
        "training_seconds",
        "validation_rmse",
        "validation_ndcg@10",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Framework grid rows are missing columns: {missing}")
    if not 0.0 < practical_regret < 1.0:
        raise ValueError("practical_regret must be in (0, 1).")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0.5, 1).")
    if rows.duplicated(["dimension", "regularization", "seed"]).any():
        raise ValueError("Framework grid contains duplicate configuration-seed rows.")

    seed_sets = rows.groupby(["dimension", "regularization"])["seed"].apply(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if seed_sets.empty or len(set(seed_sets)) != 1:
        raise ValueError("Every configuration must use the same selection seeds.")
    seeds = seed_sets.iloc[0]
    if len(seeds) < 2:
        raise ValueError("At least two paired selection seeds are required.")

    grouped = (
        rows.groupby(["dimension", "regularization"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_validation_rmse=("validation_rmse", "mean"),
            std_validation_rmse=("validation_rmse", "std"),
            mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
            std_validation_ndcg_at_10=("validation_ndcg@10", "std"),
            mean_training_seconds=("training_seconds", "mean"),
        )
    )
    ranking_winner = grouped.sort_values(
        ["mean_validation_ndcg_at_10", "dimension", "regularization"],
        ascending=[False, True, False],
        kind="stable",
    ).iloc[0]
    rmse_winner = grouped.sort_values(
        ["mean_validation_rmse", "dimension", "regularization"],
        ascending=[True, True, False],
        kind="stable",
    ).iloc[0]

    reference_seed_scores = rows.loc[
        (rows["dimension"] == ranking_winner["dimension"])
        & (rows["regularization"] == ranking_winner["regularization"]),
        ["seed", "validation_ndcg@10"],
    ].rename(columns={"validation_ndcg@10": "reference_ndcg_at_10"})
    critical_value = float(student_t.ppf(confidence_level, len(seeds) - 1))

    diagnostics: list[dict[str, Any]] = []
    for _, candidate in grouped.iterrows():
        candidate_seed_scores = rows.loc[
            (rows["dimension"] == candidate["dimension"])
            & (rows["regularization"] == candidate["regularization"]),
            ["seed", "validation_ndcg@10"],
        ]
        paired = reference_seed_scores.merge(
            candidate_seed_scores, on="seed", how="inner", validate="one_to_one"
        )
        regret = paired["reference_ndcg_at_10"] - paired["validation_ndcg@10"]
        mean_regret = float(regret.mean())
        standard_error = float(regret.std(ddof=1) / (len(regret) ** 0.5))
        upper_bound = float(mean_regret + critical_value * standard_error)
        diagnostics.append(
            {
                **_choice(candidate),
                "paired_mean_ndcg_at_10_regret": mean_regret,
                "paired_regret_standard_error": standard_error,
                "paired_regret_upper_bound": upper_bound,
                "admissible": bool(upper_bound <= practical_regret),
            }
        )

    admissible = [row for row in diagnostics if row["admissible"]]
    if not admissible:
        raise AssertionError("The ranking reference must always be admissible.")
    framework_choice = sorted(
        admissible,
        key=lambda row: (
            row["mean_validation_rmse"],
            row["mean_training_seconds"],
            row["dimension"],
            -row["regularization"],
        ),
    )[0]
    framework_selected = {
        key: framework_choice[key]
        for key in (
            "dimension",
            "regularization",
            "selection_seed_count",
            "mean_validation_rmse",
            "mean_validation_ndcg@10",
            "mean_training_seconds",
            "paired_mean_ndcg_at_10_regret",
            "paired_regret_standard_error",
            "paired_regret_upper_bound",
        )
    }

    return {
        "rule": {
            "reference": "highest_mean_validation_ndcg_at_10",
            "regret_pairing": "common_random_seed",
            "confidence_bound": "one_sided_student_t_upper",
            "confidence_level": float(confidence_level),
            "degrees_of_freedom": len(seeds) - 1,
            "critical_value": critical_value,
            "practical_ndcg_at_10_regret": float(practical_regret),
            "admissibility_rule": "paired_regret_upper_bound_lte_threshold",
            "optimize_within_admissible_set": "lowest_mean_validation_rmse",
            "tie_breakers": [
                "lower_mean_training_seconds",
                "smaller_dimension",
                "larger_regularization",
            ],
        },
        "rmse_selected": _choice(rmse_winner),
        "ndcg_selected": _choice(ranking_winner),
        "framework_selected": framework_selected,
        "admissible_configurations": len(admissible),
        "framework_matches_rmse": _same_config(framework_selected, rmse_winner),
        "framework_matches_ndcg": _same_config(framework_selected, ranking_winner),
        "configuration_table": sorted(
            diagnostics, key=lambda row: (row["dimension"], row["regularization"])
        ),
    }


def verify_prospective_protocol_lock(
    *,
    protocol_lock_path: Path,
    dataset_config_path: Path,
    modeling_path: Path,
) -> dict[str, Any]:
    """Verify the pre-development protocol lock without reading any outcomes."""
    lock = yaml.safe_load(protocol_lock_path.read_text(encoding="utf-8")) or {}
    if lock.get("status") != "frozen_before_epinions_development":
        raise ValueError("Prospective protocol is not frozen for development.")
    inputs = lock.get("locked_input_sha256", {})
    for label, path in (
        ("datasets_yaml", dataset_config_path),
        ("modeling_yaml", modeling_path),
        ("selector_py", Path(__file__)),
    ):
        actual = sha256_file(path)
        if actual != str(inputs.get(label, "")):
            raise ValueError(f"Prospective protocol hash mismatch for {label}: {actual}")
    return lock


def select_condition(
    *,
    dataset_config_path: Path,
    modeling_path: Path,
    protocol_lock_path: Path,
    dataset_name: str,
    retention_label: str,
) -> Path:
    """Apply the frozen framework to one completed validation-only grid."""
    lock = verify_prospective_protocol_lock(
        protocol_lock_path=protocol_lock_path,
        dataset_config_path=dataset_config_path,
        modeling_path=modeling_path,
    )
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(dataset_config_path)
    grid_path = (
        _resolve(root, modeling["development_output_root"])
        / dataset_name
        / retention_label
        / "grid_runs.csv"
    )
    rows = pd.read_csv(grid_path)
    expected = (
        len(modeling["explicit_mf"]["dimensions"])
        * len(modeling["explicit_mf"]["regularizations"])
        * len(modeling["explicit_mf"]["selection_seeds"])
    )
    if len(rows) != expected:
        raise ValueError(
            f"Incomplete prospective grid for {dataset_name}/{retention_label}: "
            f"expected {expected}, observed {len(rows)}."
        )
    settings = modeling["uncertainty_aware_selection"]
    selection = select_uncertainty_aware_configuration(
        rows,
        practical_regret=float(settings["practical_ndcg_at_10_regret"]),
        confidence_level=float(settings["confidence_level"]),
    )
    artifact = {
        "schema_version": 1,
        "stage": "prospective_development_validation_only",
        "test_metrics_accessed": False,
        "dataset": dataset_name,
        "retention_label": retention_label,
        "protocol_lock_sha256": sha256_file(protocol_lock_path),
        "grid_runs_sha256": sha256_file(grid_path),
        "expected_runs": expected,
        "observed_runs": len(rows),
        "selection": selection,
    }
    path = grid_path.parent / "framework_selection.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml"
        ),
    )
    parser.add_argument(
        "--modeling-config",
        type=Path,
        default=Path(
            "configs/latent_factor_beyond_rmse/epinions_prospective_modeling.yaml"
        ),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(
            "configs/latent_factor_beyond_rmse/epinions_prospective_protocol_lock.yaml"
        ),
    )
    parser.add_argument("--dataset", default="epinions")
    parser.add_argument("--retention", choices=["all", *RETENTIONS], default="all")
    return parser


def main() -> None:
    args = _parser().parse_args()
    retentions = RETENTIONS if args.retention == "all" else (args.retention,)
    for retention in retentions:
        print(
            select_condition(
                dataset_config_path=args.config,
                modeling_path=args.modeling_config,
                protocol_lock_path=args.protocol_lock,
                dataset_name=args.dataset,
                retention_label=retention,
            )
        )


if __name__ == "__main__":
    main()
