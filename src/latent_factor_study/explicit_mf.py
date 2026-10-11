"""Biased explicit-feedback matrix factorization and validation evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve
from .evaluation import prediction_metrics, ranking_metrics, rating_gain, top_k_items


@dataclass(frozen=True)
class MFRunConfig:
    dimension: int
    regularization: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    convergence_patience: int
    convergence_relative_tolerance: float
    seed: int
    device: str = "cpu"
    torch_num_threads: int = 1


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _id_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


class BiasedMatrixFactorization(nn.Module):
    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        dimension: int,
        global_mean: float,
    ) -> None:
        super().__init__()
        if num_users <= 0 or num_items <= 0:
            raise ValueError("The training matrix must contain users and items.")
        if dimension < 0:
            raise ValueError("dimension cannot be negative.")
        self.dimension = dimension
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_factors = nn.Embedding(num_users, dimension) if dimension else None
        self.item_factors = nn.Embedding(num_items, dimension) if dimension else None
        self.register_buffer("global_mean", torch.tensor(float(global_mean), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        if self.user_factors is not None and self.item_factors is not None:
            nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.item_factors.weight, mean=0.0, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        prediction = (
            self.global_mean
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
        )
        if self.user_factors is not None and self.item_factors is not None:
            prediction = prediction + torch.sum(
                self.user_factors(users) * self.item_factors(items), dim=-1
            )
        return prediction

    def regularization_per_observation(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        penalty = self.user_bias(users).squeeze(-1).square()
        penalty = penalty + self.item_bias(items).squeeze(-1).square()
        if self.user_factors is not None and self.item_factors is not None:
            penalty = penalty + self.user_factors(users).square().sum(dim=-1)
            penalty = penalty + self.item_factors(items).square().sum(dim=-1)
        return penalty

    def score_catalog(self, users: torch.Tensor) -> torch.Tensor:
        scores = (
            self.global_mean
            + self.user_bias(users)
            + self.item_bias.weight.squeeze(-1).unsqueeze(0)
        )
        if self.user_factors is not None and self.item_factors is not None:
            scores = scores + self.user_factors(users) @ self.item_factors.weight.T
        return scores


@dataclass
class FittedMF:
    model: BiasedMatrixFactorization
    user_ids: list[str]
    item_ids: list[str]
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    epoch_history: list[dict[str, float | int]]
    training_seconds: float
    config: MFRunConfig


def fit_explicit_mf(train: pd.DataFrame, config: MFRunConfig) -> FittedMF:
    required = {"user_id", "item_id", "rating"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Training data are missing columns: {missing}")
    if train.empty:
        raise ValueError("Training data are empty.")

    set_deterministic_seed(config.seed)
    torch.set_num_threads(config.torch_num_threads)
    device = torch.device(config.device)
    user_ids = sorted(set(train["user_id"].astype(str)), key=_id_sort_key)
    item_ids = sorted(set(train["item_id"].astype(str)), key=_id_sort_key)
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    user_index = train["user_id"].astype(str).map(user_to_index).to_numpy(dtype=np.int64)
    item_index = train["item_id"].astype(str).map(item_to_index).to_numpy(dtype=np.int64)
    ratings = train["rating"].to_numpy(dtype=np.float32)

    model = BiasedMatrixFactorization(
        num_users=len(user_ids),
        num_items=len(item_ids),
        dimension=config.dimension,
        global_mean=float(ratings.mean()),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    epoch_history: list[dict[str, float | int]] = []
    best_objective = math.inf
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        permutation = rng.permutation(len(train))
        objective_sum = 0.0
        squared_error_sum = 0.0
        model.train()
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            users = torch.as_tensor(user_index[indices], dtype=torch.long, device=device)
            items = torch.as_tensor(item_index[indices], dtype=torch.long, device=device)
            target = torch.as_tensor(ratings[indices], dtype=torch.float32, device=device)
            prediction = model(users, items)
            squared_errors = (prediction - target).square()
            objective = squared_errors.mean()
            if config.regularization:
                objective = objective + config.regularization * model.regularization_per_observation(
                    users, items
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            objective_sum += float(objective.detach()) * len(indices)
            squared_error_sum += float(squared_errors.detach().sum())

        epoch_objective = objective_sum / len(train)
        train_rmse = math.sqrt(squared_error_sum / len(train))
        epoch_history.append(
            {"epoch": epoch, "training_objective": epoch_objective, "training_rmse": train_rmse}
        )
        relative_improvement = (
            math.inf
            if not math.isfinite(best_objective)
            else (best_objective - epoch_objective) / max(abs(best_objective), 1.0)
        )
        if epoch_objective < best_objective:
            best_objective = epoch_objective
        if relative_improvement > config.convergence_relative_tolerance:
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.convergence_patience:
            break

    return FittedMF(
        model=model,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        epoch_history=epoch_history,
        training_seconds=time.perf_counter() - started,
        config=config,
    )


@torch.no_grad()
def predict_frame(fitted: FittedMF, frame: pd.DataFrame) -> np.ndarray:
    """Predict held-out pairs; naturally cold items receive zero item effects."""
    model = fitted.model
    model.eval()
    user_index = frame["user_id"].astype(str).map(fitted.user_to_index)
    item_index = frame["item_id"].astype(str).map(fitted.item_to_index)
    user_bias = model.user_bias.weight.detach().cpu().numpy().reshape(-1)
    item_bias = model.item_bias.weight.detach().cpu().numpy().reshape(-1)
    predictions = np.full(len(frame), float(model.global_mean.cpu()), dtype=np.float64)

    known_users = user_index.notna().to_numpy()
    user_values = user_index.fillna(0).to_numpy(dtype=np.int64)
    predictions[known_users] += user_bias[user_values[known_users]]
    known_pairs = known_users & item_index.notna().to_numpy()
    item_values = item_index.fillna(0).to_numpy(dtype=np.int64)
    predictions[known_pairs] += item_bias[item_values[known_pairs]]
    if model.user_factors is not None and model.item_factors is not None:
        user_factors = model.user_factors.weight.detach().cpu().numpy()
        item_factors = model.item_factors.weight.detach().cpu().numpy()
        predictions[known_pairs] += np.einsum(
            "ij,ij->i",
            user_factors[user_values[known_pairs]],
            item_factors[item_values[known_pairs]],
        )
    return predictions


@torch.no_grad()
def evaluate_stage(
    fitted: FittedMF,
    *,
    full_train: pd.DataFrame,
    evaluation: pd.DataFrame,
    prior_evaluation: pd.DataFrame | None = None,
    ranking_batch_size: int = 256,
    relevance_threshold: float = 4.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one stage under the exact train-known catalog contract."""
    predictions = predict_frame(fitted, evaluation)
    clipped_predictions = np.clip(predictions, 1.0, 5.0)
    clipped = prediction_metrics(evaluation["rating"], clipped_predictions)
    raw = prediction_metrics(
        evaluation["rating"], predictions, clip_min=None, clip_max=None
    )
    prediction_rows = evaluation[["user_id", "rating"]].copy()
    prediction_rows["user_id"] = prediction_rows["user_id"].astype(str)
    prediction_rows["clipped_error"] = (
        clipped_predictions - prediction_rows["rating"].to_numpy(dtype=float)
    )
    prediction_rows["raw_error"] = (
        predictions - prediction_rows["rating"].to_numpy(dtype=float)
    )
    prediction_by_user = (
        prediction_rows.groupby("user_id", as_index=False, sort=False)
        .agg(
            num_prediction_rows=("rating", "size"),
            prediction_mse=("clipped_error", lambda values: float(np.mean(values**2))),
            prediction_mae=("clipped_error", lambda values: float(np.mean(np.abs(values)))),
            prediction_raw_mse=("raw_error", lambda values: float(np.mean(values**2))),
            prediction_raw_mae=("raw_error", lambda values: float(np.mean(np.abs(values)))),
        )
    )
    prediction_by_user["prediction_rmse"] = np.sqrt(prediction_by_user.pop("prediction_mse"))
    prediction_by_user["prediction_raw_rmse"] = np.sqrt(
        prediction_by_user.pop("prediction_raw_mse")
    )

    model = fitted.model
    model.eval()
    device = model.global_mean.device
    catalog = fitted.item_ids
    catalog_lookup = set(catalog)
    item_array = np.asarray(catalog, dtype=str)
    history = (
        pd.concat([full_train, prior_evaluation], ignore_index=True)
        if prior_evaluation is not None
        else full_train
    )
    train_history = (
        history.groupby(history["user_id"].astype(str), sort=False)["item_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    evaluation_by_user = {
        str(user): rows
        for user, rows in evaluation.groupby(evaluation["user_id"].astype(str), sort=False)
    }
    rows_out: list[dict[str, Any]] = []
    eval_users = sorted(evaluation_by_user, key=_id_sort_key)

    for start in range(0, len(eval_users), ranking_batch_size):
        batch_users = eval_users[start : start + ranking_batch_size]
        user_indices = torch.tensor(
            [fitted.user_to_index[user] for user in batch_users],
            dtype=torch.long,
            device=device,
        )
        score_matrix = model.score_catalog(user_indices).cpu().numpy()
        for row_index, user_id in enumerate(batch_users):
            scores = score_matrix[row_index].copy()
            historical = train_history.get(user_id, set())
            historical_indices = [
                fitted.item_to_index[item]
                for item in historical
                if item in fitted.item_to_index
            ]
            scores[historical_indices] = -np.inf
            num_candidates = len(catalog) - len(historical_indices)
            ranked = top_k_items(item_array, scores, 20)

            user_evaluation = evaluation_by_user[user_id]
            relevant_rows = user_evaluation[
                user_evaluation["rating"] >= relevance_threshold
            ]
            gains = {
                str(row.item_id): rating_gain(
                    float(row.rating), threshold=relevance_threshold
                )
                for row in relevant_rows.itertuples(index=False)
                if str(row.item_id) in catalog_lookup and str(row.item_id) not in historical
            }
            cold = {
                str(item)
                for item in relevant_rows["item_id"].astype(str)
                if str(item) not in catalog_lookup
            }
            metrics = ranking_metrics(ranked, gains)
            rows_out.append(
                {
                    "user_id": user_id,
                    "num_candidates": num_candidates,
                    "num_relevant": len(gains),
                    "num_cold_relevant": len(cold),
                    **metrics,
                }
            )

    per_user = pd.DataFrame.from_records(rows_out).merge(
        prediction_by_user, on="user_id", how="left", validate="one_to_one"
    )
    evaluable = per_user[per_user["num_relevant"] > 0]
    ranking_summary = {
        metric: float(evaluable[metric].mean())
        for metric in ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
    }
    summary = {
        "prediction": {
            "num_rows": int(len(evaluation)),
            "clipped": clipped,
            "raw": raw,
            "item_cold_rows": int(
                (~evaluation["item_id"].astype(str).isin(catalog_lookup)).sum()
            ),
        },
        "ranking": {
            "users_total": int(len(per_user)),
            "users_evaluable": int(len(evaluable)),
            "users_without_eligible_relevant_items": int(len(per_user) - len(evaluable)),
            "cold_relevant_events": int(per_user["num_cold_relevant"].sum()),
            "relevance_threshold": float(relevance_threshold),
            **ranking_summary,
        },
    }
    return summary, per_user


def evaluate_validation(
    fitted: FittedMF,
    *,
    sparse_train: pd.DataFrame,
    full_train: pd.DataFrame,
    validation: pd.DataFrame,
    ranking_batch_size: int = 256,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validation-only wrapper; deliberately exposes no test-stage argument."""
    del sparse_train
    return evaluate_stage(
        fitted,
        full_train=full_train,
        evaluation=validation,
        prior_evaluation=None,
        ranking_batch_size=ranking_batch_size,
    )


def run_validation(
    *,
    config_path: Path,
    modeling_path: Path,
    dataset_name: str,
    retention_label: str,
    dimension: int,
    regularization: float,
    seed: int,
    smoke: bool,
) -> Path:
    dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    dataset = dataset_config["datasets"][dataset_name]
    processed_dir = _resolve(root, dataset["output_dir"])
    sparsity_dir = _resolve(root, dataset_config["controlled_sparsity"]["output_root"]) / dataset_name
    sparse_train = pd.read_parquet(sparsity_dir / f"train_{retention_label}.parquet")
    full_train = pd.read_parquet(processed_dir / "train.parquet")
    validation = pd.read_parquet(processed_dir / "validation.parquet")
    # Deliberately do not resolve or read test.parquet in a development run.

    settings = dict(modeling["explicit_mf"])
    max_epochs = int(settings["smoke_epochs"] if smoke else settings["max_epochs"])
    run_config = MFRunConfig(
        dimension=dimension,
        regularization=regularization,
        learning_rate=float(settings["learning_rate"]),
        batch_size=int(settings["batch_size"]),
        max_epochs=max_epochs,
        convergence_patience=int(settings["convergence_patience"]),
        convergence_relative_tolerance=float(settings["convergence_relative_tolerance"]),
        seed=seed,
        device=str(settings["device"]),
        torch_num_threads=int(settings["torch_num_threads"]),
    )
    fitted = fit_explicit_mf(sparse_train, run_config)
    validation_summary, per_user = evaluate_validation(
        fitted,
        sparse_train=sparse_train,
        full_train=full_train,
        validation=validation,
        ranking_batch_size=int(settings["ranking_batch_size"]),
    )
    run_id = (
        f"d{dimension}_lambda{regularization:g}_seed{seed}"
        + ("_smoke" if smoke else "")
    )
    output_dir = _resolve(root, modeling["development_output_root"]) / dataset_name / retention_label / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_parquet(output_dir / "validation_per_user.parquet", index=False)
    summary = {
        "schema_version": 1,
        "stage": "development_validation_only",
        "dataset": dataset_name,
        "retention_label": retention_label,
        "run_id": run_id,
        "smoke": smoke,
        "run_config": asdict(run_config),
        "training": {
            "rows": int(len(sparse_train)),
            "users": len(fitted.user_ids),
            "items": len(fitted.item_ids),
            "epochs_completed": len(fitted.epoch_history),
            "seconds": fitted.training_seconds,
            "epoch_history": fitted.epoch_history,
        },
        "validation": validation_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--modeling-config",
        type=Path,
        default=Path("configs/latent_factor_beyond_rmse/modeling.yaml"),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["movielens_1m", "ciaodvd", "yelp_tampa_restaurants", "epinions"],
    )
    parser.add_argument("--retention", required=True, choices=["r100", "r075", "r050", "r025"])
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--regularization", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary_path = run_validation(
        config_path=args.config,
        modeling_path=args.modeling_config,
        dataset_name=args.dataset,
        retention_label=args.retention,
        dimension=args.dimension,
        regularization=args.regularization,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(summary_path)


if __name__ == "__main__":
    main()
