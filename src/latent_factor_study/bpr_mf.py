"""Ranking-native BPR matrix factorization for the journal boundary study."""

from __future__ import annotations

import argparse
import json
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
from torch.nn import functional as F

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve
from .evaluation import ranking_metrics, rating_gain, top_k_items
from .explicit_mf import _id_sort_key


DEFAULT_BPR_MODELING_PATH = Path(
    "configs/latent_factor_beyond_rmse/bpr_modeling.yaml"
)


@dataclass(frozen=True)
class BPRRunConfig:
    dimension: int
    regularization: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    negatives_per_positive: int
    relevance_threshold: float
    seed: int
    device: str = "cpu"
    torch_num_threads: int = 1


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class BPRMatrixFactorization(nn.Module):
    """Classic BPR-MF with user/item factors and a pairwise item bias."""

    def __init__(self, *, num_users: int, num_items: int, dimension: int) -> None:
        super().__init__()
        if num_users <= 0 or num_items <= 0:
            raise ValueError("The training matrix must contain users and items.")
        if dimension <= 0:
            raise ValueError("BPR dimension must be positive.")
        self.dimension = dimension
        self.user_factors = nn.Embedding(num_users, dimension)
        self.item_factors = nn.Embedding(num_items, dimension)
        self.item_bias = nn.Embedding(num_items, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_factors.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.item_bias(items).squeeze(-1) + torch.sum(
            self.user_factors(users) * self.item_factors(items), dim=-1
        )

    def score_catalog(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_factors(users) @ self.item_factors.weight.T + self.item_bias.weight.T

    def regularization_per_triplet(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> torch.Tensor:
        penalty = self.user_factors(users).square().sum(dim=-1)
        penalty = penalty + self.item_factors(positive_items).square().sum(dim=-1)
        penalty = penalty + self.item_factors(negative_items).square().sum(dim=-1)
        penalty = penalty + self.item_bias(positive_items).squeeze(-1).square()
        penalty = penalty + self.item_bias(negative_items).squeeze(-1).square()
        return penalty


@dataclass
class FittedBPR:
    model: BPRMatrixFactorization
    user_ids: list[str]
    item_ids: list[str]
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    epoch_history: list[dict[str, float | int]]
    training_seconds: float
    positive_rows: int
    users_without_positive_training: int
    config: BPRRunConfig


def sample_unobserved_items(
    *,
    rng: np.random.Generator,
    user_indices: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Uniformly sample one unobserved catalog item for each user index."""
    if observed.ndim != 2 or observed.dtype != np.bool_:
        raise ValueError("observed must be a two-dimensional boolean array.")
    users = np.asarray(user_indices, dtype=np.int64)
    if users.ndim != 1:
        raise ValueError("user_indices must be one-dimensional.")
    if len(users) == 0:
        return np.asarray([], dtype=np.int64)
    if np.any(observed[users].all(axis=1)):
        raise ValueError("Cannot sample a negative for a user who observed the full catalog.")
    negatives = rng.integers(0, observed.shape[1], size=len(users), dtype=np.int64)
    invalid = observed[users, negatives]
    while bool(invalid.any()):
        negatives[invalid] = rng.integers(
            0, observed.shape[1], size=int(invalid.sum()), dtype=np.int64
        )
        invalid = observed[users, negatives]
    return negatives


def fit_bpr_mf(train: pd.DataFrame, config: BPRRunConfig) -> FittedBPR:
    required = {"user_id", "item_id", "rating"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Training data are missing columns: {missing}")
    if train.empty:
        raise ValueError("Training data are empty.")
    if config.negatives_per_positive != 1:
        raise ValueError("The frozen study currently requires one negative per positive.")

    set_deterministic_seed(config.seed)
    torch.set_num_threads(config.torch_num_threads)
    device = torch.device(config.device)
    user_ids = sorted(set(train["user_id"].astype(str)), key=_id_sort_key)
    item_ids = sorted(set(train["item_id"].astype(str)), key=_id_sort_key)
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    all_users = train["user_id"].astype(str).map(user_to_index).to_numpy(dtype=np.int64)
    all_items = train["item_id"].astype(str).map(item_to_index).to_numpy(dtype=np.int64)
    observed = np.zeros((len(user_ids), len(item_ids)), dtype=np.bool_)
    observed[all_users, all_items] = True

    positive_mask = train["rating"].to_numpy(dtype=float) >= config.relevance_threshold
    positive_users = all_users[positive_mask]
    positive_items = all_items[positive_mask]
    eligible_users = ~observed.all(axis=1)
    eligible_rows = eligible_users[positive_users]
    positive_users = positive_users[eligible_rows]
    positive_items = positive_items[eligible_rows]
    if len(positive_users) == 0:
        raise ValueError("BPR training requires at least one eligible positive interaction.")

    model = BPRMatrixFactorization(
        num_users=len(user_ids), num_items=len(item_ids), dimension=config.dimension
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    rng = np.random.default_rng(config.seed)
    epoch_history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        permutation = rng.permutation(len(positive_users))
        pairwise_sum = 0.0
        objective_sum = 0.0
        model.train()
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            users_np = positive_users[indices]
            positives_np = positive_items[indices]
            negatives_np = sample_unobserved_items(
                rng=rng, user_indices=users_np, observed=observed
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            positive_scores = model.score_pairs(users, positives)
            negative_scores = model.score_pairs(users, negatives)
            pairwise_loss = F.softplus(negative_scores - positive_scores)
            objective = pairwise_loss.mean()
            if config.regularization:
                objective = objective + config.regularization * model.regularization_per_triplet(
                    users, positives, negatives
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            pairwise_sum += float(pairwise_loss.detach().sum())
            objective_sum += float(objective.detach()) * len(indices)
        epoch_history.append(
            {
                "epoch": epoch,
                "training_pairwise_loss": pairwise_sum / len(positive_users),
                "training_objective": objective_sum / len(positive_users),
            }
        )

    trained_user_mask = np.zeros(len(user_ids), dtype=np.bool_)
    trained_user_mask[np.unique(positive_users)] = True
    with torch.no_grad():
        untrained = torch.as_tensor(
            np.flatnonzero(~trained_user_mask), dtype=torch.long, device=device
        )
        if len(untrained):
            model.user_factors.weight[untrained] = 0.0

    return FittedBPR(
        model=model,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        epoch_history=epoch_history,
        training_seconds=time.perf_counter() - started,
        positive_rows=int(len(positive_users)),
        users_without_positive_training=int((~trained_user_mask).sum()),
        config=config,
    )


@torch.no_grad()
def evaluate_bpr_stage(
    fitted: FittedBPR,
    *,
    full_train: pd.DataFrame,
    evaluation: pd.DataFrame,
    prior_evaluation: pd.DataFrame | None = None,
    ranking_batch_size: int = 256,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate BPR on the exact train-known catalog under the shared contract."""
    model = fitted.model
    model.eval()
    device = model.item_bias.weight.device
    catalog = fitted.item_ids
    catalog_lookup = set(catalog)
    item_array = np.asarray(catalog, dtype=str)
    history = (
        pd.concat([full_train, prior_evaluation], ignore_index=True)
        if prior_evaluation is not None
        else full_train
    )
    historical_by_user = (
        history.groupby(history["user_id"].astype(str), sort=False)["item_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    evaluation_by_user = {
        str(user): rows
        for user, rows in evaluation.groupby(evaluation["user_id"].astype(str), sort=False)
    }
    eval_users = sorted(evaluation_by_user, key=_id_sort_key)
    rows_out: list[dict[str, Any]] = []

    for start in range(0, len(eval_users), ranking_batch_size):
        batch_users = eval_users[start : start + ranking_batch_size]
        user_indices = torch.as_tensor(
            [fitted.user_to_index[user] for user in batch_users],
            dtype=torch.long,
            device=device,
        )
        score_matrix = model.score_catalog(user_indices).cpu().numpy()
        for row_index, user_id in enumerate(batch_users):
            scores = score_matrix[row_index].copy()
            historical = historical_by_user.get(user_id, set())
            historical_indices = [
                fitted.item_to_index[item]
                for item in historical
                if item in fitted.item_to_index
            ]
            scores[historical_indices] = -np.inf
            ranked = top_k_items(item_array, scores, 20)
            user_evaluation = evaluation_by_user[user_id]
            relevant = user_evaluation[
                user_evaluation["rating"] >= fitted.config.relevance_threshold
            ]
            gains = {
                str(row.item_id): rating_gain(
                    float(row.rating), threshold=fitted.config.relevance_threshold
                )
                for row in relevant.itertuples(index=False)
                if str(row.item_id) in catalog_lookup and str(row.item_id) not in historical
            }
            cold = {
                str(item)
                for item in relevant["item_id"].astype(str)
                if str(item) not in catalog_lookup
            }
            rows_out.append(
                {
                    "user_id": user_id,
                    "num_candidates": len(catalog) - len(historical_indices),
                    "num_relevant": len(gains),
                    "num_cold_relevant": len(cold),
                    **ranking_metrics(ranked, gains),
                }
            )

    per_user = pd.DataFrame.from_records(rows_out)
    evaluable = per_user[per_user["num_relevant"] > 0]
    summary = {
        "users_total": int(len(per_user)),
        "users_evaluable": int(len(evaluable)),
        "users_without_eligible_relevant_items": int(len(per_user) - len(evaluable)),
        "cold_relevant_events": int(per_user["num_cold_relevant"].sum()),
        **{
            metric: float(evaluable[metric].mean())
            for metric in ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
        },
    }
    return summary, per_user


def evaluate_bpr_validation(
    fitted: FittedBPR,
    *,
    full_train: pd.DataFrame,
    validation: pd.DataFrame,
    ranking_batch_size: int = 256,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validation-only wrapper; deliberately exposes no test-stage argument."""
    return evaluate_bpr_stage(
        fitted,
        full_train=full_train,
        evaluation=validation,
        prior_evaluation=None,
        ranking_batch_size=ranking_batch_size,
    )


def run_bpr_validation(
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

    settings = dict(modeling["bpr_mf"])
    run_config = BPRRunConfig(
        dimension=dimension,
        regularization=regularization,
        learning_rate=float(settings["learning_rate"]),
        batch_size=int(settings["batch_size"]),
        max_epochs=int(settings["smoke_epochs"] if smoke else settings["max_epochs"]),
        negatives_per_positive=int(settings["negatives_per_positive"]),
        relevance_threshold=float(settings["relevance_threshold"]),
        seed=seed,
        device=str(settings["device"]),
        torch_num_threads=int(settings["torch_num_threads"]),
    )
    fitted = fit_bpr_mf(sparse_train, run_config)
    validation_summary, per_user = evaluate_bpr_validation(
        fitted,
        full_train=full_train,
        validation=validation,
        ranking_batch_size=int(settings["ranking_batch_size"]),
    )
    run_id = (
        f"d{dimension}_lambda{regularization:g}_seed{seed}"
        + ("_smoke" if smoke else "")
    )
    output_dir = (
        _resolve(root, modeling["development_output_root"])
        / dataset_name
        / retention_label
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_parquet(output_dir / "validation_per_user.parquet", index=False)
    summary = {
        "schema_version": 1,
        "stage": "bpr_development_validation_only",
        "dataset": dataset_name,
        "retention_label": retention_label,
        "run_id": run_id,
        "smoke": smoke,
        "run_config": asdict(run_config),
        "training": {
            "rows": int(len(sparse_train)),
            "positive_rows": fitted.positive_rows,
            "users": len(fitted.user_ids),
            "items": len(fitted.item_ids),
            "users_without_positive_training": fitted.users_without_positive_training,
            "epochs_completed": len(fitted.epoch_history),
            "seconds": fitted.training_seconds,
            "epoch_history": fitted.epoch_history,
        },
        "validation": {"ranking": validation_summary},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--modeling-config", type=Path, default=DEFAULT_BPR_MODELING_PATH)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["movielens_1m", "ciaodvd", "yelp_tampa_restaurants"],
    )
    parser.add_argument(
        "--retention", required=True, choices=["r100", "r075", "r050", "r025"]
    )
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--regularization", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(
        run_bpr_validation(
            config_path=args.config,
            modeling_path=args.modeling_config,
            dataset_name=args.dataset,
            retention_label=args.retention,
            dimension=args.dimension,
            regularization=args.regularization,
            seed=args.seed,
            smoke=args.smoke,
        )
    )


if __name__ == "__main__":
    main()
