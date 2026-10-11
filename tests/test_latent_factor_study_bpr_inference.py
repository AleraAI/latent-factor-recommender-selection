from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.latent_factor_study.bpr_inference import _mean_role_users
from src.latent_factor_study.data import sha256_file


def test_bpr_role_users_are_averaged_across_five_fresh_seeds(tmp_path: Path) -> None:
    runs = []
    for seed in (1, 2, 3, 4, 5):
        path = tmp_path / f"users_{seed}.parquet"
        pd.DataFrame(
            {"user_id": ["u1", "u2"], "ndcg@10": [seed / 10, seed / 20]}
        ).to_parquet(path, index=False)
        runs.append(
            {
                "selection_role": "bpr_selected",
                "seed": seed,
                "per_user_path": path.name,
                "per_user_sha256": sha256_file(path),
            }
        )
    frame = _mean_role_users(
        tmp_path,
        json.loads(json.dumps({"runs": runs})),
        role_key="selection_role",
        role_value="bpr_selected",
        roles_are_lists=False,
    ).set_index("user_id")
    assert frame.loc["u1", "ndcg@10"] == 0.3
    assert frame.loc["u2", "ndcg@10"] == 0.15
