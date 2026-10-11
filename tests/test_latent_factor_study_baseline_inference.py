from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.latent_factor_study.baseline_inference import _popularity_users
from src.latent_factor_study.data import sha256_file


def test_popularity_inference_verifies_per_user_hash(tmp_path: Path) -> None:
    path = tmp_path / "users.parquet"
    pd.DataFrame({"user_id": [1], "ndcg@10": [0.5]}).to_parquet(path, index=False)
    manifest = json.loads(
        json.dumps(
            {
                "popularity": {
                    "per_user_path": path.name,
                    "per_user_sha256": sha256_file(path),
                }
            }
        )
    )
    frame = _popularity_users(tmp_path, manifest)
    assert frame.loc[0, "user_id"] == "1"
