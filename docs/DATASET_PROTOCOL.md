# Dataset protocol

The framework was constructed on MovieLens 1M, CiaoDVD, and a fixed
Yelp–Tampa restaurant cohort. Epinions was held out until the selection rule
and external decision criterion were frozen. The configurations and manifests
are the authoritative records of dataset versions, preprocessing, and artifact
hashes; this guide summarizes their contract.

## Frozen sources and scope

| Dataset | Source or scoped records before deduplication and 10-core filtering | Role |
| --- | --- | --- |
| MovieLens 1M | Stable 1,000,209-rating release; 6,040 users and 3,706 movies | Framework construction |
| CiaoDVD | 72,665 dated ratings; 17,615 users and 16,121 DVDs | Framework construction |
| Yelp–Tampa restaurants | 303,232 reviews; 114,328 users and 2,960 businesses | Framework construction |
| Epinions timestamped ratings | 922,267 ratings; 22,164 users and 296,277 products | Prospective external validation |

The Yelp cohort uses exact `city == Tampa`, `state == FL`, and the exact
comma-separated category token `Restaurants`. It uses the official bundle
packaged on 7 January 2025, not a current or substitute download. Epinions
contains 352,598 ratings from 12,168 users on 11,283 products after the frozen
deduplication and iterative 10-core filter.

Expected archive SHA-256 values are preserved in
`configs/latent_factor_beyond_rmse/datasets.yaml` and
`configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml`:

| Archive | SHA-256 |
| --- | --- |
| MovieLens 1M | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |
| CiaoDVD | `e36e06034dcf767f0f5cac120b03f262caef68cba86690e451571c13fa3236dd` |
| Yelp JSON bundle | `47dd6e4d279ac9d8734ddc30bfb3d78e571b9df4bb95923d7acf9a6ef3d8a4ab` |
| Epinions timestamped ratings | `987c590663ac9c0b0148012530a7a5545fbf5ec44ad1a5524896dab3135714f1` |

Obtain the archives under their source terms; see
[third-party notices](../THIRD_PARTY_NOTICES.md). Raw and processed interaction
tables and per-user records are not supplied in the current working tree.
Manifest paths and hashes may identify omitted artifacts.

Public split manifests retain counts and dataset-file checksums but replace
repaired-interaction hash lists with omitted-entry counts. These projections
are labeled `_public_export` and indexed in
`provenance/public_metadata_redactions.json`; they are not byte-identical
copies of the full local manifests. The configurations and selection locks
remain unchanged.

## Preprocessing and chronological splits

1. Load user ID, item ID, explicit rating, timestamp, and a stable source row
   number. Scope Yelp businesses before loading their reviews.
2. Resolve duplicate user–item pairs by latest timestamp, with source row as a
   deterministic tie breaker.
3. Apply simultaneous iterative 10-core filtering until both user and item
   degree constraints reach a fixed point.
4. Sort each user's history by timestamp, source row, and item ID. For history
   length `n`, use `floor(0.70*n)` as the end of training and
   `floor(0.85*n)` as the end of validation; the remainder is test.
5. Record raw archive and generated artifact hashes, counts, and split metadata.

Test partitions are excluded from model selection. Structural dataset counts
and relevance prevalence do not substitute for a model-selection analysis.

## Controlled retention and ranking candidates

Nominal training retention is 100%, 75%, 50%, or 25%. The construction datasets
use priority seed `20260817`; prospective Epinions uses `20260825`. For a
training history of length `n`, the initial retained count is
`max(3, floor(retention*n + 0.5))`. SHA-256 priorities derived from the seed,
retention, and immutable interaction ID determine selected rows.

A deterministic repair restores a training row for every item that would
otherwise disappear from the full-training catalog. It does not inspect
validation or test identities. Repair changes realized retention, so manifest
row counts and rates take precedence over nominal labels. The separate
post-confirmation repeated-retention protocol uses nested families; it must
not be confused with this original sampling contract.

The exact ranking candidate catalog is the thinned training catalog minus the
user's known history. Full-training interactions removed for sparsity remain
excluded as history; validation history is also excluded at test time.
Relevant held-out events on items absent from the training catalog are labeled
cold and are not assigned random, untrained scores.

Primary relevance is a rating of at least four, with graded gains of one for
four stars and three for five stars. Five-star-only relevance is a sensitivity
analysis, not a replacement for the locked primary definition.

## Preparing authorized local data

Run from a separate experiment checkout after placing the exact archives at
the configured paths:

```bash
python -m src.latent_factor_study.data --dataset all
python -m src.latent_factor_study.sparsity --dataset all
python -m src.latent_factor_study.data \
  --config configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml \
  --dataset epinions
python -m src.latent_factor_study.sparsity \
  --config configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml \
  --dataset epinions
```

Changing source versions, scope, ordering, filtering, or sampling creates a new
experiment. Preserve the original manifests and lock bytes. The
[experiment guide](EXPERIMENTS.md) describes the downstream selection gates.
