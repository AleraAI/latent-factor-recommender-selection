# Report reproduction

## Checks without source datasets

This repository supports inspection of the implementation and configurations,
verification of original selection locks, synthetic-data regression tests,
and regeneration of revision reports from fixed aggregate results.

From the repository root, after installing the documented dependencies:

```bash
MPLBACKEND=Agg pytest -q tests/test_latent_factor_*.py
python scripts/latent_factor/verify_selection_locks.py
python scripts/latent_factor/verify_manuscript_tables.py --check-report
python scripts/latent_factor/verify_chart_data_integrity.py \
  --check-report outputs/latent_factor_beyond_rmse/revision_analysis/chart_data_integrity.json
python scripts/latent_factor/verify_release.py
python scripts/latent_factor/regenerate_aggregate_artifacts.py --output regenerated
```

The regeneration directory must not already exist. The wrapper isolates its
outputs, checks that aggregate inputs remain unchanged, and regenerates eight
revision tables, five Pareto figures (PDF and PNG), and four coverage CSVs with
six matching LaTeX tables. Tables are written under
`regenerated/tables/{review_r1,coverage_review}/`, figures under
`regenerated/figures/review_r1/`, and companion reports under
`regenerated/reports/` and `regenerated/coverage/report_tables/`.
Coverage reports are checked against the included records and TeX fixtures in
`tests/fixtures/tables/`. No download, training, checkpoint scoring, user-level
resampling, or new statistical inference occurs. The original 32-table
verifier and the revision-report checks have separate scopes.

### Input map

- `output/latent_factor_major_revision_2026-09-14/selector_analysis/` contains
  fixed validation selections, test outcomes, paired contrasts, and provenance.
- The same revision directory's `protocol_audit/`, `search_boundary/`,
  `report_tables/`, and `user_resampling/` contain configuration metadata,
  matched-budget analyses, aggregate reports, and selection-stability results.
- `output/latent_factor_coverage_revision_2026-09-19/report_tables/` contains
  the seven changed original-margin ablation comparisons with paired intervals,
  all 128 constrained outcomes, margin-exceedance summaries, and adverse cases.
- `scripts/latent_factor/revision_report_tables.py`, `revision_pareto_plots.py`,
  and `coverage_report_tables.py` implement report generation.

## Statistical interpretation

The revision compares four selectors over 16 dataset–retention conditions and
four declared margins: 0.001, 0.0025, 0.005, and 0.010. The 128 constrained
outcomes are two rules × four margins × 16 conditions, not 128 independent
experiments. The 256-row outcome file also includes the repeated RMSE and NDCG
comparator roles. This sensitivity analysis reuses inspected test partitions;
it is not additional prospective validation or a replacement for the originally
declared 0.005 margin.

The original-margin uncertainty-aware and mean-regret rules choose different
configurations in seven conditions. Reported intervals expose uncertainty and
unfavorable cases rather than implying universal improvement. Point regret
above a margin and a two-sided interval's upper endpoint above a margin are
different descriptive quantities; neither is the original prospective
one-sided decision criterion or an estimated deployment-risk guarantee.

The frozen validation selections retain a historical `admissible` column
referring only to the original uncertainty-aware gate at 0.005. The labeled
companion file renames it; current selector/margin admissibility uses the
explicit current-rule fields or the specified inequality. Do not interpret
the historical column as the admissibility result for every margin.

Global RMSE is event-weighted within each seed before averaging over seeds;
mean-user RMSE weights users equally. Prediction vectors are not averaged
across seeds. Revision paired intervals are conditional on the fitted models,
use 2,000 shared user draws, and are unadjusted. They are distinct from the
original 10,000-draw prospective analysis. Report generation copies stored
intervals; it does not recompute them.

Regret is comparator NDCG minus selected-model NDCG, so its sign differs from
a selected-minus-comparator improvement. Relative regret divides by the
recorded comparator's test NDCG, without an artificial epsilon. Identical
configuration comparisons have zero differences and zero interval endpoints.

## What requires separately authorized local artifacts

Complete training and statistical inference require additional artifacts:

- Licensed raw datasets and exact processed/split tables, including candidate
  and relevance contracts.
- Original development/confirmation run summaries, validation and test
  per-user sufficient statistics, and any models needed for rescoring.
- Original hash-bound manifests and fitting plans. Some contain
  machine-specific paths or privacy-sensitive identifier lists and are not
  distributed in canonical form here.
- The matching Python/PyTorch runtime and dependency artifacts. The exact
  historical development wheel is not included or claimed to be retrievable
  from a package index.

These are not all supplied here. The source interfaces can be inspected with:

```bash
python -m src.latent_factor_study.major_revision_selection --help
python -m src.latent_factor_study.major_revision_fits --help
```

Supply missing artifacts only from verified, authorized sources. Do not
fabricate omitted files from aggregate values or replace historical hashes
with transformed metadata.

Fresh experiments must use separately versioned output/configuration
locations. Do not overwrite included grids or change old lock hashes to force
verification to pass. Training time is part of the uncertainty-aware and
mean-constrained tie hierarchy and grid provenance; rerunning a fit can change
recorded times even if numerical metrics agree. Original single-objective ties
use the metric, smaller dimension, then larger regularization;
uncertainty-aware and mean-constrained selection additionally use recorded
mean training time after RMSE.

Full user-resampling selection stability was evaluated on Epinions, whose
saved validation outputs support both prediction and ranking aggregation. The
original three development datasets do not provide equivalent saved prediction
sufficient statistics; this repository does not claim that analysis for them.
Calibration comparisons across all selectors are also not supplied.

## Provenance and privacy

`provenance/source_file_map.json` documents the earlier cleaned revision export,
distinguishing exact copies from transformed metadata; it is not the current
repository's complete membership manifest. Exact selection/configuration lock
bytes have not been rewritten. Files under `provenance/relocated/` are marked
`not_canonical_original` and retain source hashes and transformation
descriptions. They cannot satisfy the original file's checksum and must not
replace it.

Identifier lists or machine paths were omitted in those projections. The
current working tree supplies no per-user records, prediction vectors, or
checkpoints. Included bootstrap NPZ files contain aggregate resampling
statistics. A filename or hash referencing an omitted artifact does not mean
the artifact is included, and a source checksum does not grant redistribution
rights.

The 90-run additional-fit queue and selections were fixed before those fits;
190 matching configuration–seed evaluations were reused. Per-fit data hashes
were initially recorded with outputs. The independent cross-dataset audit was
added after fitting began and verified all 24 required unique data files and
90 completed fits before analysis integration. It is not described as an
all-files pre-fit audit. Two historical replay checks matched their original
outputs, without establishing a universal reproduction guarantee.

## Verification scope

`manifests/release_files_sha256.csv` inventories repository files, excluding the
manifest itself. `scripts/latent_factor/verify_release.py` checks the current
package against that inventory and its distribution rules. The table verifier
resolves historical manuscript table paths to `tests/fixtures/tables/` without
rewriting the stored CSV/JSON evidence. These checks do not require manuscript
source, publisher templates, or compiled publication files.

To create a standalone package in a new destination:

```bash
python scripts/latent_factor/build_github_release.py --output /path/to/new-release
python /path/to/new-release/scripts/latent_factor/verify_release.py --strict
```

The destination must not exist. The builder uses this repository's files; it
does not require the original authoring workspace. Package creation is not a
publication or an archival deposit.

See [`environment/README.md`](../environment/README.md) for saved runtime
evidence, unavailable ancillary historical pins, and the separate reporting
smoke-test environment. Regression tests and deterministic aggregate
regeneration are bounded software/reporting checks, not a clean-room full
refit, a fresh prospective study, or proof of public archival availability.
The [article DOI](https://doi.org/10.3390/computers15100703) identifies the
publication, not a separate software archive. See
[distribution and archival scope](../ARTIFACT_DEPOSIT.md).
