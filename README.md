# Prediction–Ranking Trade-Offs in Latent-Factor Recommenders

[![Tests](https://github.com/AleraAI/latent-factor-recommender-selection/actions/workflows/tests.yml/badge.svg)](https://github.com/AleraAI/latent-factor-recommender-selection/actions/workflows/tests.yml)
[![Code: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC%20BY%204.0-lightgrey.svg)](LICENSES/CC-BY-4.0.txt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](pyproject.toml)

Research software and reproducibility evidence for:

Mohamed Elrfaey and Thomas Aaron Gulliver. **Prediction–Ranking Trade-Offs in
Latent-Factor Recommenders: An Uncertainty-Aware Selection Framework.**
*Computers* **2026**, *15*(10), 703. Published 11 October 2026.
[Article](https://www.mdpi.com/2073-431X/15/10/703) ·
[Supplementary material](https://www.mdpi.com/article/10.3390/computers15100703/s1) ·
[DOI: 10.3390/computers15100703](https://doi.org/10.3390/computers15100703).

The study compares prediction- and ranking-based model selection on MovieLens
1M, CiaoDVD, a fixed Yelp–Tampa restaurant cohort, and prospectively held-out
Epinions data. It includes a 2,100-run explicit-MF construction grid, a separately
locked 700-run Epinions grid, bias/popularity and BPR-MF comparisons, and
post-confirmation sensitivity analyses.

This repository contains the implementation, frozen configurations and
selection locks, aggregate results, dataset manifests, and verification tools.
Generated LaTeX tables are retained as test fixtures. Manuscript source,
compiled publication files, raw and processed interaction tables, per-user
records, and model checkpoints are not supplied in the current working tree.
The article DOI identifies the publication, not a software archive.

## Quick start

```bash
git clone https://github.com/AleraAI/latent-factor-recommender-selection.git
cd latent-factor-recommender-selection

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt
python -m pip install -e .

MPLBACKEND=Agg pytest -q tests/test_latent_factor_*.py
python scripts/latent_factor/verify_selection_locks.py
python scripts/latent_factor/verify_manuscript_tables.py --check-report
python scripts/latent_factor/verify_chart_data_integrity.py \
  --check-report outputs/latent_factor_beyond_rmse/revision_analysis/chart_data_integrity.json
python scripts/latent_factor/verify_release.py
```

These checks use included evidence and synthetic test data; they do not require
the source datasets. They verify software and reporting consistency, not a
complete refit of the study.

## Regenerate reports from aggregates

```bash
python scripts/latent_factor/regenerate_aggregate_artifacts.py --output regenerated
```

Choose an output directory that does not already exist. This regenerates eight
revision tables, five Pareto figures in PDF and PNG, and four coverage CSVs
with six matching LaTeX tables. Table and figure outputs are organized under
`regenerated/tables/` and `regenerated/figures/`. The command checks that its
aggregate inputs remain unchanged and compares coverage reports with stored
results. It performs no downloads, fitting, checkpoint scoring, or statistical
resampling.

See [report reproduction](docs/REVISION_REPRODUCTION.md) for the input map,
estimands, and verification limits, and the [results index](docs/RESULTS_INDEX.md)
for links between reported analyses and their evidence.

## Results and interpretation

The locked Epinions evaluation met the predeclared absolute-margin ranking
criterion in all four retention conditions. Estimated NDCG@10 regret was
0.000753–0.001360, with one-sided 95% upper bounds of 0.001050–0.001590,
below the declared 0.005 margin. NDCG-selected performance was only
0.002186–0.002669, however: those losses equal 28.2–62.2% of the comparator
score. This is an absolute-margin result, not proportional ranking preservation
or evidence that the margin suits every application. Mean-user RMSE improved
by 0.0404–0.0715 relative to direct NDCG selection.

MovieLens, CiaoDVD, and Yelp results informed framework construction and are
retrospective. The four-margin sensitivity analysis (0.001, 0.0025, 0.005, and
0.010), mean-regret ablation, and other post-confirmation analyses reuse
inspected test partitions. They are not additional prospective validation.

## Repository map

| Path | Contents |
| --- | --- |
| `src/latent_factor_study/` | Preprocessing, models, evaluation, inference, and reporting |
| `configs/latent_factor_beyond_rmse/` | Dataset/model configurations and immutable locks |
| `scripts/latent_factor/` | Experiment runners, report generators, and verifiers |
| `tests/` | Reproducibility tests and generated-table fixtures |
| `data/**/manifest.json` | Source, preprocessing, split, and retention metadata |
| `outputs/latent_factor_beyond_rmse/` | Original and robustness aggregate evidence |
| `output/latent_factor_major_revision_2026-09-14/` | Selector comparisons, paired contrasts, search-budget results, and provenance |
| `output/latent_factor_coverage_revision_2026-09-19/` | Paired intervals, adverse cases, and margin-exceedance summaries |
| `manifests/` | File checksums, external-artifact inventory, and table verification |
| `environment/` | Recorded runtime evidence and dependency guidance |
| `docs/` | Dataset protocol, experiment sequence, results index, and report reproduction |

## Data and full experiments

Acquire the source datasets under their respective terms and place them at the
paths in `configs/latent_factor_beyond_rmse/datasets.yaml` and
`configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml`. These
frozen configurations specify the expected archive hashes and exact Yelp
cohort. See the [dataset protocol](docs/DATASET_PROTOCOL.md) for preprocessing,
chronological splits, relevance, and catalog rules.

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

Run new experiments in a separate checkout with separately versioned outputs.
Do not overwrite historical locks or change hashes to make a rerun pass.
Dataset substitutions, altered search grids, stopping rules, or estimands
constitute new experiments. The [experiment guide](docs/EXPERIMENTS.md)
describes the execution sequence and required external artifacts. The included
aggregates are not enough to repeat user-level inference or every training run.

## Provenance and environment

The recorded training environment used Python 3.10.9, PyTorch
2.1.0.dev20230521, pandas 1.5.3, and NumPy 1.21.5. The historical PyTorch
wheel/container is not bundled; some ancillary dependency versions were not
recorded. Current installation requirements are compatibility requirements,
not a complete historical lock. See [environment notes](environment/README.md).

Frozen configurations retain historical titles and machine-facing terminology
because their bytes are part of the selection-lock record. The pre-development
Epinions lock covers its source archive, dataset/modeling configurations, and
selector source, not every pipeline module. It is an internal protocol freeze,
not public preregistration. The release manifest separately inventories the
current package.

Experimental results and selection locks retain their original contents,
including references to artifacts that are not distributed. Public preprocessing
manifests omit row-level interaction hash lists while preserving counts and
dataset checksums; `provenance/public_metadata_redactions.json` records those
projections and their source hashes. Included bootstrap
NPZ arrays contain aggregate resampling statistics, not per-user records.
See [distribution and archival scope](ARTIFACT_DEPOSIT.md) for the package
boundary and unavailable artifacts.

## Citation

Please cite the article above when using this work. Machine-readable software
and article metadata are available in [CITATION.cff](CITATION.cff).
No separate software-archive DOI is recorded.

## Licensing

Original code, configurations, tests, and analysis scripts are licensed under
the [Apache License 2.0](LICENSE). Author-created documentation, figures, and
table content are available under
[Creative Commons Attribution 4.0 International](LICENSES/CC-BY-4.0.txt),
unless otherwise indicated.

These grants do not relicense MovieLens, CiaoDVD, Yelp, Epinions, dependencies,
or other third-party material. See [third-party notices](THIRD_PARTY_NOTICES.md)
and [NOTICE](NOTICE).
