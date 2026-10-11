# Results-to-artifact index

This index maps the article's analyses to included machine-readable evidence.
Paths are relative to the repository root. CSV/JSON results and hash-bound
locks are the evidence sources; generated TeX files are reporting fixtures.
Per-user records and complete run archives are not supplied, even where
historical metadata records their paths or hashes.

For the first table, `O/` means `outputs/latent_factor_beyond_rmse/`.
Configuration filenames are under `configs/latent_factor_beyond_rmse/`.

## Original analyses and robustness

| Analysis | Included evidence |
| --- | --- |
| Processed cohorts and chronological splits | `data/processed/latent_factor_beyond_rmse/*/manifest.json`; `data/processed/latent_factor_framework_epinions/epinions/manifest.json` |
| Realized retention, degree constraints, and catalog repair | `data/splits/latent_factor_beyond_rmse/*/sparsity_manifest.json`; `data/splits/latent_factor_framework_epinions/epinions/sparsity_manifest.json` |
| Complete 2,100-run explicit-MF development landscape and differing RMSE/NDCG choices in 11 of 12 conditions | `O/development/development_landscape_summary.csv`; `selection_lock.yaml` |
| Grid boundaries, winner gaps, seed stability, one-SE sets, and Pareto membership | `O/revision_analysis/configuration_diagnostics.csv`; `O/revision_analysis/pareto_configurations.csv` |
| Locked-test explicit-MF ranking regret, intervals, and adjusted tests | `O/confirmation/analysis/mf_selection_regret.csv`; aggregate resample NPZ files in that directory |
| Interval-aware evidence classes and prediction-regret estimands | `O/revision_analysis/practical_evidence.csv`; `O/revision_analysis/prediction_regret.csv` |
| Retrospective framework construction selections and outcomes | `O/framework_construction/selections/*/*/framework_selection.json`; `O/framework_construction/analysis/construction_framework_results.csv` |
| Prospective Epinions grid, selections, and absolute-margin decision | `epinions_prospective_*lock.yaml`; `O/prospective_epinions/development/epinions/*/grid_runs.csv`; adjacent `framework_selection.json` files; `O/prospective_epinions/confirmation/analysis/prospective_external_validation.csv` |
| Paired ranking-regret contrasts against full retention | `O/revision_analysis/retention_contrasts.csv` |
| Candidate/relevant-item counts and cold-item events | `O/revision_analysis/candidate_diagnostics.csv` |
| NDCG/Recall cutoff, clipping, and MAE sensitivities | `O/confirmation/secondary_analysis/secondary_metric_differences.csv`; canonical clipped RMSE from `O/confirmation/analysis/mf_selection_regret.csv` |
| Training-activity heterogeneity | `O/confirmation/secondary_analysis/activity_stratified_ndcg_regret.csv` |
| Bias-only and popularity comparisons | `baseline_selection_lock.yaml`; `O/baseline_confirmation/analysis/baseline_comparison.csv` |
| BPR-MF versus explicit MF/popularity and BPR selection diagnostics | `bpr_selection_lock.yaml`; `O/bpr_confirmation/analysis/bpr_boundary_comparison.csv`; `O/revision_analysis/bpr_selection_diagnostics.csv` |
| Popularity concentration, persistence, and head/mid/tail exposure | `O/revision_analysis/popularity_diagnostics.csv` |
| Popularity tie-order sensitivity across five SHA-256 permutations | `O/second_review_analysis/popularity_tie_sensitivity.csv`; `O/second_review_analysis/popularity_tie_sensitivity_by_permutation.csv` |
| Five-star relevance sensitivity | `O/five_star_sensitivity/analysis/mf_selection_regret.csv` |
| Completed-epoch audit for primary explicit MF | `O/second_review_analysis/primary_epoch_audit.csv` |
| Fresh-seed fitting times | `O/computational_analysis/confirmation_training_cost_runs.csv`; `O/computational_analysis/confirmation_training_cost_summary.csv` |
| Development fit work and exact candidate-scoring work | `O/revision_analysis/selection_cost.csv` |
| Expanded explicit-MF choices, budget, and convergence | `O/robustness/analysis/expanded_selection_diagnostics.csv`; `budget_selection_diagnostics.csv`, `explicit_convergence_diagnostics.csv`, `selected_training_curves.csv`, and `selected_validation_curves.csv` in the same directory |
| Expanded 100-epoch post-confirmation test regret | `O/robustness/analysis/expanded_post_confirmation_regret.csv` |
| Repeated-retention outcomes and variance components | `O/robustness/analysis/repeated_post_confirmation_regret_by_sample.csv`; `repeated_post_confirmation_regret_summary.csv` and `repeated_variance_components.csv` in the same directory |
| Independently tuned BPR and fixed-configuration negative-pool sensitivity | `O/robustness/analysis/bpr_independent_selection_diagnostics.csv`; `O/robustness/analysis/bpr_negative_sampler_post_confirmation.csv` |
| Explicit-negative support and fallback diagnostics | `O/second_review_analysis/bpr_negative_pool_diagnostics.csv` |
| Expanded explicit-MF and BPR search cost | `O/second_review_analysis/robustness_search_cost.csv` |

## Four-margin comparisons and coverage

Paths in this table are relative to
`output/latent_factor_major_revision_2026-09-14/`, except where stated.

| Analysis | Included evidence |
| --- | --- |
| Fixed validation selections and test outcomes | `selector_analysis/validation_selections.csv`; `validation_selections_labeled.csv`, `selection_outcomes.csv`, and `selection_freeze.json` in the same directory |
| Paired selector contrasts | `selector_analysis/paired_test_contrasts.csv` |
| Configuration and cohort audit | `protocol_audit/configuration_registry.csv`; `dataset_summary.csv` and `retention_counts.csv` in the same directory |
| Matched search-budget comparison | `search_boundary/matched_budget_summary.csv`; `search_boundary/matched_budget_condition_results.csv` |
| Epinions user-resampling selection stability | `user_resampling/selection_stability_summary.csv`; aggregate draw-level selections, frequencies, and agreement files in the same directory |
| Original-margin ablation intervals, adverse cases, and margin exceedances | Four CSV reports under `output/latent_factor_coverage_revision_2026-09-19/report_tables/` |
| Additional-fit completion and data audit | `selector_analysis/fits/completion.json`; `selector_analysis/fits/data_integrity_audit_final.json` |
| Recorded runtime of the 90 additional fits | `environment/revision-runtime-evidence.json`; `environment/revision-run-runtime-inventory.csv` at repository root |

These sensitivity analyses reuse inspected test partitions. The 128
constrained outcomes are two rules × four margins × 16 conditions, not 128
independent experiments. See [report reproduction](REVISION_REPRODUCTION.md)
for interval estimands and the historical admissibility-column caveat.

## Reporting checks

- `tests/fixtures/tables/` preserves the generated TeX subtree, including
  original tables and the `review_r1/` and `coverage_review/` subdirectories.
- `manifests/manuscript_table_verification.csv` records the original 32-table
  source, row-count, and checksum checks. Its table paths point to the
  relocated fixtures; `scripts/latent_factor/verify_manuscript_tables.py`
  verifies their contents against the unchanged evidence sources.
- `O/revision_analysis/chart_data_integrity.json` and
  `scripts/latent_factor/verify_chart_data_integrity.py` check figure-source
  data integrity; they are not a comparison with the publisher's rendered PDF.
- `manifests/release_files_sha256.csv` and
  `scripts/latent_factor/verify_release.py` describe and check the current
  package, separately from original experimental locks.

The Epinions pre-development lock covers the raw archive, dataset/modeling
configurations, and selector implementation, not every pipeline module.
Post-analysis release checksums do not expand that earlier lock's scope.
