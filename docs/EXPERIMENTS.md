# Experiment guide

This guide summarizes the frozen experimental contracts and execution
sequence. The current package supports implementation inspection and
aggregate-report reproduction. Full fitting and inference require authorized
datasets and omitted run-level or per-user artifacts; the commands below are
not a claim that every stage can run from this package alone.

Use a separate checkout and separately versioned outputs for new experiments.
Never overwrite a historical lock or revise a stored hash to force a rerun to
match. See [dataset protocol](DATASET_PROTOCOL.md) for preprocessing, splits,
candidate catalogs, and relevance.

## Primary explicit-MF development and locked confirmation

Development reads training and validation data, plus full-training history
needed for candidate exclusions. Its evaluation interface has no test-data
argument. Each construction dataset–retention condition contains 35
dimension–regularization configurations and five development seeds:
101, 211, 307, 401, and 503. The three datasets and four retention conditions
therefore produce 2,100 development runs.

The fixed budget uses Adam, learning rate 0.01, batch size 4096, a 20-epoch
ceiling, and the declared five-stale-epoch early-stop rule. A preceding
100-epoch Ciao diagnostic did not satisfy the convergence tolerance, motivating
a common finite budget rather than a per-objective stopping choice. The epoch
audit records 2,084 development runs at 20 epochs, five at 19, and eleven at 18;
all 115 primary confirmation fits reached 20 epochs.

Development summaries compare negative RMSE and NDCG@10 over the 35 seed-mean
configurations using Spearman correlation, Kendall's tau-b, top-three overlap,
and cross-objective ranks. Each condition must have all configurations and
seeds before selection. Original single-objective ties use the metric,
smaller dimension, then larger regularization.

`configs/latent_factor_beyond_rmse/selection_lock.yaml` hashes dataset/modeling
configurations, the combined development analysis, and selection files. It
fixes fresh confirmation seeds 601, 701, 809, 907, and 1009, disjoint from
development seeds. Confirmation resolves test artifacts only after the
selection gate. The declared 0.005 absolute NDCG@10 tolerance is an
author-specified decision margin, not demonstrated application utility.

With the required external artifacts available:

```bash
python -m src.latent_factor_study.grid --dataset all --retention all
python -m src.latent_factor_study.development_analysis
python scripts/latent_factor/verify_selection_locks.py
python -m src.latent_factor_study.confirmation
python -m src.latent_factor_study.inference
```

These commands use the existing protocol. A new experiment needs its own
configuration/output namespace and independently recorded selections.

## Retrospective uncertainty-aware framework construction

For each configuration, the selector pairs validation NDCG@10 by development
seed with the validation-NDCG winner. It forms a one-sided 95% Student-*t*
upper bound on regret and admits configurations whose bound is at most 0.005.
It minimizes mean validation RMSE within that set; lower mean fitting time,
smaller dimension, and larger regularization break ties.

MovieLens, CiaoDVD, and Yelp test outcomes had already been inspected when
this framework was constructed. Their selections use validation data, but
fresh-seed test summaries remain labeled `exploratory_not_prospective`.
Identical configurations reuse checksum-verified confirmation runs; seven new
choices required 35 new fits. This is construction evidence, not prospective
external validation.

```bash
python -m src.latent_factor_study.framework_construction all
```

## Prospective Epinions validation

Before Epinions model metrics were inspected, the protocol lock hashed the raw
archive, dataset/modeling configurations, and selector implementation. The
configuration fixes preprocessing, splits, retention seed, 35-point grid,
development and fresh confirmation seeds, margin, uncertainty calculation,
tie breakers, and external decision rule.

The pre-development lock does not cover every pipeline module. It is an
internal protocol freeze, not public preregistration or a complete source
snapshot. After the 700 validation-only fits, a second immutable lock binds
all four selections, their grid digests, and the frozen analysis plan.
Verification compares recorded digests against the actual `grid_runs.csv`
files before confirmation can resolve test data.

```bash
bash scripts/latent_factor/run_epinions_prospective_resumable.sh
python scripts/latent_factor/verify_selection_locks.py
python -m src.latent_factor_study.prospective_confirmation
python -m src.latent_factor_study.prospective_inference
```

The released selection lock is historical evidence: verify it rather than
attempting to recreate it in place.

The primary external estimand is NDCG-selected minus uncertainty-aware
mean-user NDCG@10 after averaging five fitted seeds. Ten thousand user-cluster
bootstrap resamples produce a one-sided 95% upper bound. The absolute-margin
criterion requires a bound no greater than 0.005 in every retention condition;
pooling cannot conceal a failed condition. RMSE trade-offs and other ranking
metrics are secondary, with two-sided intervals.

The frozen prospective chain completed on 25 August 2026. All four conditions
met the criterion: regret was 0.000753–0.001360 and upper bounds were
0.001050–0.001590. Comparator NDCG@10 was 0.002186–0.002669, making those
losses 28.2–62.2% of comparator performance. Thus the result does not establish
proportional ranking preservation. Mean-user RMSE improved by
0.0404–0.0715 relative to the NDCG-selected configuration. Historical
machine-facing decision labels remain unchanged in the result files.

## Locked boundary comparisons

BPR-MF has its own selection lock and a 2,100-run development grid using the
same 35 dimension–regularization pairs and five development seeds. It uses
Adam, learning rate 0.01, batch size 4096, and 20 epochs. Retained ratings of at
least four are positives. Each epoch samples one uniform negative per positive
from the retained catalog, excluding every item rated by that user in the
retained artifact. The sampler does not inspect removed full-training rows,
validation, or test. Selection maximizes mean validation NDCG@10, then prefers
smaller dimension and larger regularization. Sixty fresh-seed confirmation
fits support the locked comparison.

The bias-only model is the explicit model with latent dimension zero. Its
five regularizations and five development seeds give 300 development fits.
Popularity counts retained training ratings of at least four and breaks score
ties by ascending item ID; it has no fitted parameters or stochastic seed.
Both use the shared exact train-known-catalog evaluation contract and an
independent selection lock. Five-star relevance was also evaluated using
115 primary fits without reselecting hyperparameters.

```bash
bash scripts/latent_factor/run_locked_followups.sh
```

Contracts are recorded in `bpr_modeling.yaml`, `baseline_modeling.yaml`, and
their selection locks under `configs/latent_factor_beyond_rmse/`.

## Post-confirmation robustness

The separately versioned `configs/latent_factor_beyond_rmse/robustness.yaml`
was frozen after original test results had been inspected. Its analyses are
post-confirmation evidence, not a second untouched confirmation:

- **Expanded explicit MF:** nine dimensions (1–256) and nine regularization
  values (0–1), with five original development seeds and validation
  checkpoints at 20, 50, and 100 epochs.
- **Budget and convergence sensitivity:** training continues to 100 epochs;
  five consecutive relative objective improvements no larger than `1e-5`
  define the diagnostic convergence criterion.
- **Repeated retention:** priority seeds 20260819, 20260829, and 20260907
  create nested 25/50/75/100% families. Catalog repairs at 25% propagate upward.
  Non-full conditions use the expanded 81-point grid and three initialization
  seeds; summaries distinguish retention-sample and initialization variation.
- **Independently tuned BPR:** coarse dimension, regularization, learning-rate,
  and epoch searches, then refinement with one, three, or five negatives per
  positive. Fixed selections are compared under retained-unobserved negatives,
  an oracle full-training-positive exclusion diagnostic, and observed ratings
  below four with retained-unobserved fallback.

Fresh robustness-test seeds are 1201, 1301, 1409, 1511, and 1601.
Outputs are separately located under
`outputs/latent_factor_beyond_rmse/robustness/`, without rewriting primary
locks or confirmation results.

```bash
bash scripts/latent_factor/run_post_confirmation_robustness.sh
python -m src.latent_factor_study.secondary_analysis
python -m src.latent_factor_study.second_review_analysis
```

Later selector comparisons use four margins and a mean-regret ablation. Their
90 additional fits reuse 190 matching configuration–seed evaluations. The
stored paired intervals are conditional on fitted models, use 2,000 shared
user draws, and are unadjusted. They do not replace the original prospective
10,000-draw analysis. Full user-resampling selection stability is supplied for
Epinions only; equivalent saved prediction sufficient statistics are not
available for the three construction datasets. Calibration comparisons across
all selectors are not supplied.

## Reporting and verification

For dataset-free report regeneration and checks, use
[report reproduction](REVISION_REPRODUCTION.md). The
[results index](RESULTS_INDEX.md) identifies included aggregate evidence.
[Environment notes](../environment/README.md) distinguish recorded training
versions from current reporting dependencies and explain why bitwise full
reproduction is not guaranteed.
