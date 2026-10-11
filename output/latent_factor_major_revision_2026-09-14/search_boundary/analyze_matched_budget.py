"""Revision-stage, validation-only grid comparison on saved matched trajectories.

No model is fitted or rescored. The original35 subgrid is reselected from the
post-confirmation expanded runs, not silently substituted for the historical
maximum20-epoch development experiment, which allowed earlier stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_id(row):
    return f"d{int(row['dimension'])}_lambda{float(row['regularization']):g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.repo_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_path = root/'configs/latent_factor_beyond_rmse/modeling.yaml'
    robust_path = root/'configs/latent_factor_beyond_rmse/robustness.yaml'
    original = yaml.safe_load(original_path.read_text())['explicit_mf']
    robust = yaml.safe_load(robust_path.read_text())
    expanded = robust['explicit_expanded']
    datasets = robust['common']['datasets']
    retentions = robust['common']['retentions']
    seeds = robust['common']['selection_seeds']
    budgets = expanded['checkpoint_epochs']
    assert original['selection_seeds'] == seeds
    assert expanded['tie_breakers'] == ['smaller_dimension', 'larger_regularization']
    grids = {
        'original35_subgrid': (original['dimensions'], original['regularizations']),
        'expanded81_grid': (expanded['dimensions'], expanded['regularizations']),
    }
    rows, inputs = [], [original_path, robust_path, Path(__file__).resolve()]
    validation_rows = 0
    for dataset in datasets:
        for retention in retentions:
            path = root/f'outputs/latent_factor_beyond_rmse/robustness/explicit_expanded/{dataset}/{retention}/runs.csv'
            inputs.append(path)
            frame = pd.read_csv(path)
            keys = ['dimension', 'regularization', 'seed', 'epoch']
            assert not frame.duplicated(keys).any(), path
            assert frame.dataset.eq(dataset).all() and frame.retention_label.eq(retention).all()
            assert set(frame.seed) == set(seeds) and set(frame.epoch) == set(budgets)
            assert np.isfinite(frame[['validation_rmse', 'validation_ndcg@10']].to_numpy()).all()
            expected = {(d, float(l), s, e) for d in expanded['dimensions'] for l in expanded['regularizations'] for s in seeds for e in budgets}
            assert set(frame[keys].itertuples(index=False, name=None)) == expected, path
            validation_rows += len(frame)
            for epoch in budgets:
                means = frame[frame.epoch.eq(epoch)].groupby(['dimension', 'regularization'], as_index=False).agg(
                    seed_count=('seed', 'nunique'),
                    mean_validation_rmse=('validation_rmse', 'mean'),
                    mean_validation_ndcg_at_10=('validation_ndcg@10', 'mean'),
                )
                for name, (dims, regs) in grids.items():
                    candidates = means[means.dimension.isin(dims) & means.regularization.isin(regs)]
                    assert len(candidates) == len(dims)*len(regs)
                    rm = candidates.sort_values(['mean_validation_rmse', 'dimension', 'regularization'], ascending=[True, True, False], kind='stable').iloc[0]
                    nd = candidates.sort_values(['mean_validation_ndcg_at_10', 'dimension', 'regularization'], ascending=[False, True, False], kind='stable').iloc[0]
                    record = dict(
                        dataset=dataset, retention_label=retention, epoch=epoch,
                        search_scope=name, configurations=len(candidates), seeds=len(seeds),
                        rmse_selected_id=config_id(rm), ndcg_selected_id=config_id(nd),
                        configuration_disagreement=config_id(rm)!=config_id(nd),
                        validation_ndcg_cost_of_rmse_selection=float(nd.mean_validation_ndcg_at_10-rm.mean_validation_ndcg_at_10),
                        validation_rmse_cost_of_ndcg_selection=float(nd.mean_validation_rmse-rm.mean_validation_rmse),
                    )
                    for role, choice in [('rmse', rm), ('ndcg', nd)]:
                        record.update({
                            f'{role}_dimension': int(choice.dimension),
                            f'{role}_regularization': float(choice.regularization),
                            f'{role}_mean_validation_rmse': float(choice.mean_validation_rmse),
                            f'{role}_mean_validation_ndcg_at_10': float(choice.mean_validation_ndcg_at_10),
                            f'{role}_dimension_boundary': int(choice.dimension) in (min(dims), max(dims)),
                            f'{role}_regularization_boundary': float(choice.regularization) in (min(regs), max(regs)),
                        })
                        record[f'{role}_any_boundary'] = record[f'{role}_dimension_boundary'] or record[f'{role}_regularization_boundary']
                    rows.append(record)
    condition = pd.DataFrame(rows)
    summary = []
    for (scope, epoch), group in condition.groupby(['search_scope', 'epoch'], sort=True):
        summary.append(dict(
            search_scope=scope, epoch=int(epoch), conditions=len(group),
            configurations=int(group.configurations.iloc[0]), seed_count=len(seeds),
            disagreement_conditions=int(group.configuration_disagreement.sum()),
            objective_specific_choices=2*len(group),
            boundary_choices=int(group.rmse_any_boundary.sum()+group.ndcg_any_boundary.sum()),
            rmse_boundary_choices=int(group.rmse_any_boundary.sum()),
            ndcg_boundary_choices=int(group.ndcg_any_boundary.sum()),
            conditions_with_any_boundary=int((group.rmse_any_boundary|group.ndcg_any_boundary).sum()),
            conditions_with_both_boundaries=int((group.rmse_any_boundary&group.ndcg_any_boundary).sum()),
        ))
    by_scope = {}
    for scope, group in condition.groupby('search_scope'):
        changing = []
        for (dataset, retention), conditions in group.groupby(['dataset', 'retention_label']):
            for role in ['rmse', 'ndcg']:
                ids = conditions.sort_values('epoch')[f'{role}_selected_id'].tolist()
                if len(set(ids)) > 1:
                    changing.append(dict(dataset=dataset, retention_label=retention, selector=role, ordered_budgets=budgets, ordered_choices=ids))
        by_scope[scope] = dict(changed_across_any_budget=len(changing), denominator=24, changed_conditions=changing)
    checks = condition.pivot(index=['dataset','retention_label','epoch'], columns='search_scope', values=['rmse_mean_validation_rmse','ndcg_mean_validation_ndcg_at_10'])
    assert (checks[('rmse_mean_validation_rmse','expanded81_grid')] <= checks[('rmse_mean_validation_rmse','original35_subgrid')]).all()
    assert (checks[('ndcg_mean_validation_ndcg_at_10','expanded81_grid')] >= checks[('ndcg_mean_validation_ndcg_at_10','original35_subgrid')]).all()
    run_root = root/'outputs/latent_factor_beyond_rmse/robustness/explicit_expanded'
    weight_files = [p for p in run_root.rglob('*') if p.suffix in {'.pt','.pth','.ckpt','.safetensors','.npz'}]
    notes = dict(
        stage='revision_stage_existing_output_diagnostic',
        revision_environment=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, pyyaml=yaml.__version__),
        source_stage='post_confirmation_robustness',
        data_access='validation only; no test summaries used in this comparison',
        interpretation='The two grid scopes use exactly the same expanded-run trajectories, paired seeds, validation users, candidate catalog, metric definitions, and epoch budgets. This isolates finite search-scope effects at a fixed recorded training budget, not a global optimum or deployment guarantee.',
        original_comparison_boundary='original35_subgrid is a retrospective restriction of expanded trajectories. It is not the original locked development experiment, whose maximum20-epoch tolerance rule sometimes stopped at18/19epochs.',
        averaging='Arithmetic mean of five per-run global clipped RMSE values and five per-run mean-user NDCG values. No prediction ensembling.',
        tie_breaking='For either objective: best mean validation objective, then smaller dimension, then larger regularization. This is not UA tie-breaking.',
        boundary_definition='Within each displayed grid scope, a choice on the minimum/maximum dimension OR minimum/maximum regularization is a boundary choice. Lambda0 is a natural parameter boundary and its selection alone does not prove missing optimality.',
        checkpoint_wording='robustness_explicit.py records validation metric summaries at20/50/100epochs and100training-objective observations. It does not save model weights or per-user validation outputs. Call these budget evaluations, not available rescorable checkpoints.',
        stored_weight_files_found=len(weight_files),
        input_run_rows=validation_rows, underlying_training_trajectories=validation_rows//len(budgets),
        summary=summary, budget_changes=by_scope,
    )
    condition.to_csv(out/'matched_budget_condition_results.csv', index=False)
    pd.DataFrame(summary).to_csv(out/'matched_budget_summary.csv', index=False)
    (out/'matched_budget_summary.json').write_text(json.dumps(notes, indent=2)+'\n')
    hashes = dict(stage=notes['stage'], inputs=[dict(path=str(p.relative_to(root)),sha256=sha256(p),bytes=p.stat().st_size) for p in inputs])
    hashes['outputs'] = [dict(path=p.name, sha256=sha256(p),bytes=p.stat().st_size) for p in [out/'matched_budget_condition_results.csv',out/'matched_budget_summary.csv',out/'matched_budget_summary.json']]
    (out/'provenance.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'Validated {len(condition)} condition-grid-budget rows from {len(inputs)-3} run CSVs; {validation_rows} saved budget evaluations; {len(weight_files)} saved weight files.')


if __name__ == '__main__':
    main()
