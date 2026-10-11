"""Generate revision report tables from saved selector outputs only.

Re-run after missing selected-model evaluations complete. Unavailable outputs
are displayed as dashes, never inferred from validation values. Historical
primary inference remains in its original table; this report does not replace it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATASETS=['movielens_1m','ciaodvd','yelp_tampa_restaurants','epinions']
DISPLAY={'movielens_1m':'MovieLens 1M','ciaodvd':'CiaoDVD','yelp_tampa_restaurants':'Yelp--Tampa','epinions':'Epinions'}
RETENTIONS=['r100','r075','r050','r025']
RLABEL={'r100':r'100\%','r075':r'75\%','r050':r'50\%','r025':r'25\%'}
MARGINS=[.001,.0025,.005,.010]
KEYS=['dataset','retention','margin','selector']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_value(value):
    return value is not None and np.isfinite(float(value))


def number(value, *, signed=False, places=5):
    if not finite_value(value): return r'\textemdash'
    value=float(value)
    if value==0: return '0.'+'0'*places
    if abs(value)<.5*10**(-places):
        return f'{value:+.2e}' if signed else f'{value:.2e}'
    return format(value,('+' if signed else '')+f'.{places}f')


def write_table(path,caption,label,headers,rows,columns,*,note=None):
    lines=[r'\begin{table}[H]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\small',
           r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',
           ' & '.join(headers)+r' \\',r'\midrule']
    for row in rows:
        lines.append(r'\addlinespace' if row is None else ' & '.join(map(str,row))+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}%',r'}']
    if note: lines += [r'\par\smallskip\footnotesize '+note]
    lines += [r'\end{table}','']
    path.write_text('\n'.join(lines))


def json_ready(value):
    if isinstance(value,dict):return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,list):return [json_ready(v) for v in value]
    if isinstance(value,np.generic):return json_ready(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision-dir',type=Path,default=ROOT/'output/latent_factor_major_revision_2026-09-14')
    p.add_argument('--output-dir',type=Path,default=ROOT/'regenerated/reports/review_r1')
    destinations=p.add_mutually_exclusive_group()
    destinations.add_argument('--tables-dir',type=Path)
    destinations.add_argument('--paper-dir',type=Path,help='Legacy explicit destination; writes generated/review_r1 beneath it.')
    args=p.parse_args()
    evidence=args.revision_dir/'selector_analysis'
    paths=[evidence/'validation_selections.csv',evidence/'selection_outcomes.csv',evidence/'paired_test_contrasts.csv',
           args.revision_dir/'protocol_audit/configuration_registry.csv']
    before={str(path):digest(path) for path in paths}
    selections,outcomes,contrasts,registry=[pd.read_csv(path,float_precision='round_trip') for path in paths]
    assert len(selections)==len(outcomes)==256 and len(contrasts)==384
    assert not selections.duplicated(KEYS).any() and not outcomes.duplicated(KEYS).any()
    assert not contrasts.duplicated(['dataset','retention','margin','selected','comparator']).any()
    assert len(registry)==35 and not registry.run_configuration_id.duplicated().any()
    alias=registry.set_index('run_configuration_id').id.to_dict()
    assert set(selections.configuration_id).issubset(alias)
    fields=['configuration_id','dimension','regularization','admissible_count']
    merged=selections.merge(outcomes,on=KEYS,suffixes=('_validation','_outcomes'),validate='one_to_one')
    for field in fields:assert (merged[f'{field}_validation']==merged[f'{field}_outcomes']).all(),field
    assert ((outcomes.test_available & outcomes.test_seed_count.eq(5))|(~outcomes.test_available & outcomes.test_seed_count.lt(5))).all()
    present=outcomes[outcomes.test_available]
    assert np.isfinite(present[['test_global_rmse','test_mean_user_rmse','test_ndcg']].to_numpy()).all()
    available_contrasts=contrasts[contrasts.available]
    contrast_metrics=['global_rmse_selected_minus_comparator','global_rmse_selected_minus_comparator_ci_low',
                      'global_rmse_selected_minus_comparator_ci_high','ndcg_regret_comparator_minus_selected',
                      'ndcg_regret_comparator_minus_selected_ci_low','ndcg_regret_comparator_minus_selected_ci_high']
    assert np.isfinite(available_contrasts[contrast_metrics].to_numpy()).all()
    assert set(selections.dataset)==set(DATASETS) and set(selections.retention)==set(RETENTIONS)
    assert set(selections.margin)==set(MARGINS)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    texdir=(args.paper_dir/'generated/review_r1' if args.paper_dir else
            args.tables_dir if args.tables_dir else ROOT/'regenerated/tables/review_r1')
    texdir.mkdir(parents=True,exist_ok=True)
    products=[]

    def selected(dataset,retention,margin,rule):
        result=outcomes[(outcomes.dataset==dataset)&(outcomes.retention==retention)&outcomes.margin.eq(margin)&outcomes.selector.eq(rule)]
        assert len(result)==1
        return result.iloc[0]

    def contrast(dataset,retention,margin,rule,comparator):
        result=contrasts[(contrasts.dataset==dataset)&(contrasts.retention==retention)&contrasts.margin.eq(margin)&contrasts.selected.eq(rule)&contrasts.comparator.eq(comparator)]
        assert len(result)==1
        return result.iloc[0]

    ablation_rows=[];ablation_records=[]
    for di,dataset in enumerate(DATASETS):
        if di:ablation_rows.append(None)
        for ri,retention in enumerate(RETENTIONS):
            mean=selected(dataset,retention,.005,'mean');ua=selected(dataset,retention,.005,'ua');c=contrast(dataset,retention,.005,'ua','mean')
            assert c.selected_id==ua.configuration_id and c.comparator_id==mean.configuration_id
            record=dict(dataset=dataset,retention=retention,margin=.005,mean_id=mean.configuration_id,ua_id=ua.configuration_id,
                        mean_alias=alias[mean.configuration_id],ua_alias=alias[ua.configuration_id],
                        mean_admissible=int(mean.admissible_count),ua_admissible=int(ua.admissible_count),
                        different_identity=mean.configuration_id!=ua.configuration_id,test_available=bool(c.available))
            if c.available:
                assert mean.test_available and ua.test_available
                # CSV regret is comparator minus selected. Reverse both point
                # and interval endpoints for UA-minus-mean NDCG, never for RMSE.
                record.update(ua_minus_mean_test_ndcg=-float(c.ndcg_regret_comparator_minus_selected),
                              ua_minus_mean_test_ndcg_ci_low=-float(c.ndcg_regret_comparator_minus_selected_ci_high),
                              ua_minus_mean_test_ndcg_ci_high=-float(c.ndcg_regret_comparator_minus_selected_ci_low),
                              ua_minus_mean_test_global_rmse=float(c.global_rmse_selected_minus_comparator),
                              ua_minus_mean_test_global_rmse_ci_low=float(c.global_rmse_selected_minus_comparator_ci_low),
                              ua_minus_mean_test_global_rmse_ci_high=float(c.global_rmse_selected_minus_comparator_ci_high),
                              interval_target=c.interval_target)
                assert np.isclose(record['ua_minus_mean_test_ndcg'],ua.test_ndcg-mean.test_ndcg,rtol=1e-10,atol=1e-14)
                assert np.isclose(record['ua_minus_mean_test_global_rmse'],ua.test_global_rmse-mean.test_global_rmse,rtol=1e-10,atol=1e-14)
            else:
                record.update(ua_minus_mean_test_ndcg=None,ua_minus_mean_test_global_rmse=None,
                              missing_choice_ids=[str(r.configuration_id) for r in [mean,ua] if not r.test_available])
            ablation_records.append(record)
            ablation_rows.append([DISPLAY[dataset] if ri==0 else '',RLABEL[retention],record['mean_alias'],record['ua_alias'],record['mean_admissible'],record['ua_admissible'],
                                  number(record['ua_minus_mean_test_ndcg'],signed=True),number(record['ua_minus_mean_test_global_rmse'],signed=True)])
    path=texdir/'uncertainty_ablation.tex'
    ablation_availability=(r'Dashes mark unavailable selected-model evaluations, not zero effects. ' if any(not x['test_available'] for x in ablation_records)
                           else r'All 16 paired comparisons have complete five-seed held-out evaluations. ')
    write_table(path,
        r'Revision-stage uncertainty ablation at the original absolute margin $\delta=0.005$. Mean and UA use identical RMSE objectives and tie-breaking; only UA adds the paired-seed uncertainty penalty. Configuration aliases follow Supplementary Section 5; $|A|$ is the admissible-set size. Test differences are UA minus mean-constrained selection, averaged over five per-model-seed metrics. Positive NDCG and negative global RMSE differences favor UA on their respective endpoints; these need not agree. Global RMSE is event-weighted within each seed, not mean-user RMSE. '+ablation_availability+r'Corresponding paired fixed-model user-bootstrap 95\% intervals and all identities are in the full CSV report.',
        'tab:uncertainty-ablation',['Dataset','Retention','Mean ID','UA ID',r'$|A_{\rm mean}|$',r'$|A_{\rm UA}|$',r'$\Delta$ NDCG@10',r'$\Delta$ global RMSE'],ablation_rows,'llccrrrr')
    products.append(path)
    pd.DataFrame(ablation_records).to_csv(out/'uncertainty_ablation.csv',index=False);products.append(out/'uncertainty_ablation.csv')

    margin_rows=[];margin_records=[];transitions=[]
    for margin in MARGINS:
        counts=dict(margin=margin,conditions=16,ua_differs_mean=0)
        for rule in ['ua','mean']:
            counts.update({f'{rule}_equals_rmse':0,f'{rule}_equals_ndcg':0,f'{rule}_intermediate':0,f'{rule}_all35':0})
        for dataset in DATASETS:
            for retention in RETENTIONS:
                rm=selected(dataset,retention,margin,'rmse');nd=selected(dataset,retention,margin,'ndcg')
                ua=selected(dataset,retention,margin,'ua');mn=selected(dataset,retention,margin,'mean')
                counts['ua_differs_mean']+=int(ua.configuration_id!=mn.configuration_id)
                for rule,choice in [('ua',ua),('mean',mn)]:
                    intermediate=choice.configuration_id not in {rm.configuration_id,nd.configuration_id}
                    assert bool(choice.is_intermediate)==intermediate
                    counts[f'{rule}_equals_rmse']+=int(choice.configuration_id==rm.configuration_id)
                    counts[f'{rule}_equals_ndcg']+=int(choice.configuration_id==nd.configuration_id)
                    counts[f'{rule}_intermediate']+=int(intermediate)
                    counts[f'{rule}_all35']+=int(choice.admissible_count==35)
        margin_records.append(counts)
        margin_rows.append([f'{margin:.4f}',counts['ua_differs_mean'],counts['ua_equals_rmse'],counts['ua_intermediate'],counts['ua_all35'],counts['mean_equals_rmse'],counts['mean_intermediate'],counts['mean_all35']])
    path=texdir/'margin_summary.tex'
    write_table(path,
        r'Revision-stage rerunning of admissibility and selection across declared absolute NDCG@10 margins. Every count is out of 16 dataset--retention conditions, not independent external datasets or a deployment-risk probability. Equality denotes the exact historical RMSE configuration identity; intermediate choices differ from both the RMSE and NDCG choices. \emph{All 35} means every evaluated configuration passes that rule. Mean and UA differ only in the uncertainty term and share their full within-set tie-breaking hierarchy. These are selection changes, not reclassification of fixed test effects.',
        'tab:margin-sensitivity',[r'$\delta$',r'UA $\ne$ mean','UA = RMSE','UA intermediate','UA all 35','Mean = RMSE','Mean intermediate','Mean all 35'],margin_rows,'rrrrrrrr')
    products.append(path)

    details=[];references=[]
    for dataset in DATASETS:
        rows=[]
        for ri,retention in enumerate(RETENTIONS):
            if ri:rows.append(None)
            reference=selected(dataset,retention,.005,'ndcg')
            references.append(dict(dataset=dataset,retention=retention,ranking_reference_id=reference.configuration_id,
                                   ranking_reference_alias=alias[reference.configuration_id],
                                   reference_validation_ndcg=float(reference['mean_validation_ndcg@10']),
                                   reference_test_ndcg=float(reference.test_ndcg) if reference.test_available else None))
            for mi,margin in enumerate(MARGINS):
                for ji,rule in enumerate(['mean','ua']):
                    row=selected(dataset,retention,margin,rule);ref=selected(dataset,retention,margin,'ndcg');c=contrast(dataset,retention,margin,rule,'ndcg')
                    assert row.ranking_reference_id==ref.configuration_id
                    record=dict(dataset=dataset,retention=retention,margin=margin,selector=rule,
                                configuration_id=row.configuration_id,configuration_alias=alias[row.configuration_id],
                                admissible_count=int(row.admissible_count),mean_validation_rmse=float(row.mean_validation_rmse),
                                mean_validation_ndcg_at_10=float(row['mean_validation_ndcg@10']),test_available=bool(row.test_available),
                                test_seed_count=int(row.test_seed_count),test_global_rmse=float(row.test_global_rmse) if row.test_available else None,
                                test_ndcg=float(row.test_ndcg) if row.test_available else None,
                                ranking_reference_id=ref.configuration_id,reference_alias=alias[ref.configuration_id],
                                reference_test_ndcg=float(ref.test_ndcg) if ref.test_available else None,
                                signed_test_regret_reference_minus_selected=float(c.ndcg_regret_comparator_minus_selected) if c.available else None,
                                relative_test_regret_percent=float(c.relative_ndcg_regret_percent) if c.available and finite_value(c.relative_ndcg_regret_percent) else None,
                                relative_denominator_near_zero=bool(c.relative_denominator_near_zero) if c.available else None)
                    details.append(record)
                    rows.append([RLABEL[retention] if mi==ji==0 else '',f'{margin:.4f}' if ji==0 else '',rule.upper() if rule=='ua' else 'Mean',alias[row.configuration_id],int(row.admissible_count),
                                 number(row.mean_validation_rmse),number(row['mean_validation_ndcg@10']),number(record['test_global_rmse']),number(record['test_ndcg'])])
        path=texdir/f'margin_details_{dataset}.tex'
        refnote='; '.join(f"{RLABEL[x['retention']]}: {x['ranking_reference_alias']}" for x in references if x['dataset']==dataset)
        availability=(r'Dashes identify missing held-out evaluations pending completion. ' if any(not x['test_available'] for x in details if x['dataset']==dataset)
                      else r'Every displayed test result has all five model seeds. ')
        write_table(path,
            f'Revision-stage margin sensitivity for {DISPLAY[dataset]}. All 32 retention--margin--rule selections are shown. Test metrics are arithmetic averages across five model-seed evaluations, not predictions averaged into an ensemble. Validation RMSE and test RMSE are global event-weighted measures within a model seed. Configuration aliases are defined in '+r'\Cref{tab:config-registry}; $|A|$ denotes admissible candidates. '+availability+'Reference NDCG scores, signed absolute and relative test regrets, and uncertainty intervals are retained in the machine-readable reports.',
            f'tab:margin-details-{dataset.replace("_","-")}',
            ['Retention',r'$\delta$','Rule','ID',r'$|A|$','Val. RMSE','Val. NDCG@10','Test RMSE','Test NDCG@10'],rows,'llccrrrrr',
            note='The fixed validation-NDCG reference IDs are '+refnote+'. Relative regret uses the corresponding reference test NDCG denominator; a near-zero denominator is flagged rather than replaced with an epsilon.')
        products.append(path)
    assert len(details)==128
    pd.DataFrame(details).to_csv(out/'margin_details.csv',index=False);products.append(out/'margin_details.csv')
    pd.DataFrame(references).to_csv(out/'ranking_references.csv',index=False);products.append(out/'ranking_references.csv')
    for dataset in DATASETS:
        for retention in RETENTIONS:
            for rule in ['mean','ua']:
                part=[x for x in details if x['dataset']==dataset and x['retention']==retention and x['selector']==rule]
                assert len(part)==4
                assert all(a['admissible_count']<=b['admissible_count'] for a,b in zip(part,part[1:]))
                assert all(a['mean_validation_rmse']>=b['mean_validation_rmse'] for a,b in zip(part,part[1:]))
                rm_id=selected(dataset,retention,.005,'rmse').configuration_id
                transitions.append(dict(dataset=dataset,retention=retention,selector=rule,
                    selected_ids=[x['configuration_id'] for x in part],admissible_counts=[x['admissible_count'] for x in part],
                    first_grid_margin_equal_rmse=next((x['margin'] for x in part if x['configuration_id']==rm_id),None),
                    first_grid_margin_all35=next((x['margin'] for x in part if x['admissible_count']==35),None),
                    distinct_choices_on_grid=len({x['configuration_id'] for x in part})))

    joint_rows=[];joint_records=[]
    for di,dataset in enumerate(DATASETS[:3]):
        if di:joint_rows.append(None)
        for ri,retention in enumerate(RETENTIONS):
            rm=selected(dataset,retention,.005,'rmse');nd=selected(dataset,retention,.005,'ndcg');c=contrast(dataset,retention,.005,'rmse','ndcg')
            assert rm.test_available and nd.test_available and c.available
            record=dict(dataset=dataset,retention=retention,ndcg_reference_id=nd.configuration_id,rmse_selected_id=rm.configuration_id,
                        test_ndcg_reference=float(nd.test_ndcg),test_ndcg_rmse_selected=float(rm.test_ndcg),
                        signed_test_ndcg_reference_minus_rmse=float(c.ndcg_regret_comparator_minus_selected),
                        relative_regret_percent_reference_denominator=float(c.relative_ndcg_regret_percent) if finite_value(c.relative_ndcg_regret_percent) else None,
                        relative_denominator_near_zero=bool(c.relative_denominator_near_zero),
                        global_rmse_ndcg_minus_rmse=-float(c.global_rmse_selected_minus_comparator))
            joint_records.append(record)
            joint_rows.append([DISPLAY[dataset] if ri==0 else '',RLABEL[retention],number(record['test_ndcg_reference']),number(record['test_ndcg_rmse_selected']),
                               number(record['signed_test_ndcg_reference_minus_rmse'],signed=True),number(record['relative_regret_percent_reference_denominator'],signed=True,places=1),
                               number(record['global_rmse_ndcg_minus_rmse'],signed=True)])
    path=texdir/'joint_primary_tradeoff.tex'
    write_table(path,
        r'Joint prediction--ranking consequences of the original locked RMSE and NDCG selections on the 12 construction conditions, consolidated from saved historical outcomes. $N$ denotes the validation-NDCG reference and $R$ the validation-RMSE choice. NDCG regret is $N-R$; its relative value is $100(N-R)/N$ using reference test NDCG, not a business-utility scale. Prediction cost is global test RMSE of $N$ minus that of $R$. All values are averages of five per-seed metrics. Original paired ranking intervals and Holm-adjusted inference remain in \Cref{tab:mf-selection-regret}; this descriptive consolidation does not replace that inference or attach mean-user RMSE intervals to a global RMSE difference.',
        'tab:joint-primary-tradeoff',['Dataset','Retention',r'NDCG$_N$',r'NDCG$_R$',r'$\Delta$ NDCG',r'Loss / $N$ (\%)',r'$\Delta$ global RMSE'],joint_rows,'llrrrrr')
    products.append(path)
    pd.DataFrame(joint_records).to_csv(out/'joint_primary_tradeoff.csv',index=False);products.append(out/'joint_primary_tradeoff.csv')

    available=[x for x in ablation_records if x['test_available']]
    changed=[x for x in available if x['different_identity']]
    def sign_counts(records,field):
        return dict(positive=sum(x[field]>0 for x in records),negative=sum(x[field]<0 for x in records),zero=sum(x[field]==0 for x in records),available=len(records))
    summary=dict(stage='revision_stage_reporting_of_saved_outputs',not_submission_ready_while_evaluations_missing=True,
                 held_out_table_status='incomplete' if any(not x['test_available'] for x in details) else 'complete',
                 all_rule_rows=128,available_rule_test_rows=sum(x['test_available'] for x in details),
                 missing_rule_test_rows=sum(not x['test_available'] for x in details),
                 ablation_at_original_margin=dict(conditions=16,different_choices=sum(x['different_identity'] for x in ablation_records),
                    paired_test_available=len(available),paired_test_missing=16-len(available),
                    changed_choices_with_test=len(changed),
                    ua_minus_mean_ndcg_signs_all_available=sign_counts(available,'ua_minus_mean_test_ndcg'),
                    ua_minus_mean_ndcg_signs_changed=sign_counts(changed,'ua_minus_mean_test_ndcg'),
                    ua_minus_mean_global_rmse_signs_changed=sign_counts(changed,'ua_minus_mean_test_global_rmse'),
                    ndcg_CI_wholly_positive=sum(x['ua_minus_mean_test_ndcg_ci_low']>0 for x in changed),
                    ndcg_CI_wholly_negative=sum(x['ua_minus_mean_test_ndcg_ci_high']<0 for x in changed),
                    ndcg_CI_contains_zero=sum(x['ua_minus_mean_test_ndcg_ci_low']<=0<=x['ua_minus_mean_test_ndcg_ci_high'] for x in changed),
                    conditions_detail=ablation_records),
                 margin_selection_summary=margin_records,margin_transitions=transitions,
                 relative_effects_denominator='Reference configuration test NDCG; no epsilon; near-zero flags copied from audited contrasts.',
                 original_joint_point_summaries=joint_records,
                 inference_boundary='Revision contrasts use fixed-fit paired user bootstrap. Original primary ranking intervals/Holm tests remain original and are not replaced by this descriptive table. Global and mean-user RMSE are distinct estimands; primary compact tables report global point metrics only.')
    after={str(path):digest(path) for path in paths}
    assert before==after,'Input report files changed during generation; rerun from a stable completed snapshot.'
    summary_path=out/'report_summary.json';summary_path.write_text(json.dumps(json_ready(summary),indent=2,allow_nan=False)+'\n');products.append(summary_path)
    provenance=dict(inputs=[dict(path=str(path.relative_to(ROOT)),sha256=before[str(path)]) for path in paths],
                    script=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=digest(Path(__file__))),
                    outputs=[dict(path=str(path.relative_to(ROOT)),sha256=digest(path)) for path in products],
                    checks=['All16conditions×4margins×4selector identities preserved; exact aliases checked.',
                            'Outcome keys/identities/admissibility match frozen validation selections.',
                            'Unavailable output displayed as dash; available test rows require5seeds.',
                            'UA-minus-mean NDCG interval sign reversal swaps endpoints.',
                            'Subset sizes nested and selected minimum validation RMSE nonincreasing across margins.',
                            'Input hashes unchanged across report generation.'])
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(tables=7,ablation_conditions=16,margin_detail_rows=len(details),missing_rule_test_rows=summary['missing_rule_test_rows'],
                          ablation_changed_signs=summary['ablation_at_original_margin']['ua_minus_mean_ndcg_signs_changed']),indent=2))


if __name__=='__main__':main()
