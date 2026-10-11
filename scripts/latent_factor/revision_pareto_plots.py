"""Revision-stage two-objective validation plots and matched-search table.

Reads only frozen validation analysis outputs; never trains, scores test data,
or alters original selectors. Uses the thesis publication plot theme.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from latent_factor_study.plot_theme import THESIS_PALETTE, apply_thesis_plot_style, style_axis

DISPLAY = {'movielens_1m':'MovieLens 1M', 'ciaodvd':'CiaoDVD', 'yelp_tampa_restaurants':'Yelp–Tampa', 'epinions':'Epinions'}
RETENTIONS = ['r100','r075','r050','r025']
RETENTION_LABEL = {'r100':'100%','r075':'75%','r050':'50%','r025':'25%'}
ROLE_LABEL = {'rmse':'RMSE','ndcg':'NDCG','mean':'Mean','ua':'UA'}
STYLE = {
    'ua': dict(marker='D', s=200, facecolors='none', edgecolors=THESIS_PALETTE['red'], linewidths=1.5, zorder=5),
    'mean': dict(marker='s', s=115, facecolors='none', edgecolors=THESIS_PALETTE['orange'], linewidths=1.5, zorder=6),
    'ndcg': dict(marker='*', s=120, facecolors=THESIS_PALETTE['green'], edgecolors='white', linewidths=.35, zorder=7),
    'rmse': dict(marker='o', s=25, facecolors=THESIS_PALETTE['blue'], edgecolors='white', linewidths=.35, zorder=8),
}


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frontier_mask(frame):
    x = frame.mean_validation_rmse.to_numpy()
    y = frame['mean_validation_ndcg@10'].to_numpy()
    # Row i is dominated if some j is no worse on both objectives and strictly
    # better on one. Exact equality is preserved; no hidden epsilon or rounding.
    dominated = ((x[None,:] <= x[:,None]) & (y[None,:] >= y[:,None]) &
                 ((x[None,:] < x[:,None]) | (y[None,:] > y[:,None]))).any(axis=1)
    return ~dominated


def legend_handles():
    return [
        Line2D([],[], marker='o', linestyle='none', color='#B8BDC7', markersize=4, label='Candidate configuration'),
        Line2D([],[], color=THESIS_PALETTE['ink'], marker='o', markerfacecolor='white', markersize=4, linewidth=1, label='Empirical non-dominated set'),
        Line2D([],[], marker='o', linestyle='none', color=THESIS_PALETTE['blue'], markersize=5, label='RMSE selector'),
        Line2D([],[], marker='*', linestyle='none', color=THESIS_PALETTE['green'], markersize=10, label='NDCG selector'),
        Line2D([],[], marker='s', linestyle='none', color=THESIS_PALETTE['orange'], markerfacecolor='none', markersize=8, label='Mean-regret constraint'),
        Line2D([],[], marker='D', linestyle='none', color=THESIS_PALETTE['red'], markerfacecolor='none', markersize=9, label='Uncertainty-aware (UA)'),
    ]


def draw_panel(ax, landscape, selected, dataset, retention, panel_label):
    frame = landscape[(landscape.dataset==dataset)&(landscape.retention==retention)].copy()
    choices = selected[(selected.dataset==dataset)&(selected.retention==retention)]
    assert len(frame)==35 and not frame.configuration_id.duplicated().any()
    assert set(choices.selector)==set(ROLE_LABEL) and len(choices)==4
    assert np.isfinite(frame[['mean_validation_rmse','mean_validation_ndcg@10']].to_numpy()).all()
    mask = frontier_mask(frame)
    assert np.array_equal(mask, frame.empirical_pareto.to_numpy(dtype=bool)), (dataset, retention, 'Pareto mismatch')
    front = frame[mask].sort_values('mean_validation_rmse')
    ax.scatter(frame.mean_validation_rmse, frame['mean_validation_ndcg@10'], color='#B8BDC7', s=17, alpha=.8, edgecolors='white', linewidths=.3, zorder=2)
    ax.plot(front.mean_validation_rmse, front['mean_validation_ndcg@10'], color=THESIS_PALETTE['ink'], linewidth=1, alpha=.8, zorder=3)
    ax.scatter(front.mean_validation_rmse, front['mean_validation_ndcg@10'], s=25, edgecolors=THESIS_PALETTE['ink'], facecolors='white', linewidths=.7, zorder=4)
    manifest = []
    groups = {}
    for role in ['ua','mean','ndcg','rmse']:
        choice = choices[choices.selector.eq(role)].iloc[0]
        lookup = frame[frame.configuration_id.eq(choice.configuration_id)]
        assert len(lookup)==1
        lookup = lookup.iloc[0]
        for metric in ['mean_validation_rmse','mean_validation_ndcg@10']:
            assert float(choice[metric]) == float(lookup[metric]), (dataset,retention,role,metric)
        ax.scatter([choice.mean_validation_rmse],[choice['mean_validation_ndcg@10']],**STYLE[role])
        groups.setdefault(choice.configuration_id,[]).append(role)
        manifest.append(dict(dataset=dataset,retention=retention,selector=role,margin=.005,
                             configuration_id=choice.configuration_id,
                             mean_validation_rmse=float(choice.mean_validation_rmse),
                             mean_validation_ndcg_at_10=float(choice['mean_validation_ndcg@10']),
                             empirical_pareto=bool(lookup.empirical_pareto),
                             candidate_count=len(frame),pareto_count=int(mask.sum())))
    coincident = []
    for roles in groups.values():
        if len(roles)>1:
            coincident.append(' = '.join(ROLE_LABEL[r] for r in ROLE_LABEL if r in roles))
    note = 'Coincident: '+('; '.join(coincident) if coincident else 'none')
    if any(not m['empirical_pareto'] for m in manifest if m['selector']=='ua'):
        note += '\nUA is outside the two-objective frontier.'
    ax.set_title(f'{panel_label}  {DISPLAY[dataset]} | {RETENTION_LABEL[retention]} training retention', loc='left', pad=28)
    ax.text(0,1.02,note,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.0,color=THESIS_PALETTE['gray'])
    ax.set_xlabel('Validation RMSE (lower is better)')
    ax.set_ylabel('Validation NDCG@10 (higher is better)')
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.margins(x=.075,y=.10)
    style_axis(ax,grid_axis='both')
    return manifest


def save_figure(pathbase, landscape, selected, conditions, heading):
    figure,axes=plt.subplots(2,2,figsize=(10.8,8.35))
    # Reserve a separate band for the two-row legend: title padding is measured
    # in points, so an axes top at .82 crowded the first-row panel headings.
    figure.subplots_adjust(left=.10,right=.97,bottom=.14,top=.78,hspace=.66,wspace=.34)
    figure.suptitle(heading,fontsize=13,fontweight='bold',x=.5,y=.982)
    figure.text(.5,.940,'Validation-only revision diagnostic | mean and UA selectors use the original absolute margin 0.005',ha='center',fontsize=9)
    legend=figure.legend(handles=legend_handles(),loc='upper center',bbox_to_anchor=(.5,.928),ncol=3,columnspacing=2.0,handletextpad=.55,fontsize=8.5)
    records=[]
    for (dataset,retention),ax,label in zip(conditions,axes.ravel(),['(a)','(b)','(c)','(d)']):
        records.extend(draw_panel(ax,landscape,selected,dataset,retention,label))
    figure.text(.10,.044,'Scale caution: panels have different axis ranges; compare labeled values, not visual distances across panels.',fontsize=8.5,fontweight='bold')
    figure.text(.10,.023,'Coincident selectors share the exact coordinates; nested marker outlines preserve their separate identities.',fontsize=8.2,color=THESIS_PALETTE['gray'])
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    legend_bottom=legend.get_window_extent(renderer).y0
    for ax in axes[0]:
        # Titles use loc='left'; require a visible gap rather than merely no
        # geometric intersection in the raster/vector exports.
        title_top=ax._left_title.get_window_extent(renderer).y1
        assert legend_bottom-title_top >= 8, (heading, legend_bottom-title_top)
    outputs=[]
    for suffix in ['.pdf','.png']:
        path=pathbase.with_suffix(suffix)
        figure.savefig(path,dpi=180,facecolor='white')
        outputs.append(path)
    plt.close(figure)
    return records,outputs


def write_search_table(summary_path, output_path):
    summary=pd.read_csv(summary_path)
    assert len(summary)==6
    lines=[r'\begin{table}[H]',r'\centering',
           r'\caption{Revision-stage matched-budget validation comparison using the same saved post-confirmation trajectories. The 35-point subgrid restricts the expanded trajectories to the original configuration coordinates; it is not the historical maximum-20-epoch experiment, whose tolerance rule sometimes stopped earlier. Each row independently reselects the RMSE and NDCG configurations at that exact budget using five seeds. Boundary counts use the extreme dimensions or regularization values of each displayed grid; a boundary flag does not establish a missing global optimum. No test outcome enters this diagnostic.}',
           r'\label{tab:matched-search}',r'\small',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
           r'Search coordinates & Epochs & Disagreement / 12 & Boundary choices / 24 \\',r'\midrule']
    for epoch in [20,50,100]:
        for scope,label in [('original35_subgrid','Original 35-point subgrid'),('expanded81_grid','Expanded 81-point grid')]:
            row=summary[(summary.search_scope==scope)&summary.epoch.eq(epoch)].iloc[0]
            lines.append(f'{label} & {epoch} & {int(row.disagreement_conditions)} & {int(row.boundary_choices)} '+r'\\')
        if epoch!=100: lines.append(r'\addlinespace')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    output_path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision-dir',type=Path,default=ROOT/'output/latent_factor_major_revision_2026-09-14')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'regenerated/reports/pareto_plots')
    parser.add_argument('--tables-dir',type=Path)
    parser.add_argument('--figures-dir',type=Path)
    parser.add_argument('--paper-dir',type=Path,help='Legacy explicit destination; cannot be combined with --tables-dir or --figures-dir.')
    args=parser.parse_args()
    if args.paper_dir and (args.tables_dir or args.figures_dir):
        parser.error('--paper-dir cannot be combined with --tables-dir or --figures-dir')
    inputs=[args.revision_dir/'selector_analysis/validation_landscapes.csv',args.revision_dir/'selector_analysis/validation_selections.csv',args.revision_dir/'search_boundary/matched_budget_summary.csv']
    frame=pd.read_csv(inputs[0]);selection=pd.read_csv(inputs[1]);selection=selection[selection.margin.eq(.005)]
    assert len(frame)==560 and len(selection)==64
    figdir=(args.paper_dir/'figures/generated/review_r1' if args.paper_dir else
            args.figures_dir if args.figures_dir else ROOT/'regenerated/figures/review_r1')
    tabledir=(args.paper_dir/'generated/review_r1' if args.paper_dir else
              args.tables_dir if args.tables_dir else ROOT/'regenerated/tables/review_r1')
    figdir.mkdir(parents=True,exist_ok=True)
    tabledir.mkdir(parents=True,exist_ok=True)
    auditdir=args.output_dir;auditdir.mkdir(parents=True,exist_ok=True)
    apply_thesis_plot_style(base_font_size=9.2)
    plotted,outputs=save_figure(figdir/'validation_pareto_full_retention',frame,selection,[(d,'r100') for d in DISPLAY], 'Prediction–ranking choices across all four full-retention validation landscapes')
    for dataset in DISPLAY:
        records,paths=save_figure(figdir/f'validation_pareto_{dataset}',frame,selection,[(dataset,r) for r in RETENTIONS],f'{DISPLAY[dataset]}: all four training-retention validation landscapes')
        plotted.extend(records);outputs.extend(paths)
    table=tabledir/'search_boundary.tex';write_search_table(inputs[2],table);outputs.append(table)
    manifest=pd.DataFrame(plotted).drop_duplicates(['dataset','retention','selector'])
    assert len(manifest)==64 and len(manifest[['dataset','retention']].drop_duplicates())==16
    manifest.to_csv(auditdir/'plotted_selector_membership.csv',index=False)
    outputs.append(auditdir/'plotted_selector_membership.csv')
    provenance=dict(stage='revision_stage_validation_only',margin=.005,
                    main_display='All four datasets at full training retention, not selected by observed results.',
                    supplementary_display='Four per-dataset 2x2 figures cover all 16 dataset-retention conditions.',
                    frontier='Exact two-objective nondomination: minimize mean global validation RMSE; maximize mean-user validation NDCG@10.',
                    data_checks='All 560 candidate coordinates finite; 35 per condition; exact equality of 64 selected metric pairs with landscape lookup; Pareto independently recomputed and compared; no coordinate jitter.',
                    ua_off_frontier_conditions=manifest[manifest.selector.eq('ua')&~manifest.empirical_pareto][['dataset','retention','configuration_id']].to_dict('records'),
                    inputs=[dict(path=str(p.relative_to(ROOT)),sha256=hash_file(p)) for p in inputs+[Path(__file__).resolve(),ROOT/'src/latent_factor_study/plot_theme.py']],
                    outputs=[dict(path=str(p.relative_to(ROOT)),sha256=hash_file(p)) for p in outputs])
    (auditdir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Created 5 figure pairs and matched-search table; verified {len(manifest)} selector memberships.')
    print('UA off frontier:',provenance['ua_off_frontier_conditions'])


if __name__=='__main__':
    main()
