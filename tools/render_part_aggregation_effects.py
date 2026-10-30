"""Render recorded full-Test paired effects; never choose models or recompute tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'exp/261029/part_aware_dynamic_aggregation'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--versions', nargs='+')
    mode.add_argument('--control-family', choices=['v4', 'v5', 'v6', 'v6_long'])
    parser.add_argument('--output-stem', help='Separate figure filename stem to preserve earlier completed comparisons')
    args = parser.parse_args()
    records = []; sources = {}; snapshots = {}
    def preserve_source(path,source):
        digest=hashlib.sha256(source).hexdigest()
        archive=OUT/'figures/data_sources'/(digest+'.json');archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():assert archive.read_bytes()==source
        else:
            temporary=archive.with_name(archive.name+'.'+str(os.getpid())+'.tmp')
            temporary.write_bytes(source);temporary.replace(archive)
        key=str(path.relative_to(ROOT));sources[key]=digest;snapshots[key]=str(archive.relative_to(ROOT))
    for version in args.versions or []:
        assert version in ['v2', 'v3', 'v4', 'v5', 'v6', 'v6_long']
        path = OUT/(version+'_comparison.json')
        source = path.read_bytes(); result = json.loads(source)
        assert result['all_complete'] and not result['pending']
        name = version+'_part_dynamic'
        assert name in result['completed']
        preserve_source(path,source)
        for suffix in ['frame_pa', 'RPE_Z']:
            record = result['tests'][name+'_'+suffix]
            assert record['status'] == 'tested' and record['clusters'] == 18
            assert np.isfinite([record['effect'], *record['ci95']]).all()
            records.append(dict(version=version, metric=suffix, **record))
    if args.control_family:
        path = OUT/'artifact_audits'/(args.control_family+'_part_vs_controls.json')
        source = path.read_bytes(); result = json.loads(source)
        assert result['status'] == 'complete' and result['segments'] == 80
        preserve_source(path,source)
        labels = dict(fixed='Fixed', global_dynamic='Global dynamic', part_static='Part static')
        for control in result['controls']:
            label = labels[control.removeprefix(args.control_family+'_')]
            for suffix in ['frame_pa', 'RPE_Z']:
                record = result['tests'][control+'_'+suffix]
                assert record['status'] == 'tested' and record['clusters'] == 18
                records.append(dict(version=label, metric=suffix, **record,
                                    holm_p_value=record['holm_p_value_within_three_controls']))
    height=max(4.6,.65*(len(records)//2)+1.5)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, height), layout='constrained')
    for ax, suffix, title in zip(axes, ['frame_pa', 'RPE_Z'],
                                ['Frame-aligned PA-MPJPE', 'Root absolute depth error']):
        rows = [r for r in records if r['metric'] == suffix]
        for i, row in enumerate(rows):
            low, high = row['ci95']; value = row['effect']
            ax.plot([low, high], [i, i], color='#28778e', linewidth=2)
            ax.scatter([value], [i], color='#174c66', s=40, zorder=3)
            ax.text(.99, i+.2, f"n={row['samples']:,} | Holm p={row['holm_p_value']:.3f}",
                    transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8, color='#52606d')
        ax.axvline(0, color='#718096', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(rows)), [r['version'] if args.control_family else ('V6 long' if r['version']=='v6_long' else r['version'].upper()) for r in rows])
        ax.set_ylim(len(rows)-.5, -.5)
        ax.set_title(title, fontsize=12)
        reference='control' if args.control_family else 'RGB reference'
        ax.set_xlabel('Part dynamic minus '+reference+' (mm)\nNegative: lower error', fontsize=10)
        ax.grid(axis='x', alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.spines[['top', 'right']].set_visible(False)
    title=args.control_family.upper().replace('_',' ')+' part dynamic vs. temporal controls' if args.control_family else 'Part-aware aggregation: completed full-Test effects'
    fig.suptitle(title, fontsize=14)
    fig.supxlabel('80 clips | 18 subject clusters | Paired 95% cluster bootstrap intervals\n'
                  'Each pair has its own common detections. Frozen models; no Test-based selection.', fontsize=9)
    folder = OUT/'figures'; folder.mkdir(exist_ok=True)
    stem=args.output_stem or (args.control_family+'_part_control_effects' if args.control_family else 'completed_paired_effects')
    assert stem and Path(stem).name==stem and stem not in ['.', '..']
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(folder/(stem+'.'+extension), dpi=180, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    manifest = dict(versions=args.versions,control_family=args.control_family, source_sha256=sources, source_snapshots=snapshots, actual_recorded_full_Test_statistics=records,
                    selection_performed=False, inference_performed=False, statistics_reestimated=False,
                    reference='Temporal controls, pairwise common detections' if args.control_family else 'Strong Train-selected v2_rgb_only, pairwise common detections')
    (folder/(stem+'.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(folder/(stem+'.png'))


if __name__ == '__main__':
    main()
