from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT=_Path(__file__).resolve().parents[2]
_sys.path.insert(0,str(_PROJECT_ROOT/"src"))
_sys.path.insert(0,str(_PROJECT_ROOT))
"""Standalone held-out validation curves for the timed fine-tuning search."""
import csv
from datetime import datetime,timezone
import json
import os

from multihmr.data import OUT,WORK


def main():
    os.environ.setdefault('MPLCONFIGDIR',str(WORK/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((OUT/'manifest.json').read_text())
    paths=sorted(p for p in (OUT/'runs').iterdir() if p.name.startswith(('w1_','w2_')))
    rows=[];series=[];baseline=None
    for path in paths:
        if not (path/'baseline.json').exists() or not (path/'started.json').exists():continue
        initial=json.loads((path/'baseline.json').read_text())
        started=json.loads((path/'started.json').read_text())['started_unix']
        if baseline is None:baseline=initial
        records=[dict(initial,step=0,elapsed_seconds=0)]
        if (path/'validation.jsonl').exists():
            for line in (path/'validation.jsonl').read_text().splitlines():
                if line.strip():records.append(json.loads(line))
        items=[]
        for value in records:
            row=dict(run=path.name,hours=(started-manifest['goal_start_unix']+value['elapsed_seconds'])/3600,
                     step=value['step'],mpjpe_mm=value['mpjpe_mm'],root_mpjpe_mm=value['root_mpjpe_mm'],
                     pa_mpjpe_mm=value['pa_mpjpe_mm'],coverage_percent=100*value['coverage'])
            rows.append(row);items.append(row)
        series.append((path.name,items))
    assert rows and baseline is not None
    with (OUT/'validation_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    colors=['#666666','#0072B2','#009E73','#CC79A7','#E69F00','#D55E00']
    fig,(metric,coverage)=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for i,(name,items) in enumerate(series):
        x=[r['hours'] for r in items];color=colors[i%len(colors)]
        metric.plot(x,[r['pa_mpjpe_mm'] for r in items],marker='o',markersize=4,label=name,color=color)
        coverage.plot(x,[r['coverage_percent'] for r in items],marker='o',markersize=4,color=color)
    metric.axhline(baseline['pa_mpjpe_mm'],color='#999999',linestyle='--',linewidth=1,label='Pretrained reference')
    floor=100*baseline['coverage']-.5
    coverage.axhline(floor,color='#999999',linestyle=':',linewidth=1,label='Selection coverage floor')
    eligible=[r for r in rows if r['coverage_percent']>=floor]
    best=min(eligible,key=lambda r:r['pa_mpjpe_mm'])
    metric.scatter([best['hours']],[best['pa_mpjpe_mm']],marker='*',s=150,color='#D55E00',zorder=5)
    metric.text(.99,.97,f'Best eligible: {best["pa_mpjpe_mm"]:.2f} mm',ha='right',va='top',transform=metric.transAxes)
    metric.set_ylabel('Root-aligned PA-MPJPE (mm)')
    metric.set_ylim(bottom=0,top=max(r['pa_mpjpe_mm'] for r in rows)*1.12)
    coverage.set_ylabel('GT matching coverage (%)')
    coverage.set_ylim(min(95,min(r['coverage_percent'] for r in rows)-.5),100)
    start=datetime.fromtimestamp(manifest['goal_start_unix'],timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    coverage.set_xlabel(f'Hours since search start ({start})')
    metric.set_title('Held-out Train-subject validation: checkpoint selection metric')
    for ax in [metric,coverage]:ax.grid(alpha=.25);ax.legend(loc='best',fontsize=8)
    fig.tight_layout();path=OUT/'validation_progress.png';fig.savefig(path,dpi=170);plt.close(fig)
    print(str(path))


if __name__=='__main__':main()
