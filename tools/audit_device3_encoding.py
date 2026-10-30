"""Independent protocol/cache/report gates for the encoding experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from device3_encoding_scope import GOAL,EVAL_NAMES,TRAIN_NAMES,LABELS,METRICS,PAIRS,ENCODINGS
from goal_runtime_control import write,register_job,identity
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def protocol(g):
    out=Path(g['output_root']);cfgs={n:read(out/'configs'/(n+'.json')) for n in EVAL_NAMES}
    assert g['deadline_unix']-g['started_unix']==12*3600
    frozen=read(out/'original_pretrained_manifest.json');split=(ROOT/'dataset/split.json').read_bytes()
    assert sha(frozen['path'])==frozen['sha256']
    fixed=['initialization_checkpoint','initialization_sha256','seed','batch_size','augment',
        'last_blocks','head_lr','backbone_lr','adapter_lr','fusion_lr','pa_weight','root_weight',
        'validation_minutes','backbone_warmup_seconds','warmup_steps','minimum_coverage','residual_regularization']
    reference=cfgs['rgb3'];records=[]
    for n,c in cfgs.items():
        assert c['split_sha256']==hashlib.sha256(split).hexdigest()
        assert c['output_root']==str(out) and c['deadline_unix']==g['deadline_unix']
        assert c['allowed_gpus']==g['allowed_gpus']==[1,2,4,5]
        assert c['initialization_kind']=='released_original'
        assert c['initialization_checkpoint']==frozen['path'] and c['initialization_sha256']==frozen['sha256']
        assert all(c[k]==reference[k] for k in fixed)
        assert c['hours']==(0 if n=='pre_rgb3' else 1.5 if n=='rgb3' else 3)
        assert c['encoding'] in ENCODINGS
        if n=='pre_rgb3':assert c['weight_stage']=='released_pretrained'
        elif n!='rgb3':assert n==c['condition']+'_'+c['encoding']
        if c['condition']=='event3_event4':assert c['primary_sensor']=='event3' and c['auxiliary_sensor']=='event4'
        records.append(dict(name=n,hours=c['hours'],encoding=c['encoding'],same_original_initialization=True))
    plan=read(out/'statistical_analysis_plan.json');assert [tuple(p) for p in plan['pairs']]==PAIRS
    assert plan['conditions']==EVAL_NAMES and not plan['Test_metrics_examined_before_plan']
    write(out/'protocol_audit.json',dict(status='passed',records=records,statistical_pairs=len(PAIRS),
        common_hyperparameters=fixed,seed=reference['seed'],original_checkpoint_sha256=frozen['sha256'],checked_unix=time.time()))

def cache_session(name):
    import h5py
    import numpy as np
    from utils.event_encoding import EventFrames,ENCODINGS as SPECS
    g=read(GOAL);out=Path(g['output_root']);receipt=read(out/'encoding_receipts'/(name+'.json'))
    files={e:h5py.File(out/'encoding_cache'/e/(name+'.h5'),'r',locking=False) for e in ENCODINGS};checks=[]
    try:
        base=files['count33'];frames=base['frames'][:];clock=base['timestamp_ns'][:]
        for e,f in files.items():
            assert f.attrs['complete'] and f.attrs['event_encoding']==e
            assert f.attrs['split_sha256']==read(out/'manifest.json')['split_sha256']
            assert np.array_equal(f['frames'][:],frames) and np.array_equal(f['timestamp_ns'][:],clock)
            for sensor in ('event3','event4'):
                assert f[sensor].shape==(len(frames),SPECS[e]['channels'],224,224)
                assert np.array_equal(f[sensor+'_available'][:],base[sensor+'_available'][:])
        selected=np.unique(np.linspace(0,len(frames)-1,min(3,len(frames)),dtype=int))
        for sensor in ('event3','event4'):
            path=ROOT/'dataset'/name/('device'+sensor[-1])/'event.npz'
            source=next(v for v in receipt['source_archives'] if v['sensor']==sensor)
            assert sha(path)==source['sha256']
            event=EventFrames.load(path,size=224)
            for i in selected:
                encoded,meta=event.encode(int(clock[i]),encodings=ENCODINGS)
                assert not meta['event_available'] or meta['latest_event_timestamp_ns']<=int(clock[i])
                for e in ENCODINGS:
                    expected=np.rint(encoded[e]*255).astype(np.uint8)
                    assert np.array_equal(files[e][sensor][i],expected),(name,sensor,e,int(i))
                checks.append(dict(sensor=sensor,cache_row=int(i),source_frame=int(frames[i]),timestamp_ns=int(clock[i]),all_three_encodings_exact=True))
            del event
        digests={e:sha(out/'encoding_cache'/e/(name+'.h5')) for e in ('bins100','surface20')}
        assert all(digests[e]==receipt['encodings'][e]['sha256'] for e in digests)
        return dict(session=name,actual_archive_reencoding=checks,cache_sha256=digests)
    finally:
        for f in files.values():f.close()

def cache(g):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    out=Path(g['output_root'])
    while not (out/'encoding_cache_complete.json').exists():
        if time.time()>g['deadline_unix']-900:return
        time.sleep(30)
    manifest=read(out/'manifest.json');names=sum([manifest[r+'_sessions'] for r in ('training','validation','test')],[])
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(cache_session,n) for n in names]):rows.append(f.result())
    assert len(rows)==78
    write(out/'actual_encoding_cache_audit.json',dict(status='passed',sessions=len(rows),
        scope='All78 sessions both sensors; source archive SHA, actual three timestamps per sensor independently re-encoded for all three methods; full new H5 hashes; exact causal frame map and channel count',
        records=rows,checked_unix=time.time(),Test_used_for_selection=False))

def report(g):
    import numpy as np
    out=Path(g['output_root']);c=read(out/'comparison.json');runtime=read(out/'runtime/summary.json');text=Path(g['report']).read_text();lines=text.splitlines()
    assert lines[0].startswith('# ') and lines[2].startswith('| 조건 / 인코딩 |')
    def table(prefix):
        idx=next(i for i,s in enumerate(lines) if s.startswith(prefix))+2;rows=[]
        while idx<len(lines) and lines[idx].startswith('|'):
            rows.append([v.strip().replace('**','') for v in lines[idx].strip('|').split('|')]);idx+=1
        return rows
    def val(cell,v,places=2):
        if v is None:assert cell.rstrip('%')=='미평가';return
        x=float(cell.replace(',','').rstrip('%'));assert np.isfinite(x) and abs(x-v)<=.5*10**(-places)+1e-8,(cell,v)
    rows=table('| 조건 / 인코딩 |');assert len(rows)==11
    for cells,n in zip(rows,EVAL_NAMES):
        assert cells[0]==LABELS[n] and len(cells)==6
        for cell,k in zip(cells[1:],METRICS):val(cell,c['metrics'][n][k])
    rows=table('| 조건 | RPE-Z');assert len(rows)==11
    for cells,n in zip(rows,EVAL_NAMES):
        assert cells[0]==LABELS[n]
        for cell,k in zip(cells[1:],('root_depth_error_mm','absolute_mpjpe_mm','pelvis_velocity_error_mmps','root_acceleration_error_mmps2')):val(cell,c['metrics'][n][k])
    rows=table('| 조건 | 전체 GT 검출수');assert len(rows)==11
    for cells,n in zip(rows,EVAL_NAMES):
        v=c['coverage'][n];assert cells[0]==LABELS[n] and cells[1]==f'{v["matched"]:,} / {v["total"]:,}';val(cells[2],100*v['fraction'])
    rows=table('| 조건 | 캐시 fps');assert len(rows)==11
    for cells,n in zip(rows,EVAL_NAMES):
        a=runtime[n+'_cached'];b=runtime[n+'_online'];assert cells[0]==LABELS[n]
        expected=[a['fps'],b['fps'],a['loading_seconds'],max(r['peak_allocated_bytes'] for r in a['segments'])/1024**3,b['host_peak_rss_bytes']/1024**3]
        for cell,v in zip(cells[1:],expected):val(cell,v)
    for title,suffix in [('| 비교 | frame-aligned PA 변화','_pa'),('| 비교 | 3D Root 변화','_root_position')]:
        rows=table(title);assert len(rows)==len(PAIRS)
        for cells,(a,b) in zip(rows,PAIRS):
            v=c['tests'][a+'__'+b+suffix];assert cells[0]==a+' → '+b
            val(cells[1],v['effect'],4);val(cells[3],v['holm_p_value'],4);val(cells[4],v.get('relative_percent'),4)
            if v['ci95'] is None:assert cells[2]=='미평가'
            else:
                for cell,x in zip(cells[2].strip('[]').split(','),v['ci95']):val(cell,x,4)
    rows=table('| 조건 | 정지(mm)');assert len(rows)==11
    for cells,n in zip(rows,EVAL_NAMES):
        assert cells[0]==LABELS[n]
        for cell,label in zip(cells[1:],('still','slow','fast')):
            v=c['motion_strata'][n][label];val(cell,v['sum_mm']/v['count'] if v['count'] else None)
    rows=table('| 조건 | 자체 frame0');assert len(rows)==11
    for cells,n in zip(rows,EVAL_NAMES):
        v=c['own_metrics'][n];assert cells[0]==LABELS[n] and int(cells[1].replace(',',''))==v['samples']
        val(cells[2],v['frame_aligned_pa_mpjpe_mm']);val(cells[3],v['pelvis_error_mm'])
    write(out/'all_report_tables_audit.json',dict(status='passed',main_rows=11,primary_numeric_columns=5,
        verified_tables=['primary','RPE-Z and absolute/temporal','detection coverage','cached/online speed/loading/memory','PA effects/CI/Holm','3D Root effects/CI/Holm','motion strata','own population'],
        report_sha256=sha(g['report']),checked_unix=time.time()))

def watch_report(g):
    out=Path(g['output_root'])
    while not (out/'ready_for_agent_review.json').exists():
        if time.time()>g['deadline_unix']-120:return
        time.sleep(30)
    report(g)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['protocol','cache','report','report-watch']);args=parser.parse_args()
    g=read(GOAL);register_job(g,os.getpid(),'encoding_audit:'+args.stage)
    {'protocol':protocol,'cache':cache,'report':report,'report-watch':watch_report}[args.stage](g)
if __name__=='__main__':main()
