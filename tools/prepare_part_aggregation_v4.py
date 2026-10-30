"""Freeze a Train-only V4 follow-up; never launch jobs or inspect Test results."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/261029/part_aware_dynamic_aggregation'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    assert not (OUT/'v4_design_frozen.json').exists(), 'Frozen plan already exists'
    base=read(OUT/'configs/v3_part_dynamic.json')
    diagnosis=read(OUT/'diagnostics/v3_part_dynamic.json')['controls']
    reference=OUT/'checkpoints/v2_rgb_only/best.pt';warm=OUT/'checkpoints/v3_part_dynamic/best.pt'
    shared=dict(base,architecture='calibrated_part_frozen_rgb_v4',architecture_version='calibrated_part_frozen_rgb_v4',
        freeze_RGB_base=True,frozen_RGB_checkpoint=str(reference),frozen_RGB_checkpoint_sha256=sha(reference),
        fusion_initialization_checkpoint=str(warm),fusion_initialization_sha256=sha(warm),
        uniform_aggregation_warmup_steps=100,comparison_names=['v2_rgb_only','v4_fixed','v4_global_dynamic','v4_part_dynamic','v4_part_static'],
        reference_name='v2_rgb_only')
    sources=list(base['source_sha256'])+['src/multihmr/part_aggregation_v4.py']
    shared['source_sha256']={p:sha(ROOT/p) for p in sources}
    shared['v4_model_source_sha256']=shared['source_sha256']['src/multihmr/part_aggregation_v4.py']
    snapshots=OUT/'source_snapshots'
    for p,h in shared['source_sha256'].items():
        target=snapshots/(h+'.txt')
        if not target.exists():target.write_bytes((ROOT/p).read_bytes())
    names=[]
    for suffix,variant,static in [('fixed','fixed_all',False),('global_dynamic','part_agnostic',False),('part_dynamic','part_aware',False),('part_static','part_aware',True)]:
        name='v4_'+suffix;names.append(name)
        config=dict(shared,name=name,variant=variant,static_part_horizon=static)
        write(OUT/'configs'/(name+'.json'),config)
        pilot=dict(config,name='pilot_'+name,pilot_steps=8,pilot_train_limit=32,pilot_validation_limit=16,workers=0,
            warmup_steps=2,uniform_aggregation_warmup_steps=2,
            pilot_training_sessions=['20260720_131207_863008_200_b'],pilot_validation_sessions=['20260721_092034_900028_101_b'])
        write(OUT/'configs'/('pilot_'+name+'.json'),pilot)
    plan=dict(design_frozen_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        architecture_version=shared['architecture_version'],source_sha256=shared['source_sha256'],primary_names=names,
        comparison_names=shared['comparison_names'],Test_used=False,Test_files_read=False,
        evidence='Frozen V3 Train holdout: motion-conditioned horizon varies, but intact pose error is worse than zero/missing/shuffled event controls',
        evidence_PA_mm={k:v['pa_mpjpe_mm'] for k,v in diagnosis.items()},
        frozen_RGB_checkpoint_sha256=shared['frozen_RGB_checkpoint_sha256'],fusion_initialization_sha256=shared['fusion_initialization_sha256'],
        changes=['Fix every RGB backbone/head parameter and keep their module mode eval even during training',
            'All four event controls share the same Train-selected V3 part fusion initialization; reset only output projection to zero',
            'Train fusion only, with matched motion supervision/support/learning budgets; dynamic/global/static/fixed isolate temporal policy',
            'Keep strong v2_rgb_only as an explicitly reused constant RGB reference; do not call it a fresh V4 RGB training result'],
        unproven_hypothesis='Isolating fusion optimization from RGB weight adaptation may retain useful temporal selection without degrading the strong RGB representation',
        execution_requires='Matching CPU freeze/gradient/identity/reload audit, then actual registered CUDA 8-step pilots before main training',
        deadline_unix=1793321712)
    write(OUT/'v4_design_frozen.json',plan);print(json.dumps(plan,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
