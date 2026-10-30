"""Freeze semantic event representations using completed Train evidence only."""
from datetime import datetime,timezone,timedelta
from prepare_part_aggregation_v4 import ROOT,OUT,read,sha,write

def main():
    assert not (OUT/'v6_design_frozen.json').exists()
    controls=read(OUT/'diagnostics/v5_part_dynamic.json')['controls']
    assert controls['intact']['pa_mpjpe_mm']>=controls['missing_event']['pa_mpjpe_mm']
    assert controls['intact']['pa_mpjpe_mm']>=controls['shuffle_event']['pa_mpjpe_mm']
    base=read(OUT/'configs/v5_part_dynamic.json')
    shared=dict(base,architecture='calibrated_part_semantic_frozen_rgb_v6',architecture_version='calibrated_part_semantic_frozen_rgb_v6',
        event_ViT_size=448,event_ViT_shared_frozen_RGB=True,event_internal_encoding='fixed sqrt(log-count/log9), positive orange/negative cyan, weak3x3support',
        uniform_aggregation_warmup_steps=500,
        comparison_names=['v2_rgb_only','v6_fixed','v6_global_dynamic','v6_part_dynamic','v6_part_static'])
    sources=list(base['source_sha256'])+['src/multihmr/part_aggregation_v6.py']
    shared['source_sha256']={p:sha(ROOT/p) for p in sources}
    shared['v6_model_source_sha256']=shared['source_sha256']['src/multihmr/part_aggregation_v6.py']
    for p,h in shared['source_sha256'].items():
        target=OUT/'source_snapshots'/(h+'.txt')
        if not target.exists():target.write_bytes((ROOT/p).read_bytes())
    names=[]
    for suffix,variant,static in [('fixed','fixed_all',False),('global_dynamic','part_agnostic',False),('part_dynamic','part_aware',False),('part_static','part_aware',True)]:
        name='v6_'+suffix;names.append(name);cfg=dict(shared,name=name,variant=variant,static_part_horizon=static)
        write(OUT/'configs'/(name+'.json'),cfg)
        pilot=dict(cfg,name='pilot_'+name,pilot_steps=8,pilot_train_limit=32,pilot_validation_limit=16,workers=0,
            warmup_steps=2,uniform_aggregation_warmup_steps=2,pilot_training_sessions=['20260720_131207_863008_200_b'],
            pilot_validation_sessions=['20260721_092034_900028_101_b'])
        write(OUT/'configs'/('pilot_'+name+'.json'),pilot)
    plan=dict(design_frozen_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),source_sha256=shared['source_sha256'],
        architecture_version=shared['architecture_version'],primary_names=names,comparison_names=shared['comparison_names'],
        Test_used=False,Test_files_read=False,Train_holdout_evidence_PA_mm={k:v['pa_mpjpe_mm'] for k,v in controls.items()},
        changes=['Replace small learned event CNN with the same frozen RGB ViT inner encoder, shared weights and eval mode; independent trainable event projection starts from RGB query projection',
            'Deterministic internal pseudo-RGB transforms the same causal224 two-polarity log-count bins; resize448→ViT14 patches gives32 cells and original-coordinate stride7/centre3',
            'Initialize spatial Transformer residual as identity, event-use bias2 and zero fusion output; uniform500 update warmup for representation adaptation',
            'Keep fixed/global/part/static matched; same RGB/V3 Train initialization, native data, timestamps, geometry, motion supervision and update/time caps'],
        multiple_changes_not_single_factor_causal_claim=True,
        extra_compute='Three frozen448 event ViT forwards per RGB query; no separate trainable ViT weights',
        frozen_RGB_checkpoint_sha256=shared['frozen_RGB_checkpoint_sha256'],fusion_initialization_sha256=shared['fusion_initialization_sha256'],
        deadline_unix=1793321712,execution_requires='Matching single-owner CPU full-builder/shared-weight/projection/reload audit, real Train event feature probe, then four registered CUDA8step pilots before main')
    write(OUT/'v6_design_frozen.json',plan);print(plan)
if __name__=='__main__':main()
