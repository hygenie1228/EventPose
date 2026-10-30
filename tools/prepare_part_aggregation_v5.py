"""Freeze a Train-only dense-grid follow-up, preserving all input observations."""
from datetime import datetime,timezone,timedelta
from prepare_part_aggregation_v4 import ROOT,OUT,read,sha,write

def main():
    assert not (OUT/'v5_design_frozen.json').exists()
    controls=read(OUT/'diagnostics/v4_part_dynamic.json')['controls']
    assert controls['intact']['pa_mpjpe_mm']>controls['missing_event']['pa_mpjpe_mm']
    spatial=read(OUT/'diagnostics/training_event_spatial_scale.json');assert spatial['Train_only']
    base=read(OUT/'configs/v4_part_dynamic.json')
    shared=dict(base,architecture='calibrated_part_dense_frozen_rgb_v5',architecture_version='calibrated_part_dense_frozen_rgb_v5',
        event_feature_grid=32,event_patch_stride=7,part_heatmap_sigma_rgb_px=72.8,
        comparison_names=['v2_rgb_only','v5_fixed','v5_global_dynamic','v5_part_dynamic','v5_part_static'])
    sources=list(base['source_sha256'])+['src/multihmr/part_aggregation_v5.py']
    shared['source_sha256']={p:sha(ROOT/p) for p in sources}
    shared['v5_model_source_sha256']=shared['source_sha256']['src/multihmr/part_aggregation_v5.py']
    for p,h in shared['source_sha256'].items():
        target=OUT/'source_snapshots'/(h+'.txt')
        if not target.exists():target.write_bytes((ROOT/p).read_bytes())
    names=[]
    for suffix,variant,static in [('fixed','fixed_all',False),('global_dynamic','part_agnostic',False),('part_dynamic','part_aware',False),('part_static','part_aware',True)]:
        name='v5_'+suffix;names.append(name);cfg=dict(shared,name=name,variant=variant,static_part_horizon=static)
        write(OUT/'configs'/(name+'.json'),cfg)
        pilot=dict(cfg,name='pilot_'+name,pilot_steps=8,pilot_train_limit=32,pilot_validation_limit=16,workers=0,
            warmup_steps=2,uniform_aggregation_warmup_steps=2,pilot_training_sessions=['20260720_131207_863008_200_b'],
            pilot_validation_sessions=['20260721_092034_900028_101_b'])
        write(OUT/'configs'/('pilot_'+name+'.json'),pilot)
    plan=dict(design_frozen_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),source_sha256=shared['source_sha256'],
        architecture_version=shared['architecture_version'],primary_names=names,comparison_names=shared['comparison_names'],
        Test_used=False,Test_files_read=False,Train_holdout_evidence_PA_mm={k:v['pa_mpjpe_mm'] for k,v in controls.items()},
        spatial_evidence='Train median arm200ms paths21.665/22.797px versus current60 native pixels per feature stride; stride7 gives30px. This is scale evidence, not proof of causality or gain.',
        changes=['Event/RGB aggregation feature grid16→32, Conv14/stride14→Conv7/stride7; same original224 event bins',
            'Calibrated sampling, local occupancy, support statistics all use exact stride7 centre3 coordinates',
            'Preserve weak part supervision physical RGB sigma72.8px; preserve null semantics and motion horizon policy',
            'Same frozen strong RGB and same V3 coarse fusion initialization as V4; sum 2x2 kernel coefficients and interpolate positional embeddings; zero output projection'],
        frozen_RGB_checkpoint_sha256=shared['frozen_RGB_checkpoint_sha256'],fusion_initialization_sha256=shared['fusion_initialization_sha256'],
        deadline_unix=1793321712,execution_requires='Matching CPU synthetic geometry/control/gradient/reload tests plus actual full builder/frozen RGB parameter audit, then four registered real CUDA8step pilots before main')
    write(OUT/'v5_design_frozen.json',plan);print(plan)
if __name__=='__main__':main()
