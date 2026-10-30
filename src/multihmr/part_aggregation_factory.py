"""Explicit architecture dispatch keeps frozen v1 checkpoints on their forward."""
def build_part_model(config,device='cuda',checkpoint=None):
    if config.get('architecture_version')=='calibrated_part_semantic_frozen_rgb_v6':
        from .part_aggregation_v6 import build_part_model as build
    elif config.get('architecture_version')=='calibrated_part_dense_frozen_rgb_v5':
        from .part_aggregation_v5 import build_part_model as build
    elif config.get('architecture_version')=='calibrated_part_frozen_rgb_v4':
        from .part_aggregation_v4 import build_part_model as build
    elif config.get('architecture_version')=='calibrated_part_motion_horizon_v3':
        from .part_aggregation_v3 import build_part_model as build
    elif config.get('architecture_version')=='calibrated_part_temporal_transformer_v2':
        from .part_aggregation_v2 import build_part_model as build
    else:
        from .part_aggregation_model import build_part_model as build
    return build(config,device,checkpoint)
