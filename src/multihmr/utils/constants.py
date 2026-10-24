# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SMPLX_DIR = str(PROJECT_ROOT / 'data/human_model_files')
MEAN_PARAMS = str(PROJECT_ROOT / 'data/human_model_files/smpl/smpl_mean_params.npz')
CACHE_DIR_MULTIHMR = str(PROJECT_ROOT / 'data/human_model_files/pose_estimate')

ANNOT_DIR = 'data'
BEDLAM_DIR = 'data/BEDLAM'
EHF_DIR = 'data/EHF'
THREEDPW_DIR = 'data/3DPW'

SMPLX2SMPL_REGRESSOR = str(PROJECT_ROOT / 'data/human_model_files/smplx/smplx2smpl.pkl')
