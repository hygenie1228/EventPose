# Human3R / Multi-HMR — GT evaluation on fixed random Test segments

The evaluation covers device0 in 64 Test sessions, with five random 10-second segments per video. It uses the original, zero-based RGB frame intervals `[start,end)` stored in `dataset/split.json`. All 96,124 RGB frames were processed at their native approximately 30 fps. Human3R state was reset at each segment.

Networks used AMP FP16; Multi-HMR used batch size 32. Multi-HMR completed the entire Test evaluation on GPUs 0–4 before Human3R started. Both methods used public pretrained weights without fine-tuning.

The same 55×10475 joint regressor was applied to GT and predicted SMPL-X meshes. The main table uses 22 body joints including the pelvis. Errors are shown in mm; lower is better. Internal coordinates and raw `_m` fields remain in metres and are multiplied by 1000 for display.

## 22 joints

| Method | MPJPE (mm) | Root-aligned MPJPE (mm) | Root-aligned PA-MPJPE (mm) |
|---|---:|---:|---:|
| Human3R | 417.31 | 101.09 | 59.71 |
| Multi-HMR | 274.23 | 94.68 | 58.04 |

**Multi-HMR is recommended when prioritizing metric-scale location accuracy (MPJPE) in this evaluation.** It achieved lower errors on all three 22-joint metrics. The methods received different camera information, as described below.

## 55 joints

| Method | MPJPE (mm) | Root-aligned MPJPE (mm) | Root-aligned PA-MPJPE (mm) |
|---|---:|---:|---:|
| Human3R | 423.82 | 122.28 | 72.08 |
| Multi-HMR | 266.78 | 120.27 | 72.39 |

These are instance-weighted averages over the same 106,739 person-frames matched by both methods, from 109,282 valid GT person-frames. Missed detections were not assigned zero error.

| Method | Matched GT person-frames | Fraction of valid GT |
|---|---:|---:|
| human3r | 106,965 | 97.880% |
| multi_hmr | 107,032 | 97.941% |

Alignment is performed independently for each person in each frame. MPJPE uses original camera coordinates with no alignment. Root alignment subtracts each prediction's and GT's pelvis (index 0). Following pelvis centering, PA independently fits a proper Sim(3) Procrustes transformation—rotation, positive isotropic scale, and translation—for each person-frame, without reflections. A single transformation is not shared across the sequence.

GT meshes were reconstructed with neutral SMPL-X, 16 betas, non-PCA hands, `flat_hand_mean=False`, and zero expression, then transformed to camera coordinates using calibration. GT correspondence uses the label's exact `source_rgb_position`, without interpolation.

People were associated by Hungarian assignment using 2D reprojection distances for visible GT body joints in front of the camera, gated at 25% of the image diagonal. Human3R estimated its own focal length; Multi-HMR received calibrated camera intrinsics. GT pose and depth were not inference inputs.

Verification covered all 320 segments × two methods, every GT row, independent MPJPE/root-MPJPE checks, and 2,560 SciPy PA cross-checks.

Raw results and sampling manifest: `exp/261023/humanpose_gt_segments/`. The discontinued full-video FP32 experiment is preserved separately and excluded from these results.
