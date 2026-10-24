# Partial results from the discontinued full-video FP32 experiment

This full-video FP32 experiment was stopped after the user changed the evaluation scope. These historical results cover only the shared saved portions of **two sessions, totaling 9,974 person-frames**. They are not results from the current random 10-second × 5-segment AMP evaluation in `dataset/split.json` and are excluded from its final aggregate.

Both methods use identical matched GT person-frames and 22 SMPL-X body joints. Errors below retain the historical metre units; lower is better.

| Method | MPJPE (m) | Root-aligned MPJPE (m) | Root-aligned PA-MPJPE (m) |
|---|---:|---:|---:|
| Human3R | 0.423571 | 0.109485 | 0.055720 |
| Multi-HMR | 0.318580 | 0.107192 | 0.055365 |

| Session | Shared GT rows | Shared person-frames | Last RGB index |
|---|---:|---:|---:|
| 20260722_163417_881917_222_b | 5,992 | 5,990 | 6,000 |
| 20260724_130105_753300_233_f | 3,992 | 3,984 | 4,000 |

This historical partial report is separate from the completed Test report. Partial values must not be interpreted as final whole-Test performance.
Evaluation definitions and complete-run results: [Test evaluation report](human3r_multihmr_gt_evaluation.md).
