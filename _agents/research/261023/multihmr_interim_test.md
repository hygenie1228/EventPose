# Multi-HMR interim Test evaluation

Checkpoint: `/home/namhj/EventPose/data/finetuned/multihmr_261023_14h/test_snapshots/20261024_040315/model.pt` (legacy compatibility path).

Complete evaluation of 64 Test sessions, device0, and all 320 segments. Model selection used the internal training holdout.
Shared matched population: 107,000 person-frames. All errors are in mm.

| Model | MPJPE | Root-aligned MPJPE | PA-MPJPE |
|---|---:|---:|---:|
| pretrained | 274.58 | 94.77 | 58.10 |
| finetuned | 75.16 | 48.26 | 32.50 |

All 2,560 independent PA checks passed. This frozen interim snapshot is distinct from the model selected after training finishes.
