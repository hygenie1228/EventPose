# Agent prompt: RGB-only Multi-HMR fine-tuning

You are taking over the preparation of an RGB-only, multi-person 3D human pose fine-tuning experiment in `/home/namhj/EventPose`. Reproduce the techniques described below using the implementation and completed experiment artifacts as authoritative evidence. Explain any intentional deviations before comparing results.

## Current operating state

The user explicitly requested that GPU computation be paused. Existing training processes and their scheduler have been suspended, and the original Goal is paused. **Do not resume, replace, terminate, or duplicate those processes. This document does not authorize starting GPU work.** Prepare your implementation and configuration separately; run training only after the user requests it.

Read `_agents/instruct.md` before modifying the workspace. Preserve other agents' work. Treat `dataset/`, `orig_dataset/`, `etri_processed/`, and `data/` as read-only. Put new experiment outputs and checkpoints in a separate `exp/YYMMDD/{experiment_name}/` directory. Use `.venv-humanpose/bin/python` for scientific computation.

The latest GPU allocation for this project is physical GPUs 2–6, subject to the current pause. Historical GPU assignments in old configuration files are not current authorization.

## Objective

Adapt the original RGB Multi-HMR model to device0 RGB images with calibrated intrinsics and metric 3D supervision. Improve pose accuracy while preserving detection coverage and absolute position accuracy. Use Train-only internal validation for model selection, then evaluate the original and fine-tuned models on the same frozen Test population.

Use RGB pixels as the only sensor input. GT pose, identity, shape, and projected centers are supervision or evaluation data; never feed them into the final inference pipeline. Do not use event images, depth, GT crops, or GT person boxes as inference inputs.

## Read these files first

- `scripts/train_multihmr.py`: model freezing, target construction, losses, optimization, validation, checkpointing; also contains the newer single-GPU controlled trial.
- `src/multihmr/data.py`: original RGB dataset preparation and participant-disjoint split.
- `src/multihmr/event_data.py`: shared-frame dataset used by the newer controlled RGB baseline; its RGB mode does not require event pixels as model inputs.
- `src/utils/pose_evaluation.py`: original checkpoint loading, matching, and result collection.
- `scripts/evaluate_multihmr.py`: actual detection-based Test inference.
- `exp/261023/multihmr_finetune_14h/manifest.json`, `wave1.json`, `wave2.json`, `training_overrides.json`, `experiment_summary.json`, and `selected.json`.
- `exp/261024/rgb_event_36h/configs/rgb_baseline.json`: newer 12-hour RGB control configuration.

Inspect the code rather than assuming the old launchers support arbitrary output directories: the original and newer implementations currently use experiment-specific roots. Adapt those paths for a new experiment before launching anything. Do not reuse historical `name`, checkpoint, resume, or deadline fields unchanged.

## Data protocol

1. Use device0 only. Keep the existing participant-disjoint Train/Test split: Train has 219 sessions / 68 participants; Test has 64 sessions / 20 participants.
2. Reuse the internal split in the original manifest: 187 training sessions and 32 holdout sessions. Keep co-occurring and repeated participants together. Do not randomly split adjacent frames into training and validation.
3. The original RGB preparation samples approximately 2 fps across Train videos and selects 64 fixed random GT frames per holdout session. Reuse the actual saved indices and targets for strict reproduction.
4. Distinguish two populations. The original RGB experiment uses its original RGB indices. The newer RGB/event control uses 115,838 training frames and 2,045 holdout frames after excluding frames without a sufficiently recent causal event observation, identically across trained conditions. This filtering is for controlled comparison, not an inherent requirement of RGB-only learning. Do not silently mix these populations.
5. Store decoded RGB training frames and derived targets on local SSD through the experiment directory. Do not write caches or lock files into source data directories.

## RGB preprocessing and geometry

- Load the original `data/human_model_files/pose_estimate/multiHMR_896_L.pt` checkpoint and the existing SMPL-X assets.
- Preserve image aspect ratio, resize into a 896×896 canvas using LANCZOS, and center-pad. Do not stretch the image to a square.
- Transform camera intrinsics with the actual resized width/height and padding offsets. Use the camera-coordinate GT and the repository's existing units and joint regressor.
- Normalize RGB with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Training RGB augmentation: `ColorJitter(0.2, 0.2, 0.15, 0.03)`. Disable augmentation in validation and Test.
- The selected historical recipe does **not** use horizontal flipping. A separate deeper trial tested flipping but was not selected. If you introduce flipping, correctly mirror camera X, left/right joints, rotations, and principal point; do not flip RGB alone.

## Previously successful recipe: a two-wave search

The historical experiment was named `multihmr_finetune_14h`. It was a bounded experiment window containing parallel candidates, preparation, validation, and refinement—not 14 hours of continuous training for every candidate.

All first-wave candidates started from the same original pretrained checkpoint:

| Candidate | Trainable backbone blocks | Head learning rate | Backbone learning rate | Nominal training time |
|---|---:|---:|---:|---:|
| Head-only | 0 | 5e-6 | 0 | 6 hours |
| Last four blocks | 4 | 5e-6 | 5e-7 | 6 hours |
| Last eight blocks | 8 | 1e-5 | 1e-6 | 6 hours |

Freeze all parameters first. Enable `mlp_classif`, `mlp_offset`, and `x_attention_head`. For backbone adaptation, enable only the last N transformer blocks and encoder normalization. Use activation checkpointing on enabled backbone blocks with `use_reentrant=False`.

Select the first-wave checkpoint on internal holdout PA-MPJPE with detection coverage checked. Continue from the selected last-eight-block checkpoint in a second wave:

| Candidate | Trainable backbone blocks | Head LR | Backbone LR | PA weight | Root weight |
|---|---:|---:|---:|---:|---:|
| Refine — selected historically | 8 | 3.5e-6 | 3.5e-7 | 100 | 25 |
| PA-focused comparison | 8 | 3.5e-6 | 3.5e-7 | 200 | 10 |
| Deeper comparison | 12 | 3.5e-6 | 1.75e-7 | 150 | 20 |

The second-wave nominal duration was approximately 6.67 hours. The deeper comparison used horizontal flip probability 0.5 through `training_overrides.json`; the selected refine recipe did not.

The historical launcher used two-GPU DDP per candidate, batch 16 per GPU, global batch 32, and two data workers per process. Do not launch that DDP branch as a single-process command: it expects `RANK` / `LOCAL_RANK` and world size 2.

This is a search-and-refinement recipe. Do not describe it as a mandatory head-only warmup followed sequentially by last-four and last-eight stages: the first-wave candidates ran independently.

## Newer, simpler 12-hour RGB control

For an independent half-day baseline, the newer implementation uses the following distinct recipe:

```json
{
  "mode": "rgb",
  "seed": 261024,
  "hours": 12,
  "last_blocks": 8,
  "head_lr": 0.00001,
  "backbone_lr": 0.000001,
  "batch_size": 16,
  "workers": 2,
  "pa_weight": 100,
  "root_weight": 25,
  "validation_minutes": 45,
  "backbone_warmup_seconds": 1800,
  "warmup_steps": 100,
  "augment": true,
  "minimum_coverage": 0.95
}
```

This branch uses one GPU, starts from the **original** checkpoint, trains heads first, and enables the last eight backbone blocks after 30 minutes. Its batch is 16 total, not the historical DDP global batch of 32. It is a fresh control experiment, not a continuation of the previously selected RGB model. At the user's pause, its final training and Test evaluation were not complete; do not claim it has reproduced the historical final Test score.

The JSON above describes the technique, not a complete launch configuration. The existing branch also requires its experiment discriminator, unique name, absolute deadline, prepared indices, and cache identity. Adapt the implementation to your independent experiment rather than pointing a new run at the suspended `rgb_baseline` directory.

## Optimization and numerical stability

- AdamW, weight decay 0.01; separate parameter groups for heads and backbone.
- First 100 optimizer steps: linear LR warmup. Apply wall-time cosine decay toward 10% of each group's initial LR over the configured duration.
- FP16 autocast and GradScaler with initial scale 512.
- Compute geometry losses in FP32. Require a finite total loss, unscale gradients, then clip global gradient norm to 1.0.
- Check gradient norms and scaler behavior. A skipped overflow step is not a successful weight update; record both attempted and successful updates.
- Log loss components, learning rates, elapsed time, successful steps, and peak allocated GPU memory. Report actual training time and updates; equal wall-time budgets do not imply equal numbers of examples processed.

## Supervision and loss

Reuse `targets`, `aligned_errors`, and `losses` from `scripts/train_multihmr.py` rather than implementing an approximate replacement.

- Project valid, visible GT person centers using the transformed intrinsics. They supervise detection cells and offsets on the 64×64 grid with patch size 14. Handle duplicate centers in the same cell as the existing implementation does.
- Training uses GT center indices to supervise pose outputs. **Validation and Test must use actual predicted detections**, with no GT center indices supplied to the model.
- Regress joints from predicted vertices using the existing regressor. Pose losses use the first 22 body joints.
- Fit a proper similarity alignment for PA loss without differentiating through the SVD; gradients still flow through the transformed predicted joints.
- Use camera-coordinate pelvis supervision to retain absolute translation information.

The selected historical and newer baseline loss is:

```text
L = 100 * PA_joint_error_m
  + 25 * pelvis_relative_joint_error_m
  + focal_style_detection_loss
  + smooth_L1_offset
  + 0.2 * smooth_L1_body_rotation
  + 0.02 * smooth_L1_shape
  + smooth_L1_absolute_pelvis_m
```

Absolute pelvis Smooth L1 uses beta 0.1 in metres. Internal geometry and pose losses use metres; display pose errors in millimetres. The shape and rotation terms supervise the parameters supported by the existing model implementation.

## Validation, selection, and checkpoints

1. Evaluate the original initialization on the fixed internal holdout before training.
2. Run detection-based validation approximately every 45 minutes and at the end. Use the same detection threshold and NMS as the existing validation code.
3. Select by body22 per-person per-frame PA-MPJPE, while checking detection coverage. Avoid improving the score merely by dropping difficult people.
4. The historical recipe required coverage at least initialization coverage minus 0.005; the selected refinement config records 0.9776858108108109. The newer controlled branch uses a 0.95 gate and explicitly ranks sub-threshold candidates by coverage first. Keep the chosen policy fixed and document it.
5. Save `last.pt` about every five minutes and `best.pt` on valid holdout improvement, using atomic replacement. Preserve optimizer, scaler, original start/deadline, successful update counts, and RNG states where supported. The newer branch saves RNG states; the legacy checkpoint implementation is less complete in this respect.
6. Distinguish a trainable-parameter delta from a standalone model. Reconstruct using the exact original checkpoint plus the delta before inference; do not treat a delta as full pretrained weights.
7. Verify selected checkpoint step, validation step, and saved metrics correspond. Do not label stale validation metrics as a measurement of a later checkpoint.
8. Existing fixed deadlines are historical metadata, not reusable budgets. A new run needs its own explicitly authorized budget; it must not reset or modify the suspended original Goal's deadline.

## Final evaluation and deliverables

- Compare original and fine-tuned RGB models on the same frozen Test: 64 sessions, 320 ten-second chunks, 96,124 RGB frames. Preserve original frame mappings and calibrated camera coordinates.
- Report the four main errors in mm: exact frame0-fixed pelvis-aligned MPJPE, frame0-fixed proper Sim(3) PA-MPJPE, per-frame pelvis-aligned MPJPE, and per-frame PA-MPJPE.
- Also report unaligned absolute MPJPE, pelvis position error, full-GT detection/matching coverage, consecutive-frame pelvis velocity error, and root-relative acceleration error.
- Use common matched populations for cross-model pose comparisons. Report the common sample count and full-GT coverage separately. Missing detections are not zero error.
- Benchmark both models sequentially on the same GPU and the same inputs. Include video staging/decoding, preprocessing, detection, reconstruction, matching, and saving; report model loading and peak memory separately.
- Preserve configuration, split and source identities, checkpoint provenance, logs, raw Test predictions, and a report whose numbers can be recomputed from those predictions.
- Inspect long-running training/evaluation at most every 30 minutes, except explicit user status requests or concrete failures. Do not repeatedly poll GPUs or relaunch a job because an observation timed out.

For context, the completed historical selected RGB model achieved internal holdout PA-MPJPE **33.28 mm**. In the earlier four-model comparison, its Test frame-aligned PA-MPJPE was **32.63 mm** on the four-model common population, and controlled full-pipeline throughput was approximately **7.347 fps** on GPU 3 / Quadro RTX 8000. These are distinct measurements and populations, not guaranteed targets for a new experiment. Verify against `exp/261024/nlf_finetune_12h/comparison.json` and its final audit before citing them.

Return a concrete plan/configuration and the implementation changes needed for your independent run. Keep the current GPU pause intact until the user requests execution.
