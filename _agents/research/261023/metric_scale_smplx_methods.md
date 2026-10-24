# Review of metric-scale SMPL-X human pose / motion estimation

Research date: 2026-10-23. Requirements: **SMPL-X-based recovery, absolute positions and motion in metres, and public training/fine-tuning code.**

## Overall recommendation: Human3R

**If selecting one method at the time of this review, choose [Human3R: Everyone Everywhere All at Once](https://fanegg.github.io/Human3R/).** It jointly estimates multi-person SMPL-X meshes, global human motion, and camera poses, with official training code and configurations. Its representation fits EventPose's SMPL-X GT and multi-person recordings and provides a reasonable RGB starting point for research into event inputs. [Official repository](https://github.com/fanegg/Human3R), [training documentation](https://github.com/fanegg/Human3R/blob/master/docs/train.md).

This is an overall requirements-based recommendation, not a claim of lowest error on every dataset. GVHMR should be compared for body/limb motion and static-camera conditions. The methods below primarily take RGB: **this review did not identify a completed event-only SMPL-X metric-scale training/inference system.**

Monocular metric scale is inferred from learned information. If physical distance accuracy is the priority and multiple cameras or depth are available at inference, correcting Human3R predictions using **multi-view/depth constraints with calibration in metres** is a stronger project design. This is a proposed extension, separate from Human3R's default features.

## 1. Distinctions required for metric scale

| Concept | Meaning | Relevance to this objective |
|---|---|---|
| Root-relative 3D pose | Pelvis-relative joints; body dimensions may be in metres, but actual location is absent | Insufficient |
| Camera-space absolute pose | Includes depth and xyz relative to the camera | Required for pose and distance estimation |
| World-space metric motion | Recovers xyz and trajectories in a fixed coordinate system | Final objective |
| Metric-scale prediction | Estimates physical dimensions using data and body/scene priors | What monocular methods provide |
| Measured scale anchoring | Observes scale through a known baseline, depth, or reference length | Strengthens physical scale accuracy |

SMPL-X files define the body-generation function and shape space. **Their presence does not determine the absolute distance or location of a person in a video.** Monocular pinhole observations remain ambiguous when body size and distance are scaled together. Known intrinsics improve projection but do not fully resolve this ambiguity.

Output in metres and centimetre-level accuracy are different claims. Low PA-MPJPE can coexist with incorrect absolute translation or travel distance. Consequently, metric-scale verification requires metrics without scale-fitting alignment.

## 2. Method comparison

“Public training” means an official training entry point, configuration, loss, or training guide was inspected, not merely that training is described in a paper. It does not imply an ETRI loader is already supplied.

| Method | Default input / scope | SMPL-X support | Metric / global position | Public training / fine-tuning | Assessment |
|---|---|---|---|---|---|
| **Human3R, ICLR 2026** | RGB sequence; multiple people, scene, camera | **Native SMPL-X** | Learned metric geometry and camera poses support global human reconstruction | **Available:** `src/train.py`, `config/trian_human3r.yaml`, BEDLAM loader, losses | **Overall first choice** for joint whole-body and global-motion training |
| **GVHMR, SIGGRAPH Asia 2024 / TPAMI 2026** | Video features, 2D joints, camera rotations; per-person motion | **SMPL-X body**; motion representation is body-focused | Body/motion priors provide metric trajectories; gravity coordinates must be connected to dataset world coordinates | **Available:** `tools/train.py`, Hydra configs | **Essential comparator**, especially for static-camera body motion; detailed hands/face require extensions |
| **Multi-HMR, SMPL-X variant of ECCV 2024** | Single RGB frame + intrinsics; multiple people, whole body | **Native SMPL-X** | Absolute camera-space pose and distance; known extrinsics permit world conversion | **Available:** `train.py`, BEDLAM preprocessing and training guide | **Simple metric pose baseline**; temporal modeling and tracking require additions |
| **SMPLest-X, TPAMI 2025** | RGB person crop; whole body | **Native SMPL-X** | Primarily pose/mesh recovery; metric global-motion system requires integration | **Available:** `main/train.py`, `scripts/train.sh` | Whole-body comparator; absolute translation and temporal modeling need attention |
| **SMPLer-X, NeurIPS 2023** | RGB person crop; whole body | **Native SMPL-X** | Pose/mesh-focused; global trajectories and scale anchoring need additions | **Available:** `main/train.py`, training/fine-tuning instructions | Useful adaptation baseline, but not the first choice for global motion |
| **PromptHMR / PromptHMR-video, CVPR 2025** | RGB images/video; multi-person global-motion pipeline | SMPL-X-family output | Camera/world reconstruction pipeline | **README explicitly states core training code is unavailable** | **Fails a required condition**; external models' train files are not PromptHMR training code |
| **MetricHMR → MetricHMSR, arXiv v4: 2026-06-30** | RGB image; metric human and scene recovery | Paper's human head uses **SMPL** | Directly targets absolute translation | Training implementation not verified in the officially linked repository | **SMPL-X and public-training requirements not established** |
| **SAM 3D Body** | RGB image; whole body | Default model is **MHR** | Body/camera outputs, distinct from a complete global-motion training system | Official release is inference-focused; complete training path not verified | **Fails native SMPL-X requirement** |
| **EventHPE, ICCV 2021** | Events, event flow, etc.; pose/shape | Default model is **SMPL** | Event-based body estimation, not the requested SMPL-X metric-world system | **Available:** `event_pose_estimation/train.py`, `flow_net/train.py` | Reference for event encoders/representations; **SMPL-X conversion and metric-world validation required** |
| **EasyMocap calibrated multi-view fitting** | Synchronized RGB cameras + K/R/t | SMPL-X fitting | Extrinsic translation in metres can anchor geometric scale | **Fitting optimization**, not pretrained-network fine-tuning | Useful geometric/scale baseline, but not a single trainable model meeting the request |

Primary sources:

- Human3R: [paper](https://arxiv.org/abs/2510.06219), [training](https://github.com/fanegg/Human3R/blob/master/docs/train.md).
- GVHMR: [official training commands](https://github.com/zju3dv/GVHMR#reproduce), [body pose/shape representation](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/model/gvhmr/utils/endecoder.py).
- Multi-HMR: [SMPL-X assets and training instructions](https://github.com/naver/multi-hmr#training-multi-hmr). The current repository also contains Anny models; **select an SMPL-X checkpoint**.
- SMPLest-X: [README](https://github.com/MotrixLab/SMPLest-X), [training script](https://github.com/MotrixLab/SMPLest-X/blob/main/scripts/train.sh). SMPLer-X: [repository](https://github.com/MotrixLab/SMPLer-X).
- PromptHMR: [unavailable training code](https://github.com/yufu-wang/PromptHMR#training).
- MetricHMSR: [paper](https://arxiv.org/abs/2506.09919), [project](https://metaverse-ai-lab-thu.github.io/MetricHMSR/), [officially linked code](https://github.com/Metaverse-AI-Lab-THU/MetricHMSR). The original MetricHMR title is distinguished from MetricHMSR.
- SAM 3D Body: [official inference repository](https://github.com/facebookresearch/sam-3d-body).
- EventHPE: [paper](https://arxiv.org/abs/2108.06819), [official training code](https://github.com/JimmyZou/EventHPE).
- EasyMocap: [SMPL-X multi-view example](https://github.com/zju3dv/EasyMocap/blob/master/doc/quickstart.md), [official requirement for extrinsics in metres](https://chingswy.github.io/easymocap-public-doc/quickstart/debug_mv1p.html).

## 3. Global-motion accuracy from one consistent source

The table comes from the [global human-motion results in Human3R v2](https://arxiv.org/html/2510.06219v2). Joint errors originally reported in mm were divided by 1,000 to retain this research note's historical metre presentation. RTE is a **percentage**, not a distance. Results from incompatible papers/settings were not combined into one ranking.

| Method | EMDB-2 WA-MPJPE ↓ (m) | EMDB-2 W-MPJPE ↓ (m) | EMDB-2 RTE ↓ (%) | RICH WA-MPJPE ↓ (m) | RICH W-MPJPE ↓ (m) | RICH RTE ↓ (%) |
|---|---:|---:|---:|---:|---:|---:|
| GVHMR | 0.1110 | 0.2765 | 2.0 | **0.0788** | **0.1263** | **2.4** |
| TRAM | 0.0764 | 0.2224 | 1.4 | 0.1278 | 0.2380 | 6.0 |
| JOSH | **0.0689** | **0.1747** | **1.3** | 0.0890 | 0.1325 | 3.0 |
| WHAM | 0.1356 | 0.3548 | 6.0 | 0.1084 | 0.1961 | 4.5 |
| **Human3R** | 0.1122 | 0.2679 | 2.2 | 0.1100 | 0.1849 | 3.3 |

WA-MPJPE aligns the entire 100-frame window; W-MPJPE aligns its first two frames. RTE uses rigid alignment without rescaling and normalizes by total travel distance. **These are not unaligned absolute ETRI errors.**

Human3R is not the most accurate method across this table. It is selected for the combination of global motion, multi-person whole-body SMPL-X, and public training. GVHMR's potential advantage should be tested directly on ETRI's predominantly static cameras. Lower TRAM/JOSH errors do not establish compliance with the required SMPL-X fine-tuning path.

## 4. Verified public training locations

| Candidate | Entry point | Configuration / data / loss evidence | ETRI adaptation required |
|---|---|---|---|
| **Human3R** | [`src/train.py`](https://github.com/fanegg/Human3R/blob/master/src/train.py) | [`config/trian_human3r.yaml`](https://github.com/fanegg/Human3R/blob/master/config/trian_human3r.yaml), [BEDLAM loader](https://github.com/fanegg/Human3R/blob/master/src/dust3r/datasets/bedlam.py), [SMPLLoss](https://github.com/fanegg/Human3R/blob/master/src/dust3r/losses.py) | Sequence loader, parameter/coordinate conversion, missing-depth masks, checkpoint initialization |
| **GVHMR** | [`tools/train.py`](https://github.com/zju3dv/GVHMR/blob/main/tools/train.py) | [`mixed.yaml`](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/configs/exp/gvhmr/mixed/mixed.yaml), [`gvhmr_pl.py`](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/model/gvhmr/gvhmr_pl.py) | Person clips, video features/2D joints, camera/gravity conventions, SMPL-X body targets |
| **Multi-HMR** | [`train.py`](https://github.com/naver/multi-hmr/blob/master/train.py) | [Official training instructions](https://github.com/naver/multi-hmr#training-multi-hmr) | RGB+K, multi-person SMPL-X camera-space targets, image loader |
| **SMPLest-X** | [`main/train.py`](https://github.com/MotrixLab/SMPLest-X/blob/main/main/train.py) | [`scripts/train.sh`](https://github.com/MotrixLab/SMPLest-X/blob/main/scripts/train.sh), [HumanData preparation](https://github.com/MotrixLab/SMPLest-X/blob/main/humandata_prep/README.md) | HumanData conversion, full-image/crop camera transforms, absolute translation loss, temporal extensions |

The official Human3R configuration is actually named **`trian_human3r`**, despite its apparent typo. Its training guide describes freezing CUT3R and Multi-HMR encoders while training human-related modules; this is the recommended adaptation starting point. [Documentation](https://github.com/fanegg/Human3R/blob/master/docs/train.md).

`pretrained` initializes CUT3R, `pretrained_mhmr` initializes Multi-HMR, and `resume` is also supported. **Resuming optimizer state and fine-tuning weights with a fresh optimizer are different operations.** A completed ETRI CLI was not executed or verified in this research task. [Configuration](https://github.com/fanegg/Human3R/blob/master/config/trian_human3r.yaml), [loading code](https://github.com/fanegg/Human3R/blob/master/src/train.py).

## 5. Connecting local SMPL-X assets and GT

The following existing files were verified locally and should be reused by configuring their paths, without downloading or copying replacements:

- `data/human_model_files/smplx/SMPLX_NEUTRAL.npz`
- `data/human_model_files/smplx/SMPLX_MALE.npz`
- `data/human_model_files/smplx/SMPLX_FEMALE.npz`
- `data/human_model_files/pose_estimate/multiHMR_896_L.pt`; its filename matches the initialization asset referenced by Human3R, but checkpoint keys/content still require validation at load time.

Human3R defaults to body assets under `src/models`; connect its loader to the local assets above. `SMPLModel` creates neutral 10/11-beta layers, and the default training loader expects BEDLAM parameters. Additional SMPL evaluation assets and regressors are referenced: **one SMPL-X NPZ is not sufficient evidence that every training/evaluation dependency is satisfied.** [Body loader](https://github.com/fanegg/Human3R/blob/master/src/dust3r/smpl_model.py).

| Local data / asset | Integration rule |
|---|---|
| `params/smplx_pose_axis_angle`, 165 dimensions | Verify ordering before splitting root/body/jaw/eyes/left/right hands; identical dimensionality does not imply matching head conventions |
| Human3R pose head | Layer expects **53×3** root+body+hands+jaw pose; local 55×3 eyes need separate handling |
| `params/smplx_betas`, 16 dimensions | Does not match 10/11-beta heads; initially use the first 10 matching shape-basis components and measure mesh differences, or extend/reinitialize a 16-beta head |
| BEDLAM's 11th shape component | Do not assign arbitrary 16-beta GT the same semantics; inspect shape/kid conventions |
| `params/translation_world_m` | Translation passed to SMPL-X; not automatically the head or pelvis joint location |
| `frames/*` and validity | Use exact source-frame mappings/timestamps; mask bbox, pose, and depth validity separately |

Parameter evidence: [SMPL-X layer](https://github.com/fanegg/Human3R/blob/master/src/dust3r/utils/smpl_layer.py), [GT generation](https://github.com/fanegg/Human3R/blob/master/src/dust3r/smpl_model.py), [BEDLAM loader](https://github.com/fanegg/Human3R/blob/master/src/dust3r/datasets/bedlam.py).

**Translation reference points require particular care.** Human3R's `person_center='head'` and some `smpl_transl` paths use the primary keypoint location. Direct L1 supervision against ETRI `translation_world_m` would therefore be incorrect. Generate common-coordinate vertices/joints through the same SMPL-X forward function, construct the correct target, and convert to standard SMPL-X `transl` on export.

Camera transforms must also distinguish pelvis locations from model translation. The local renderer uses `X_c = R_cw X_w + t_cw`. Points can be recovered as `X_w = R_cw^T(X_c - t_cw)`, but **SMPL-X root orientation/transl conversion must account for the shaped pelvis**; do not apply a point translation formula directly to root axis-angle parameters. Local references: `tools/visualize_processed_session.py`, `tools/render_smplx_overlay.py`.

## 6. Proposed ETRI fine-tuning design

This section proposes project extensions beyond upstream defaults. At the time of this research note, no loader implementation, installation, or training run was performed.

1. **Start with an RGB Human3R baseline.** Export per-person SMPL-X and camera/world vertices, and compare with GT using known calibration. Explicitly connect the model's reference world, possibly anchored to the first camera, to dataset world coordinates.
2. **Implement an ETRI sequence loader.** Prefer consecutive clips from one camera initially. Do not interpret upstream `num_views=4` as permission to insert four unrelated cameras without preserving timestamps, camera poses, and sequence semantics.
3. **Use supervised SMPL-X losses.** Include pose rotations, shape, camera-space absolute vertices/joints, correctly referenced translation, and 2D reprojection. Do not remove distance information by root centering or per-sample scale normalization before absolute losses.
4. **Add metric-scene supervision where available.** Use only valid measured depth; missing depth is not a 0 m target. Inspect required scene-loss keys/masks and zero-valid batches and modify them explicitly.
5. **Consider temporal losses and identity.** Joint/root velocity and acceleration must use actual time intervals. Associate multi-person detections across time and handle occlusion validity.
6. **Extend to events after verifying RGB.** Add an event-representation encoder with RGB teacher features/SMPL-X supervision, fusion, or distillation. Replicating a grayscale event image into three RGB channels is not evidence of a completed event model.

The existing [dataset analysis](../../dataset/data_analysis.md) lists 283 GT sessions: 171 with matching depth and 112 without. The 171 depth sessions are not all necessarily Train; intersect them with the actual split. Preserve Train 219 / Test 64 sessions and participant separation, with validation selected internally from Train.

GT consists of MAMMA-derived SMPL-X labels, so evaluation is not equivalent to independent physical metrology. Sensor depth, known camera baselines, and reference lengths provide additional scale checks.

## 7. Metre-valued output and verification conventions

| Export field | Units / meaning |
|---|---|
| `vertices_world_m`, `joints_world_m` | Metres; absolute dataset-world coordinates |
| `translation_world_m` | Metres; standard SMPL-X forward translation, distinct from head/pelvis location |
| `root_position_world_m` | Metres; actual root/pelvis position defined from the generated mesh/joints |
| `global_orient`, `body_pose`, hand/jaw/eye pose | Explicitly documented rotations, such as axis-angle in radians |
| `betas` | Dimensionless shape coefficients; record dimensionality and model version |
| `root_velocity_world_mps` | m/s, computed using actual timestamps |

Verify raw depth integer units and extrinsic translation units separately. A default `0.001` or a visualization range does not prove every sensor uses millimetres. For quantized meshes, apply storage metadata such as `vertex_scale_mm_per_unit / 1000`; do not apply this conversion again to an already metre-valued `translation_world_m`.

| Verification metric | Why it matters |
|---|---|
| Absolute MPJPE / PVE (m in this historical note) | Actual location/body error in a common dataset frame, **without alignment** |
| Root position error (m) | Distance and overall placement |
| Depth-axis error (m) | Separates difficult monocular camera-z error |
| Trajectory error (m), velocity error (m/s) | Travel distance, speed, and drift |
| Scale bias by body size and distance | Consistency of physical dimensions and placement |
| PA-MPJPE / root-relative MPJPE | Supplemental pose diagnostics, not primary evidence of metric-scale success |

**0.08 m equals 8 cm.** Do not present root error, global joint error, and PA-aligned error as interchangeable accuracy claims. Report Sim(3)-aligned results separately from absolute metric performance. Current project performance tables use mm; the metre values above preserve this historical research note's original numbers.

## 8. Reproducibility and limitations

Default-branch commits checked through the official GitHub API at review time:

| Repository | Verified commit |
|---|---|
| `fanegg/Human3R` | `402f2b2c7f20514e99cb42e4126c46b4ff75593f` |
| `zju3dv/GVHMR` | `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` |
| `naver/multi-hmr` | `651fb411e1cbcc626aaa5f38805ecab9cc891f7a` |
| `Metaverse-AI-Lab-THU/MetricHMSR` | `56510de8ff227de02c468cc951af34c6225b95c6` |

This review verified code availability and public training paths, not successful ETRI training, throughput, or accuracy. Human3R's training documentation describes a 48 GB GPU; inference-memory figures must not substitute for training requirements. Highest-accuracy and fastest configurations may differ, so one FPS figure cannot represent every variant.

**The original overall selection was Human3R**, combining native SMPL-X, multiple people, metric global motion, and public fine-tuning. Compare GVHMR for body-motion accuracy and SMPL-X Multi-HMR for single-frame absolute pose. Verify physical scale through ETRI calibration/depth and unaligned errors. Later measured Test reports should take precedence over this initial requirements-based recommendation.
