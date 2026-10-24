# Multi-HMR alternatives: SMPL-X, metric scale, speed, and training code

Research date: 2026-10-24. This is a review of official papers and code, not a benchmark of candidate models on the current RTX 8000. Existing Multi-HMR training/Test jobs were retained.

## Recommendation

For one additional comparison, choose **NLF with SMPL-X output**. It provides SMPL-X output and public PyTorch/TensorFlow training code, converting nonparametric 3D predictions through efficient SMPL-family fitting. However, it is a NeurIPS 2024 method rather than a 2025–2026 release, and higher speed than the current Multi-HMR implementation has not yet been demonstrated. It does not directly regress SMPL-X parameters.

This review did not verify any candidate satisfying all of: released after 2024, native SMPL-X, metric camera translation, public training code, and faster end-to-end processing than the current Multi-HMR. SMPLest-X (2025) is a newer candidate, but its large model and recovery of absolute original-camera placement require care.

## Comparison

| Method | Release | SMPL-X | Absolute scale / coordinates | Training code | Speed evidence and assessment |
|---|---|---|---|---|---|
| NLF | NeurIPS 2024 | Supported through nonparametric prediction → SMPL-X fitting | Camera-space 3D and translation outputs; check actual K and each output's units before normalizing to metres | Public PyTorch/TF | Efficient fitting; full detector+inference+fitting speed on this hardware not measured at review time. Recommended next comparison |
| SMPLest-X | 2025, TPAMI | Native | SMPL-X mesh and camera translation; distinguish fixed-focal crop translation from calibrated original-camera placement | Public | Huge model; no verified speed advantage over Multi-HMR |
| PromptHMR / video | CVPR 2025; additional BEDLAM2 weights | Native | Camera-space placement and video world reconstruction; full pipeline includes SLAM, Metric3D, and other components | Official training code unavailable | Motion-recovery candidate, but speed including preprocessing is unverified and the training-code requirement is unmet |
| SMPLer-X-S/B | ICCV 2023; later weights | Native | Crop-camera recovery; calibrated absolute translation needs checking | Public | Official V100 batch1: S 36.17 / B 33.09 FPS. Do not interpret as detector+decode+I/O throughput. Not a newer method |
| Multi-HMR 2 | 2026-06 | **Anny-based** | Metric camera-space recovery and tracking | Public repository documents core inference | New metric candidate, excluded by the SMPL-X requirement |
| Fast SAM 3D Body | 2026-03 | **MHR-based**; separate SMPL conversion | Conversion and camera conventions need checking | Training-code requirement not verified in this review | Paper reports up to 10.9× over original SAM3DB, not over Multi-HMR. Excluded as a native SMPL-X method |
| SAT-HMR | CVPR 2025 | **SMPL-based** | Camera-space multi-person recovery | Training code not checked in this review | Paper reports 24 FPS / up to 5× speedup, but it does not meet the SMPL-X requirement |

## NLF integration checks

The official demo exposes `detect_smpl_batched(frame_batch, model_name='smplx')` and returns `pose`, `betas`, `trans`, `vertices3d`, and `joints3d`. Verify mm/m conventions for each raw output; store geometric coordinates in metres and display evaluation errors in millimetres. Evaluate hands, face, and temporal stability separately. Applying an image model to video does not itself provide temporal consistency or identity tracking.

Compare speed on the same RTX 8000, Test segments, resolution, person count, precision, and batch size. Include RGB reading/decoding, detection, model execution, SMPL-X fitting, and storage, and report GPU inference separately. Metric scale must also be evaluated through unaligned MPJPE, pelvis position error, and matching coverage; PA-MPJPE alone removes scale error.

## Official sources

- [NLF implementation and training-code availability](https://github.com/isarandi/nlf)
- [NLF official demo: SMPL-X API](https://github.com/isarandi/nlf/blob/main/demo.ipynb)
- [NLF project: nonparametric prediction and parametric fitting](https://istvansarandi.com/nlf/)
- [SMPLFitter: supported models and fitting throughput](https://github.com/isarandi/smplfitter); fitting-only throughput is not full inference speed.
- [SMPLest-X official code and training commands](https://github.com/MotrixLab/SMPLest-X)
- [SMPLest-X camera translation implementation](https://github.com/MotrixLab/SMPLest-X/blob/main/models/SMPLest_X.py)
- [PromptHMR: unavailable training code and video pipeline](https://github.com/yufu-wang/PromptHMR)
- [SMPLer-X: V100 batch1 FPS table](https://github.com/MotrixLab/SMPLer-X)
- [Multi-HMR 2 official Anny implementation](https://github.com/naver/multi-hmr2)
- [Fast SAM 3D Body paper](https://arxiv.org/abs/2603.15603)
- [SAT-HMR paper: SMPL and throughput](https://openaccess.thecvf.com/content/CVPR2025/papers/Su_SAT-HMR_Real-Time_Multi-Person_3D_Mesh_Estimation_via_Scale-Adaptive_Tokens_CVPR_2025_paper.pdf)
