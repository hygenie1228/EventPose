# Measured Human3R / Multi-HMR throughput

Measured on 2026-10-23, per Quadro RTX 8000 GPU. This short benchmark is separate from the final accuracy evaluation.

Each method processed the first 100 frames of the same two Test device0 videos: `20260722_163417_881917_222_b` and `20260724_130105_753300_233_f`. Networks used AMP FP16. Timing excluded model loading and GT cache loading, but included video decoding, preprocessing, model execution, joint reconstruction, GT matching, and result storage. Human3R ran sequentially; Multi-HMR used batch size 32. Throughput may differ with other GPU workloads and across complete sessions.

| Segment | Human3R (fps) | Multi-HMR (fps) | Multi-HMR / Human3R throughput |
|---|---:|---:|---:|
| 222_b, first 100 frames | 3.475 | 6.742 | 1.94× |
| 233_f, first 100 frames | 3.578 | 7.554 | 2.11× |

64 videos × 10 seconds × 5 segments × 30 fps gives approximately 96,000 frames. Using the aggregate throughput of these two clips estimates about 2.52 hours for Human3R on three GPUs and 1.87 hours for Multi-HMR on two GPUs. Model loading, segment seeking, session initialization, final aggregation, and uneven work distribution add overhead; these measurements do not guarantee completion within two hours.

At the time of this benchmark, the earlier full-video evaluation used FP32 and batch size 1. These AMP measurements must not be presented as that run's observed throughput. AMP is not numerically identical to FP32, and benchmark predictions are excluded from final accuracy results.

Raw results: `exp/261023/humanpose_gt_eval/human3r_amp_benchmark.json` and `exp/261023/humanpose_gt_eval/multihmr_amp_end_to_end_benchmark.json`.

Benchmark code: `tools/benchmark_human3r_amp.py` and `tools/benchmark_multihmr_end_to_end.py`.
