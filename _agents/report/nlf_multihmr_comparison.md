# NLF / Multi-HMR Test comparison

All 64 device0 sessions, 320 fixed segments, and 96,124 RGB frames. Body22 errors use the same 104,863 person-frames matched by every listed method. All pose errors are mm; lower is better. Missing detections are reported through coverage, not zero errors.

Only person-chunks matched at exact frame 0 by every listed model enter this common population. First-frame transforms remain fixed for all 10 seconds; frame alignment is recomputed independently.

| Model | First frame aligned MPJPE | First frame aligned PA-MPJPE | Frame aligned MPJPE | Frame aligned PA-MPJPE |
|---|---:|---:|---:|---:|
| Multi-HMR pretrained | 160.04 | 146.65 | 94.94 | 58.19 |
| NLF pretrained | 116.61 | 111.94 | 78.83 | 53.31 |
| Multi-HMR fine-tuned | **68.53** | **73.26** | **46.68** | **32.54** |

Multi-HMR fine-tuned results use a final selected checkpoint, evaluated on all 320 Test segments. Prediction source: `.experiment_storage/multihmr_finetune_14h/final_test/multihmr_finetuned`.

MPJPE alignment uses pelvis translation. PA alignment uses proper Sim(3), without reflection.

## Absolute placement, coverage, and observed throughput

| Model | Absolute MPJPE (mm) | Pelvis position (mm) | Matched GT | Observed fps |
|---|---:|---:|---:|---:|
| Multi-HMR pretrained | 272.30 | 252.69 | 97.941% | **7.21** |
| NLF pretrained | 127.63 | 101.80 | 98.691% | 5.12 |
| Multi-HMR fine-tuned | **70.58** | **52.81** | **98.723%** | 4.97 |

Observed fps includes clip decoding, model execution, joint recovery, and matching, but excludes initial checkpoint loading and session video staging. Runs can overlap with other GPU jobs; these figures do not establish a controlled hardware speed ratio. A separate isolated end-to-end benchmark is required.

PA aligns each person-frame with proper Sim(3), removing scale error. Absolute MPJPE and pelvis position therefore determine metric placement quality. Test is not used for checkpoint selection.

Only complete methods are listed. NLF fine-tuning remains pending until its final Test is complete. Exact source directories are recorded in comparison.json under method_sources.

## Motion consistency

| Model | Pelvis velocity error (mm/s) | Root-relative body22 acceleration error (mm/s²) |
|---|---:|---:|
| Multi-HMR pretrained | **316.30** | **10614.54** |
| NLF pretrained | 598.78 | 19105.87 |
| Multi-HMR fine-tuned | 368.06 | 11968.41 |

Temporal metrics require the same GT person to be matched by every listed model in consecutive source frames. They never bridge missing frames or segment boundaries. Frame intervals use source frame numbers and nominal video fps. These measure agreement with the dataset motion labels, whose own temporal noise may affect the values. No smoothing is applied.

## Whole-body 55-joint diagnostic

| Model | MPJPE (mm) | Root MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|---:|
| Multi-HMR pretrained | 264.38 | 120.26 | 72.19 |
| NLF pretrained | 129.76 | 89.80 | 59.12 |
| Multi-HMR fine-tuned | **83.27** | **68.59** | **50.76** |

## Recommendation status

This comparison is interim. A final overall recommendation requires both the fine-tuned NLF Test and the controlled runtime study. No missing method is assumed to outperform a completed one.
