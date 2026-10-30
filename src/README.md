# Human motion estimation source

Keep only this README and directories at the top level of `src/`. The layout follows the separation of executable entry points and reusable implementation in the reference `ChallengeHOR/` project.

| Location | Purpose |
|---|---|
| `multihmr/` | Bundled upstream Multi-HMR model, blocks, losses, training code, and local DINOv2 backbone source |
| `multihmr/data.py` | Multi-HMR training data preparation |
| `multihmr/event_model.py`, `event_data.py`, `event_inference.py` | Event-only input adaptation and camera-calibrated RGB/event feature fusion; controlled training frames and RGB-clock Test inference |
| `nlf/` | NLF model loading, output conversion, and training data support |
| `hamer/` | Bundled official HaMeR ViT-H/MANO source, GT hand crop data, and calibrated RGB3/Event4 fusion |
| `utils/` | Shared evaluation, geometry, metrics, and visualization |

`utils/event_encoding.py` implements causal split counts, time bins, time surfaces,
and the native gray event-frame baseline. `utils/video_frames.py` decodes exact
native frame intervals without changing evaluation output globals. Event-only
paths never decode RGB pixels; known camera calibration converts their output to
the common RGB camera coordinate system. Fusion uses camera rays/origins rather
than treating event pixels as RGB-aligned. Experiment cache preparation and wall
deadline/job scheduling helpers live in `tools/`, not in runtime model code.

Model-specific training and evaluation executables live in `scripts/train_multihmr.py`, `scripts/train_nlf.py`, `scripts/evaluate_multihmr.py`, and `scripts/evaluate_nlf.py`. Shared entry points are `scripts/train.py`, `scripts/demo.py`, `scripts/evaluate.py`, and `scripts/visualize.py`. Temporary experiments, inspections, benchmarks, monitors, and migration helpers belong in `tools/`.

Store experiment configurations, logs, metrics, and fine-tuned checkpoints in `exp/YYMMDD/{experiment}/`, with checkpoints under `checkpoints/`. Reuse pretrained weights and licensed SMPL-X assets from `data/`. Source datasets are read-only. Store geometric coordinates in metres and report pose errors in millimetres.

Hand entry points are `scripts/train_hamer.py` and `scripts/evaluate_hamer.py`.
The hand study uses `_h` sessions and explicitly authorized GT hand boxes,
converted from source RGB coordinates to rectified/resized RGB pixels. Its four
10-second alignment metrics use the wrist and 21 hand joints; it omits RPE.
See `hamer/README.md` for setup and reproducibility details.

## Browser research workspace

`apps/research-workbench/` contains the browser file/media editor, independent Chat and AI Agent panes, native Codex goal integration, and lightweight server/account telemetry, as explicitly requested by the user. Its backend is independent of motion training and evaluation.
