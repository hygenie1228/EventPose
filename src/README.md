# Human motion estimation source

Keep only this README and directories at the top level of `src/`. The layout follows the separation of executable entry points and reusable implementation in the reference `ChallengeHOR/` project.

| Location | Purpose |
|---|---|
| `multihmr/` | Bundled upstream Multi-HMR model, blocks, losses, training code, and local DINOv2 backbone source |
| `multihmr/data.py` | Multi-HMR training data preparation |
| `nlf/` | NLF model loading, output conversion, and training data support |
| `utils/` | Shared evaluation, geometry, metrics, and visualization |

Model-specific training and evaluation executables live in `scripts/train_multihmr.py`, `scripts/train_nlf.py`, `scripts/evaluate_multihmr.py`, and `scripts/evaluate_nlf.py`. Shared entry points are `scripts/train.py`, `scripts/demo.py`, `scripts/evaluate.py`, and `scripts/visualize.py`. Temporary experiments, inspections, benchmarks, monitors, and migration helpers belong in `tools/`.

Store experiment configurations, logs, metrics, and fine-tuned checkpoints in `exp/YYMMDD/{experiment}/`, with checkpoints under `checkpoints/`. Reuse pretrained weights and licensed SMPL-X assets from `data/`. Source datasets are read-only. Store geometric coordinates in metres and report pose errors in millimetres.
