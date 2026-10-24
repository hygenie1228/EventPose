# EventPose

SMPL-X metric-scale human pose estimation, fine-tuning and device0 Test evaluation.

## Code

- `scripts/`: runnable entry points.
- `src/`: project learning, data, evaluation and reporting implementations.
- `src/multihmr/`: bundled official NAVER Multi-HMR source, including model, blocks, loss, datasets and original training code. Source attribution and local modifications are in `UPSTREAM.md`; original license and notice are retained.
- `src/multihmr/backbone_source/`: bundled DINOv2 backbone source. Backbone construction uses local source; no runtime GitHub code download is required.
- `src/nlf/`: copied official NLF source plus model loading, metric-output conversion, and training-data integration. Multi-HMR and NLF are regular repository source, not submodules.
- `thirdparties/Human3R/`: optional Human3R Git submodule, pinned by `.gitmodules` and the repository gitlink. Initialize it with `git submodule update --init --recursive` when using Human3R.
- `tools/`: auxiliary utility files and temporary compatibility entry points for existing jobs; no external-source directories.
- `data/`: pretrained weights and licensed human model assets.
- `exp/YYMMDD/{experiment}/`: configs, logs, results and fine-tuned `checkpoints/`.

The reference `ChallengeHOR/` checkout is for architecture reference and is not a runtime dependency. Normal training, demo, evaluation, and visualization code does not depend on `tools/`. Only support and compatibility files needed by already running experiments remain there; unused utility scripts are archived locally under `exp/_code_backup/` and excluded from Git.

## Local storage links

The current workstation uses these symbolic links at the repository root:

| Link | Target | Contents |
|---|---|---|
| `data` | `/home/namhj/share/experiment/EventPose/data` | Pretrained weights, licensed human models, and reusable assets |
| `dataset` | `/mnt/nfs_L/etri_processed` | Processed source sessions and Train/Test split; read-only |
| `exp` | `/home/namhj/share/experiment/EventPose/exp` | Dated experiment configurations, logs, predictions, metrics, and fine-tuned checkpoints |

These links and their contents are excluded from Git. Cloning this repository does not copy or recreate the storage targets. On another machine, mount the corresponding storage or prepare local directories and link them explicitly. Inspect existing paths before creating links:

```bash
ln -s /home/namhj/share/experiment/EventPose/data data
ln -s /mnt/nfs_L/etri_processed dataset
ln -s /home/namhj/share/experiment/EventPose/exp exp
```

The shared `data` and `exp` targets use the workstation's NFS-mounted storage. Selected active experiments also use `.experiment_storage/` on local SSD for image and intermediate caches, linked from their experiment directories to reduce repeated NFS I/O. Do not remove these caches while jobs are running. Fine-tuned weights are stored in each experiment's `checkpoints/` directory; legacy `data/finetuned/` links currently preserve compatibility with already launched jobs.

## Install and assets

Use Python 3.10 and install matching CUDA builds of torch 2.3.0 and torchvision 0.18.0, then `pip install -r requirements.txt`. Optional xformers must match the installed torch/CUDA; the bundled backbone supports its fallback path.

Prepare licensed SMPL-X and mean-shape assets separately:

```text
data/human_model_files/smplx/SMPLX_NEUTRAL.npz
data/human_model_files/smpl/smpl_mean_params.npz
data/human_model_files/pose_estimate/multiHMR_896_L.pt
```

Obtain SMPL-X from its official distribution and the checkpoint/mean parameters following `src/multihmr/README.md`. Weights, datasets and licensed body models are not committed. A repository clone contains the implementation; actual training/evaluation also requires these assets and the dataset. There is no dependency on a separate Multi-HMR clone or the developer's torch.hub source cache.

```bash
python scripts/train.py --method multihmr --help
python scripts/evaluate.py --method multihmr --help
python scripts/train.py --method nlf --help
python scripts/evaluate.py --method nlf --help
python scripts/demo.py --help
python scripts/visualize.py --help
```

Current experiment runners use the documented experiment manifests and prepared caches; inspect their arguments before launching. Do not start duplicate supervisors for the same experiment. Source datasets are read-only. Coordinates are stored in metres; reported pose errors are millimetres. Reports compare Test performance and include pretrained baselines.

## Browser research workspace

The optional [Research Workbench](src/apps/research-workbench/README.md) provides a server file explorer, code/Markdown/media viewer, and multiple chats using the signed-in local Codex CLI. It runs independently of training and evaluation. See its README for startup, access-key setup, and tests.
