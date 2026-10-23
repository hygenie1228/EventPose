# Workspace Instructions

## Immutable datasets

- Treat `dataset/` (server L) and `orig_dataset/` (server I) as strictly read-only.
- Never create, modify, rename, move, or delete any file or directory under either dataset root.
- Do not place caches, lock files, logs, temporary files, converted assets, or visualization outputs in either dataset.
- The user canceled the temporary RGBD copy into server L. Copy authorization is revoked. The only authorized cleanup exception is removing the newly copied `dataset/20260720_125338_040324_200_f/RGBD/device0.h5` through `device4.h5`, including partial copies. Preserve every `device*_mamma.h5` GT file. All other dataset writes remain prohibited; server I remains read-only.

## Experiments and outputs

- Run every experiment in a named subdirectory of a date-scoped directory: `exp/YYMMDD/{exp_name}/` (for example, `exp/261023/processed_visualization/`). Never place experiment results directly under the date directory.
- Store results, derived samples, visualization assets, logs, and manifests inside that dated experiment directory.
- Implement all Python scripts under `tools/`. Never create or implement Python files under `exp/`, including `exp/261023/`.
- Use a temporary directory inside the dated experiment directory when temporary files are required.

## Persistent model assets

- Store model checkpoints, downloaded model weights, human body model files, and other important reusable assets under `data/`.
- Reuse the SMPL/SMPL-X assets in `data/human_model_files/`; do not copy them into a dataset or experiment directory unless an isolated experiment explicitly requires a derived artifact.
- Keep large or important reusable outputs out of `exp/` when they are checkpoints or model assets; place them in an appropriately named subdirectory under `data/` instead.

## GT dataset processing

- Process all 283 server-L sessions with GT into `etri_processed/`, matching reference session `20260720_125338_040324_200_f` for available modalities.
- If corresponding server-I originals are absent, omit unavailable Depth and device5/6; do not fabricate data or wait for missing originals.
- Each session contains device directories only. Do not copy conversion logs, manifests, lock files, or top-level labels/RGB/Event directories.
- Device0–4 store `label.h5`, `event.npz`, `rgb.mp4`, and available lossless `depth.mp4`; device5/6 use server-I inputs and store RGB as `video.mp4` at 5 fps.
- Labels contain only `params`, `bboxes`, and `frames`, including required validity and frame mappings. Exclude meshes and person masks.
- Finish core data for every session before generating remaining `event.mp4` and `preview.mp4`. Preserve existing finished videos.
- Event visualization follows `tools/visualize_data.py`, includes every stored frame, and retains native resolution. Previews contain no text, last at most 60 seconds, and use reduced resolution with VS Code compatible encoding.
- Keep depth numerically lossless and validate decoded values against original uint16 frames.
- `etri_processed/split.json` contains exactly `{"train": [session names], "test": [session names]}`. Current split: Train 68 participant IDs / 219 sessions; Test 20 IDs / 64 sessions. Keep IDs disjoint, including co-occurring IDs. Store split metadata separately under the experiment directory.
- Prioritize Test sessions within each processing stage; generating Test visualization must still wait until all core data is complete.
- Run long conversion jobs in the background and check every 30 minutes. Keep the goal active until all required outputs have been verified.
- GPU jobs may use all eight GPUs, subject to actual encoder session limits.

## Repository and local storage

- Repository remote: `https://github.com/hygenie1228/EventPose`.
- Ignore external storage symlinks (`data`, `dataset`, `orig_dataset`, `etri_processed`, `exp`), `.experiment_storage/`, and Python caches in Git. Do not delete symlinks when asked to ignore them.
- `.experiment_storage/` is local SSD backing for selected experiment directories linked from `exp/261023/`, used to avoid slow NFS intermediate I/O. Active background jobs depend on it; do not remove it while in use.
