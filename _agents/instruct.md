# Workspace Instructions

## Code organization — explicit user requirements

- Use `/home/namhj/EventPose/ChallengeHOR` as an architectural reference only; it is not a runtime dependency.
- `scripts/` contains only user-facing launchers for training, demo/inference, evaluation, and visualization. Do not put temporary experiment scripts, benchmarks, migration scripts, inspections, audits, monitors, or one-off helpers there; these belong in `tools/`.
- `src/` is model-first: Multi-HMR code and its data code belong in `src/multihmr/`; NLF code and its data code belong in `src/nlf/`. Common evaluation, metrics, and visualization implementations belong in `src/utils/`. Its top level contains only `README.md` and directories, never loose Python files. Do not create generic `src/data`, `src/evaluation`, `src/inference`, `src/training`, or `src/visualization` directories.
- Put the executable training/evaluation implementation in the respective `scripts/train_multihmr.py`, `scripts/train_nlf.py`, `scripts/evaluate_multihmr.py`, and `scripts/evaluate_nlf.py`; do not create separate generic training/evaluation source folders. Reusable model/data/metric helpers remain in the model directory or `src/utils/`.
- Put Multi-HMR implementation directly in `src/multihmr/`, copied substantially unchanged from upstream. Preserve upstream licenses/notices; limit integration edits to necessary imports and paths. Include the backbone source needed to construct the model, so cloning this Git repository does not require a separate Multi-HMR checkout or a developer-only torch.hub cache.
- Reserve `thirdparties/` for external mask/image-processing components; do not put the main Multi-HMR implementation there.
- Manage remaining external repositories under `thirdparties/` as pinned Git submodules, following ChallengeHOR. Multi-HMR and NLF are copied source under `src/`, never submodules. `tools/` must contain files only, not source directories; remove the obsolete `tools/third_party/` layout after preserving dependencies.
- Keep temporary/auxiliary code in `tools/`. Compatibility wrappers for live processes may remain temporarily, but do not duplicate the core implementation there.
- Store each experiment's fine-tuned checkpoints in its own `exp/YYMMDD/{experiment}/checkpoints/`, not `data/finetuned/`. Pretrained weights and licensed SMPL-X assets remain in `data/` and must have documented setup steps.

## Pose metric reporting

- Do not append checkpoint-stage/status labels such as "frozen interim model" to model names in performance tables. The user allows results from training-intermediate checkpoints; they objected to the label, not the evaluation. Keep checkpoint provenance in a separate supporting paragraph, and never misrepresent an intermediate checkpoint as a final one.
- Bold the best value in each metric column (lower error, higher coverage/throughput). Do not claim a best value when the comparison is still incomplete.
- The Multi-HMR main table uses 10-second chunks and four mm-valued metrics: first-frame-aligned MPJPE, first-frame-aligned PA-MPJPE, frame-aligned MPJPE, frame-aligned PA-MPJPE. First-frame alignment uses exactly frame 0 and stays fixed across the chunk; frame alignment is recomputed per frame. MPJPE alignment is pelvis translation; PA is proper Sim(3). Omit the status column.
- Archive unused temporary Python utilities under `exp/_code_backup/`. Keep only currently needed experiment support in `tools/`; normal training/demo/evaluation/visualization must work without `tools/` source dependencies.

- Display pose estimation errors (MPJPE and PA-MPJPE variants) in millimetres (mm) in user-facing tables and reports. Preserve metre-valued geometric coordinates and explicitly named `_m` raw metric fields internally; multiply by 1000 for display.

## Immutable datasets

- Treat `dataset/` (server L) and `orig_dataset/` (server I) as strictly read-only.
- Never create, modify, rename, move, or delete any file or directory under either dataset root.
- Do not place caches, lock files, logs, temporary files, converted assets, or visualization outputs in either dataset.
- The user canceled the temporary RGBD copy into server L. Copy authorization is revoked. The only authorized cleanup exception is removing the newly copied `dataset/20260720_125338_040324_200_f/RGBD/device0.h5` through `device4.h5`, including partial copies. Preserve every `device*_mamma.h5` GT file. All other dataset writes remain prohibited; server I remains read-only.

## Experiments and outputs

- Run every experiment in a named subdirectory of a date-scoped directory: `exp/YYMMDD/{exp_name}/` (for example, `exp/261023/processed_visualization/`). Never place experiment results directly under the date directory.
- Store results, derived samples, visualization assets, logs, and manifests inside that dated experiment directory.
- Put runnable experiment entry points in `scripts/` and learning/evaluation implementations in `src/`. Keep `tools/` compatibility links only where live jobs still need old paths. Never implement Python files under `exp/`.
- Use a temporary directory inside the dated experiment directory when temporary files are required.

## Persistent model assets

- Store pretrained weights and human body model files under `data/`. Store experiment fine-tuned checkpoints inside that experiment's `exp/YYMMDD/{exp_name}/checkpoints/` (user instruction). Existing `data/finetuned` links are only live-job compatibility paths.
- Reuse the SMPL/SMPL-X assets in `data/human_model_files/`; do not copy them into a dataset or experiment directory unless an isolated experiment explicitly requires a derived artifact.
- Keep downloaded reusable assets in `data/`; experiment-specific fine-tuned checkpoints belong to the corresponding experiment directory.

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

- Store research notes under `_agents/research/YYMMDD/` (for example `261023/` and `261024/`). Write all documents in `_agents/research/261023/` in English, as explicitly requested by the user. Keep `src/README.md` in English.

- Repository remote: `https://github.com/hygenie1228/EventPose`.
- Ignore external storage symlinks (`data`, `dataset`, `orig_dataset`, `etri_processed`, `exp`), `.experiment_storage/`, and Python caches in Git. Do not delete symlinks when asked to ignore them.
- `.experiment_storage/` is local SSD backing for selected experiment directories linked from `exp/261023/`, used to avoid slow NFS intermediate I/O. Active background jobs depend on it; do not remove it while in use.

## Fine-tuning monitoring frequency

- During fine-tuning, inspect progress at most once every 30 minutes unless the user asks for an update or a concrete failure requires intervention. Avoid frequent model/tool polling and repetitive status messages to conserve tokens. Lightweight background recovery supervisors may continue running.

## Browser workspace

- Keep the browser research workspace under `apps/research-workbench/`, separate from motion-estimation `src/` and `scripts/`.
- Provide left-to-right file explorer, file/code/media viewer, and multiple independent AI chats, using the currently signed-in Codex CLI.
- Bind the remote workspace to the server network interface and a browser-compatible port >= 10000; the current deployment uses `0.0.0.0:18080`.
- Implement routine choices without asking clarification questions. Long training/evaluation progress checks remain limited to once every 30 minutes unless intervention or a user-requested update is needed.

- Remote workbench reachability: UFW is enabled on this server. A server-local HTTP 200 does not prove access from the user's computer. Do not claim external access was verified without an external client check. Opening TCP 18080 requires an administrator if passwordless sudo is unavailable.

- Persist all browser Agent user/assistant/tool events under `_agents/research/YYMMDD/agent_sessions/` as complete JSONL and readable Markdown. Record this agent's implementation/research notes under the same date-scoped research directory. Put deliverable reports under `_agents/report/`.
- Workbench defaults: white/light theme, readable 15–16px text, prominent 30px close buttons, one Codex panel on the right and a second below the viewer, and top-bar one/two-panel buttons and light/dark theme buttons. The bottom panel must resize up to 50% of the workspace height. Each viewer/chat must scroll independently and keep the chat composer visible; show file/chat loading states.
