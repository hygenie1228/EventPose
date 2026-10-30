# HaMeR hand integration

Official source: https://github.com/geopavlakos/hamer at commit
`3a01849f4148352e9260b69bf28b65d1671a4905`. Preserve `LICENSE.md` and
`UPSTREAM_README.md`. Upstream ViT-H, MANO transformer decoder, MANO wrapper,
and geometry source are bundled here. Package initializers avoid eager imports
of optional upstream demo renderers and detectors. `event_model.py` constructs
the released backbone and decoder directly and verifies their loaded weights.

Use `.venv-humanpose/bin/python`. Core dependencies already provided by the
workspace are torch, torchvision, timm, einops, yacs, smplx, chumpy, numpy,
scipy, h5py, OpenCV, scikit-image, Pillow, psutil, requests and pynvml.
No detector/ViTPose installation is needed for this GT-crop evaluation.

Keep `dataset/` and `data/` read-only. Existing licensed MANO assets must be
available at `data/human_model_files/mano/MANO_RIGHT.pkl`. The official
pretrained bundle is downloaded from the URL in upstream `fetch_demo_data.sh`
to `exp/261028/hamer_hand/assets/`. `tools/setup_hamer_hand.py` downloads or
resumes this archive, extracts only the HaMeR checkpoint/config/mean parameters,
and records their SHA256 hashes in `pretrained_manifest.json`.

For the frozen four-condition study:

```bash
.venv-humanpose/bin/python tools/prepare_hamer_hand.py audit
.venv-humanpose/bin/python tools/prepare_hamer_hand.py sample
.venv-humanpose/bin/python tools/check_hamer_hand.py cpu
.venv-humanpose/bin/python tools/prepare_hamer_hand.py cache --workers 4
.venv-humanpose/bin/python tools/run_hamer_hand.py
```

The last two commands are normally started as separately registered background
jobs; the scheduler waits for required caches. Do not start duplicates. Only
physical GPU0 is authorized for this study. The scheduler verifies its UUID and
idle compute processes, checks the actual pretrained model, requires eight-step
gradient pilots, then runs RGB for 90 minutes and each RGB+Event encoding for
three hours. Each trial starts from official HaMeR weights. The original RGB
checkpoint is evaluated separately without an extra training trial.

Executable learning/evaluation loops live in `scripts/train_hamer.py` and
`scripts/evaluate_hamer.py`; helpers in `tools/` are preparation/scheduling/audit
utilities, never runtime dependencies of these two entry points. RGB uses the
official 256-square crop with the ViT's 192-wide central region. Fusion uses
the body study's count33/bins100/surface20 encoders, fixed gain4, independent
event CNN, calibration-only fixed-depth ray sampling and zero-initialized
residual projection. Event images remain in their own camera coordinates.

The protocol is conditional hand reconstruction using GT boxes and anatomical
left/right, not hand detection. GT comes from independently reconstructed
SMPL-X 21-joint hands. SMPL-X betas and pose parameters are never treated as
MANO-compatible labels. GT joints are stripped from model inputs. Subject
holdout selects checkpoints; Test never selects methods or checkpoints.

Four error columns use a wrist translation or proper Sim(3) fitted at exactly
the original segment frame0 (fixed) or separately each frame. All four use the
same eligible samples. Missing original frame0 sequences are explicitly counted.
RPE and RPE-Z are omitted. Coordinates/raw errors remain metres; reports display
millimetres. Reports are Korean and refresh at most every 30 minutes during
long training/evaluation. Completion requires all four training runs, five full
Test rows including original HaMeR, raw predictions, and final artifact checks.
