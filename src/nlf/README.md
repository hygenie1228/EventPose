# NLF integration

Official NLF source is copied into this package; see `UPSTREAM.md`, `UPSTREAM_README.md`, and `LICENSE`. Local integration lives in `model.py` and `data.py`. Training and Test executables are `scripts/train_nlf.py` and `scripts/evaluate_nlf.py`.

The current pretrained asset is the official v0.2.0 `nlf_l_multi.torchscript`. Its fitted `vertices3d` and `joints3d` are absolute camera-space coordinates in millimetres. Evaluation applies the shared SMPL-X regressor to fitted vertices and converts coordinates to metres.

In this particular release, raw `trans` retains the translation from fitting centered points, while fitted vertices and joints have the predicted point mean added back. Use `standard_translation_m_v020` to export standard camera-space SMPL-X translation. Do not apply that correction blindly to newer upstream releases. Reconstruction with corrected translation matched the emitted fitted vertices within **0.000656 mm maximum absolute coordinate difference** on the smoke example. This checks output consistency, not accuracy against physical ground truth.

Zero-detection outputs may contain the 1,024-vertex fitting subset rather than the full 10,475-vertex mesh. `regressed_joints_m` handles this case explicitly as an empty `(0,55,3)` result.

Fine-tuning updates the localizer head and optional final backbone stages. The pretrained canonical field and detector stay frozen. The internal Train holdout selects checkpoints; Test is used only for reported comparisons. Checkpoints belong to each experiment's `checkpoints/` directory.
