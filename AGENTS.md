# Code conventions

Keep this repository a small adapter around the pinned official Ultralytics source.
Reuse upstream owners; do not copy the upstream model implementation or add a second framework.
Keep training, ONNX export and X5 compilation separate. Do not edit model forward paths during training.
Preserve the 640x352 full-frame resize contract and validate numerical parity after export changes.
Use explicit paths and record source hashes. Keep weights, datasets, credentials and generated evidence out of Git.
Prefer deleting duplication to adding abstractions. Avoid fallback modes, experiment controllers and verbose routine logs.
Keep errors that identify invalid data, incompatible models or failed compiler commands.
Validate real data/model paths where available; do not replace numerical checks with mocked hardware tests.
Use feature branches for later changes; never force-push or overwrite another task's work.
