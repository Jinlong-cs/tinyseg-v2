# Source provenance

Official source: https://github.com/ultralytics/ultralytics

Pinned commit: `3d75949c41f78a2c6763eb347ac7a94793098e89` (Ultralytics 8.4.175).
`requirements.txt` installs this exact source. This is a small independent adapter repository,
not a GitHub fork or an alternative implementation of the model. Upstream architecture and pretrained
weights remain owned by Ultralytics; the upstream AGPL-3.0 license is retained in `LICENSE`.

The export adapters derive from the previously exercised 352x640 X5 pipeline.
A source refactor does not certify a new checkpoint or compiled binary: each release still needs
float parity, quantized-runtime validation and X5 replay. No weights or business dataset are distributed here.
