# SPDX-License-Identifier: AGPL-3.0-only
"""Uniform validation of any saved binary semantic epoch."""

import argparse
from pathlib import Path

from common import check_upstream, read_manifest, sha256, write_json

# isort: split

import torch

from train import load_checkpoint, validate

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    check_upstream()
    torch.set_num_threads(4)
    rows = [r for r in read_manifest(args.manifest) if r["split"] == "val"]
    if not rows:
        raise ValueError("No validation samples")
    model = load_checkpoint(args.checkpoint).to(args.device)
    metrics = validate(model, rows, args.device, args.batch)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out, dict(checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest), **metrics)
    )
