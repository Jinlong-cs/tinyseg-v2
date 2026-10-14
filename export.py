# SPDX-License-Identifier: AGPL-3.0-only
"""Static opset11 main logits, checked against PyTorch on frozen validation images."""

import argparse
from pathlib import Path

from common import UPSTREAM, check_upstream, read_manifest, rgb_input, sha256, write_json

# isort: split

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

from train import load_checkpoint


def export(args):
    check_upstream()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = load_checkpoint(args.checkpoint).float().cpu().eval()
    x = np.random.default_rng(42).random((1, 3, 352, 640), dtype=np.float32)
    path = out / "model.onnx"
    with torch.no_grad():
        if model(torch.from_numpy(x)).shape != (1, 2, 44, 80):
            raise ValueError("Unexpected semantic output contract")
    torch.onnx.export(
        model,
        torch.from_numpy(x),
        str(path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=11,
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    if any(o.version != 11 for o in graph.opset_import if o.domain in ("", "ai.onnx")):
        raise ValueError("X5 requires opset11")
    if {"ArgMax", "NonMaxSuppression"} & {n.op_type for n in graph.graph.node}:
        raise ValueError("Export must contain only raw logits")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 3, 352, 640] or session.get_outputs()[0].shape != [1, 2, 44, 80]:
        raise ValueError("ONNX shape contract mismatch")
    rows = [r for r in read_manifest(args.check_manifest) if r["split"] == "val"]
    if not rows or args.limit < 1:
        raise ValueError("At least one validation image is required")
    cases = []
    for row in [None] + rows[: args.limit]:
        a = x if row is None else rgb_input(row["image"])[None]
        with torch.no_grad():
            reference = model(torch.from_numpy(a)).numpy()
        prediction = session.run(None, {"images": a})[0]
        delta = np.abs(reference - prediction)
        masks = []
        for raw in (reference, prediction):
            z = cv2.resize(raw[0].transpose(1, 2, 0), (640, 352), interpolation=cv2.INTER_LINEAR)
            masks.append(z[:, :, 1] > z[:, :, 0])
        cases.append(
            dict(
                id="random_seed42" if row is None else row["id"],
                max_abs=float(delta.max()),
                finite=bool(np.isfinite(prediction).all()),
                over_tolerance=int((delta > 1e-4 + 1e-4 * np.abs(reference)).sum()),
                mask_flip_fraction=float((masks[0] != masks[1]).mean()),
            )
        )
    passed = all(c["finite"] and c["over_tolerance"] == 0 and c["mask_flip_fraction"] <= 1e-4 for c in cases)
    report = dict(
        passed=passed,
        checkpoint_sha256=sha256(args.checkpoint),
        onnx_sha256=sha256(path),
        upstream_commit=UPSTREAM,
        opset=11,
        input=[1, 3, 352, 640],
        output=[1, 2, 44, 80],
        names={0: "non_drivable", 1: "drivable"},
        preprocessing="whole-frame LINEAR RGB/255",
        postprocessing="bilinear both logits to 352x640, strict fg>bg; nearest mask to native size",
        torch=torch.__version__,
        onnx=onnx.__version__,
        onnxruntime=ort.__version__,
        check_manifest_sha256=sha256(args.check_manifest),
        atol=1e-4,
        rtol=1e-4,
        max_mask_flip_fraction=1e-4,
        cases=cases,
    )
    write_json(out / "export.json", report)
    if not passed:
        raise RuntimeError("ONNX numerical parity failed; do not compile this artifact")
    print(f"Export parity passed: {len(cases)} cases")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--check-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=64)
    export(p.parse_args())
