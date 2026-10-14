# SPDX-License-Identifier: AGPL-3.0-only
"""Train-only calibration and explicit Horizon X5 INT8 compilation."""

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

from common import read_manifest, rgb_input, sha256, write_json

# isort: split


def calibrate(args):
    import numpy as np

    rows = [r for r in read_manifest(args.manifest) if r["split"] == "train"]
    if not 0 < args.count <= len(rows):
        raise ValueError("Calibration count must not exceed unique train samples")
    # Round-robin domains without repeating rows, with seeded order inside each domain.
    rng = np.random.default_rng(42)
    groups = {}
    for row in rows:
        groups.setdefault(row.get("domain", "default"), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < args.count:
        for domain in sorted(groups):
            if groups[domain] and len(selected) < args.count:
                selected.append(groups[domain].pop())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    data_dir = out / "rgb"
    data_dir.mkdir()
    receipt = []
    for i, row in enumerate(selected):
        target = data_dir / f"{i:05}.rgb"
        rgb_input(row["image"], normalize=False).tofile(target)
        receipt.append(
            {
                "id": row["id"],
                "split": "train",
                "domain": row.get("domain", "default"),
                "image_sha256": sha256(row["image"]),
                "rgb_sha256": sha256(target),
            }
        )
    write_json(
        out / "manifest.json",
        {
            "source_manifest_sha256": sha256(args.manifest),
            "layout": "CHW RGB float32 raw 0..255; mapper alone applies 1/255",
            "samples": receipt,
        },
    )


def audit(out, name):
    """Read actual placement/precision and input range from compiler artifacts."""
    import onnx

    log = re.sub(r"\x1b\[[0-9;]*m", "", (out / "hb_mapper_makertbin.log").read_text())
    start = log.rfind("The main quantized node information:")
    end = log.find("The quantized model output:", start)
    if start < 0 or end < 0:
        raise ValueError("Compiler node table not found; precision cannot be certified")
    table = log[start:end]
    nodes = []
    for line in table.splitlines():
        match = re.match(r"^(\S+)\s+(BPU|CPU)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
        if match:
            nodes.append(
                dict(zip(["name", "placement", "subgraph", "op", "cosine", "threshold", "dtype"], match.groups()))
            )
    eligible = [line for line in table.splitlines() if re.search(r"\s(?:BPU|CPU)\s", line)]
    if not nodes or len(nodes) != len(eligible):
        raise ValueError("Incomplete compiler node audit")
    graph = onnx.load(str(out / (name + "_original_float_model.onnx")))
    ranges = [
        onnx.helper.get_attribute_value(a)
        for n in graph.graph.node
        if n.op_type == "HzPreprocess"
        for a in n.attribute
        if a.name == "input_type"
    ]
    if ranges != [b"YUV_BT601_VIDEO_RANGE_128"]:
        raise ValueError(f"Unexpected mapper input range: {ranges}")
    binary = out / (name + ".bin")
    with (out / "hb_model_info.txt").open("w") as stream:
        subprocess.run(["hb_model_info", str(binary)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    info = (out / "hb_model_info.txt").read_text()
    if "bt601_video" not in info or "InputDataType.NV12" not in info:
        raise ValueError("BIN input contract is not limited-range BT601 NV12")
    return dict(
        node_count=len(nodes),
        nodes=nodes,
        placement_counts=dict(collections.Counter(n["placement"] for n in nodes)),
        dtype_counts=dict(collections.Counter(n["dtype"] for n in nodes)),
        exceptions=[n for n in nodes if n["placement"] != "BPU" or n["dtype"] != "int8"],
        runtime_input="NV12 BT601 video range",
        precision_scope="internal compute; public outputs may be FLOAT32",
    )


def compile_model(args):
    import yaml

    model, cal, out = Path(args.onnx).resolve(), Path(args.calibration).resolve(), Path(args.out).resolve()
    if not model.is_file() or not cal.is_dir():
        raise ValueError("ONNX file and calibration/rgb directory must exist inside the compiler environment")
    export_report = json.loads((model.parent / "export.json").read_text())
    if not export_report["passed"] or export_report["onnx_sha256"] != sha256(model):
        raise ValueError("ONNX must match a passed export parity report")
    files = sorted(cal.iterdir())
    if not files or any(p.suffix != ".rgb" or p.stat().st_size != 3 * 352 * 640 * 4 for p in files):
        raise ValueError("Calibration directory must contain only CHW float32 352x640 .rgb files")
    receipt_path = cal.parent / "manifest.json"
    receipt = json.loads(receipt_path.read_text())["samples"]
    if len(receipt) != len(files) or any(r["split"] != "train" for r in receipt):
        raise ValueError("Calibration receipt must contain exactly the train-only input files")
    if len({r["id"] for r in receipt}) != len(receipt):
        raise ValueError("Repeated calibration sample ids")
    if any(sha256(path) != row["rgb_sha256"] for path, row in zip(files, receipt)):
        raise ValueError("Calibration data changed after preparation")
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Model prefix must contain only letters, digits, underscores or hyphens")
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "model_parameters": {
            "onnx_model": str(model),
            "march": "bayes-e",
            "layer_out_dump": False,
            "working_dir": str(out),
            "output_model_file_prefix": args.name,
        },
        "input_parameters": {
            "input_name": "images",
            "input_type_rt": "nv12",
            "input_space_and_range": "bt601_video",
            "input_type_train": "rgb",
            "input_layout_train": "NCHW",
            "norm_type": "data_scale",
            "scale_value": 1 / 255,
        },
        "calibration_parameters": {
            "cal_data_dir": str(cal),
            "cal_data_type": "float32",
            "calibration_type": "default",
            "optimization": "set_all_nodes_int8",
        },
        "compiler_parameters": {"jobs": args.jobs, "compile_mode": "latency", "debug": True, "optimize_level": "O3"},
    }
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(
        out / "inputs.json",
        {
            "onnx_sha256": sha256(model),
            "calibration_files": len(files),
            "calibration_manifest_sha256": sha256(receipt_path),
            "calibration_sha256": {p.name: sha256(p) for p in files},
            "requested_precision": "all INT8",
            "actual_precision": "not audited; inspect compiler node report",
            "executed": False,
        },
    )
    if args.run:
        for step, command in [
            ("checker", ["hb_mapper", "checker", "--model-type", "onnx", "--model", str(model), "--march", "bayes-e"]),
            ("makertbin", ["hb_mapper", "makertbin", "--config", str(path), "--model-type", "onnx"]),
        ]:
            with (out / f"{step}.log").open("w") as log:
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = out / f"{args.name}.bin"
        write_json(
            out / "compiled.json",
            {
                "bin_sha256": sha256(binary),
                "onnx_sha256": sha256(model),
                "config_sha256": sha256(path),
                "precision_audit": audit(out, args.name),
                "runtime_validation": "pending HB_ONNXRuntime and X5 replay",
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--count", type=int, default=512)
    c.set_defaults(func=calibrate)
    c = sub.add_parser("compile")
    c.add_argument("--onnx", required=True)
    c.add_argument("--calibration", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--jobs", type=int, default=4)
    c.add_argument("--run", action="store_true", help="Run SDK commands; omitted means generate configuration only")
    c.set_defaults(func=compile_model)
    args = p.parse_args()
    args.func(args)
