# SPDX-License-Identifier: AGPL-3.0-only
"""Shared input contract and provenance; no model download on import."""

import hashlib
import json
import os
from pathlib import Path

UPSTREAM = "3d75949c41f78a2c6763eb347ac7a94793098e89"
HEIGHT, WIDTH = 352, 640
WORK = Path(os.environ.get("TINY_WORK_DIR", "work")).resolve()
for key, suffix in {
    "YOLO_CONFIG_DIR": "cache/ultralytics",
    "TORCH_HOME": "cache/torch",
    "MPLCONFIGDIR": "cache/matplotlib",
    "XDG_CACHE_HOME": "cache",
    "TMPDIR": "tmp",
}.items():
    directory = WORK / suffix
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha256(path):
    with Path(path).open("rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_manifest(path):
    """Resolve image/label/mask paths relative to a JSONL manifest."""
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if row["id"] in ids or row["split"] not in {"train", "val"}:
            raise ValueError(f"Duplicate id or invalid split: {row['id']}")
        ids.add(row["id"])
        for key in ("image", "label", "mask"):
            if key in row:
                row[key] = str((path.parent / row[key]).resolve())
                expected = row.get(key + "_sha256")
                if expected and sha256(row[key]) != expected:
                    raise ValueError(f"Prepared {key} changed: {row['id']}")
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def rgb_input(path, normalize=True):
    import cv2
    import numpy as np

    image = cv2.imread(str(path))
    if image is None or image.shape not in {(352, 640, 3), (512, 640, 3)}:
        raise ValueError(f"Expected 640x352 or 640x512 image: {path}")
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    result = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    return result / 255 if normalize else result


def check_upstream():
    import ultralytics

    if ultralytics.__version__ != "8.4.175":
        raise RuntimeError("Install requirements.txt: this adapter requires pinned Ultralytics 8.4.175")
