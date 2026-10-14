# SPDX-License-Identifier: AGPL-3.0-only
"""Convert native annotations to the fixed binary training contract."""

import argparse
import json
from pathlib import Path

from common import HEIGHT, WIDTH, read_manifest, rgb_input, sha256, write_json

# isort: split


def prepare(manifest, out):
    import cv2
    import numpy as np

    rows = read_manifest(manifest)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "masks").mkdir()
    result, pixels, groups = [], {}, {}
    for i, row in enumerate(rows):
        rgb = rgb_input(row["image"], normalize=False).transpose(1, 2, 0).astype(np.uint8)
        import hashlib

        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        if pixel_hash in pixels and pixels[pixel_hash] != row["split"]:
            raise ValueError(f"Train/val image leakage: {row['id']}")
        pixels[pixel_hash] = row["split"]
        if row.get("group"):
            if row["group"] in groups and groups[row["group"]] != row["split"]:
                raise ValueError(f"Capture group crosses train/val: {row['group']}")
            groups[row["group"]] = row["split"]
        if "mask" in row:
            mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
            native = cv2.imread(row["image"])
            if mask is None or mask.shape != native.shape[:2] or not np.isin(mask, [0, 1, 255]).all():
                raise ValueError(f"Mask must match native image and contain 0/1/255: {row['id']}")
            mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
            source = row["mask"]
        else:
            # Legacy normalized polygons: 0=drivable, 1=stairs/non-drivable; exclusion wins overlaps.
            polygons = {0: [], 1: []}
            for line in Path(row["label"]).read_text().splitlines():
                values = np.asarray([float(v) for v in line.split()])
                if values.size < 7 or values.size % 2 != 1 or values[0] not in (0, 1):
                    raise ValueError(f"Invalid polygon: {row['id']}")
                xy = values[1:].reshape(-1, 2)
                if not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
                    raise ValueError(f"Polygon outside normalized coordinates: {row['id']}")
                points = np.floor(xy * [WIDTH, HEIGHT] + 0.5).astype(np.int32)
                points = np.clip(points, [0, 0], [WIDTH - 1, HEIGHT - 1])
                polygons[int(values[0])].append(points)
            mask = np.zeros((HEIGHT, WIDTH), np.uint8)
            for cls in (0, 1):
                for points in polygons[cls]:
                    cv2.fillPoly(mask, [points], 1 - cls, lineType=cv2.LINE_8)
            source = row["label"]
        image_name, mask_name = f"images/{i:07}.png", f"masks/{i:07}.png"
        if not cv2.imwrite(str(out / image_name), rgb[:, :, ::-1]) or not cv2.imwrite(str(out / mask_name), mask):
            raise OSError("Failed to write prepared sample")
        result.append(
            {
                "id": row["id"],
                "split": row["split"],
                "domain": row.get("domain", "default/view"),
                "group": row.get("group"),
                "image": image_name,
                "mask": mask_name,
                "image_sha256": sha256(out / image_name),
                "mask_sha256": sha256(out / mask_name),
                "source_image_sha256": sha256(row["image"]),
                "source_label_sha256": sha256(source),
            }
        )
    if {r["split"] for r in result} != {"train", "val"}:
        raise ValueError("Both train and val are required; split by capture sequence before preparation")
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result))
    write_json(
        out / "dataset.json",
        {
            "source_manifest_sha256": sha256(manifest),
            "input": [3, HEIGHT, WIDTH],
            "classes": ["non_drivable", "drivable"],
            "ignore": 255,
            "counts": {s: sum(r["split"] == s for r in result) for s in ("train", "val")},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    prepare(a.manifest, a.out)
