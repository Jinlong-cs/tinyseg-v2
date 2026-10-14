# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed-shape binary semantic recipe on the official model; full epoch-boundary resume."""

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

from common import HEIGHT, UPSTREAM, WIDTH, check_upstream, read_manifest, sha256, write_json

# isort: split

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics.nn.tasks import SemanticSegmentationModel

DEFAULTS = dict(
    seed=42,
    batch=32,
    epochs=200,
    min_epochs=100,
    patience=40,
    min_delta=0.0005,
    lr=0.0003,
    weight_decay=0.0001,
    warmup_epochs=5,
    final_lr_ratio=0.01,
    workers=4,
    amp=True,
    save_period=5,
    balance_products=True,
)


class Samples(Dataset):
    def __init__(self, rows, seed=42, epoch=0):
        self.rows, self.seed, self.epoch = rows, seed, epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import hashlib

        row = self.rows[index]
        image = cv2.imread(row["image"])
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != (HEIGHT, WIDTH, 3) or mask.shape != (HEIGHT, WIDTH):
            raise ValueError(f"Run prepare_data.py first: {row['id']}")
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError(f"Invalid mask classes: {row['id']}")
        x = image[:, :, ::-1].astype(np.float32) / 255
        if self.epoch:
            digest = hashlib.sha256(f"{self.seed}/{self.epoch}/{row['id']}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            if rng.random() < 0.5:
                x, mask = x[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.8:
                x = np.clip(rng.uniform(0.9, 1.1) * (x - 0.5) + 0.5 + rng.uniform(-0.05, 0.05), 0, 1)
        return (
            torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            torch.from_numpy(mask.astype(np.int64)),
            index,
        )


def loss_fn(outputs, gt):
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    valid = gt != 255
    if not valid.any():
        return None
    resized = [F.interpolate(x.float(), (HEIGHT, WIDTH), mode="bilinear", align_corners=False) for x in outputs]
    ce = [F.cross_entropy(x, gt, ignore_index=255, reduction="sum") / valid.sum() for x in resized]
    p, y = torch.softmax(resized[0], 1)[:, 1] * valid, (gt == 1).float() * valid
    dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)
    main = ce[0] + 0.5 * dice[valid.sum((1, 2)) > 0].mean()
    return main + 0.4 / (len(ce) - 1) * sum(ce[1:]) if len(ce) > 1 else main


def pretrained(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    base = (ck.get("ema") or ck["model"]).float()
    if not isinstance(base, SemanticSegmentationModel):
        raise ValueError("Expected official YOLO26n-sem pretrained checkpoint")
    model = SemanticSegmentationModel(deepcopy(base.yaml), nc=2, verbose=False)
    dst = model.state_dict()
    matching = {k: v for k, v in base.state_dict().items() if k in dst and v.shape == dst[k].shape}
    missing = model.load_state_dict(matching, strict=False).missing_keys
    if any(".classifier.1." not in k and ".aux_head.1." not in k for k in missing):
        raise ValueError(f"Unexpected pretrained mismatch: {missing}")
    model.names = {0: "non_drivable", 1: "drivable"}
    return model


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SemanticSegmentationModel(ck["yaml"], nc=2, verbose=False)
    model.load_state_dict(ck["model"], strict=True)
    model.names = {0: "non_drivable", 1: "drivable"}
    model.model[-1].export = False
    return model.eval()


def order_for(rows, epoch, cfg):
    rng = np.random.default_rng(cfg["seed"] + epoch)
    if not cfg["balance_products"]:
        return rng.permutation(len(rows)).tolist()
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["domain"].split("/")[0], []).append(i)
    count = max(map(len, groups.values()))
    chunks = [
        np.concatenate([rng.permutation(ids) for _ in range(math.ceil(count / len(ids)))])[:count]
        for _, ids in sorted(groups.items())
    ]
    return rng.permutation(np.concatenate(chunks)).tolist()


@torch.inference_mode()
def validate(model, rows, device, batch=32, workers=0):
    model.eval()
    stats = {r["domain"]: np.zeros(4, np.int64) for r in rows}
    for x, y, ids in DataLoader(Samples(rows), batch_size=batch, num_workers=workers):
        logits = model(x.to(device)).float().cpu().numpy()
        for j, i in enumerate(ids.tolist()):
            z = cv2.resize(logits[j].transpose(1, 2, 0), (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            p, g, v = z[:, :, 1] > z[:, :, 0], y[j].numpy() == 1, y[j].numpy() != 255
            stats[rows[i]["domain"]] += [np.sum(p & g & v), np.sum(p & ~g & v), np.sum(~p & g & v), np.sum(~p & ~g & v)]
    domains, products = {}, {}
    for domain, counts in stats.items():
        tp, fp, fn, tn = map(int, counts)
        iou = tp / max(1, tp + fp + fn)
        domains[domain] = dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=iou, false_drivable=fp / max(1, fp + tn))
        products.setdefault(domain.split("/")[0], []).append(iou)
    total = sum(stats.values())
    return dict(
        domains=domains,
        macro_iou=float(np.mean([np.mean(v) for v in products.values()])),
        false_drivable=float(total[1] / max(1, total[1] + total[3])),
    )


def main(args):
    check_upstream()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    cfg = DEFAULTS.copy()
    if args.config:
        supplied = json.loads(Path(args.config).read_text())
        if supplied.keys() - cfg.keys():
            raise ValueError(f"Unknown config keys: {supplied.keys() - cfg.keys()}")
        cfg.update(supplied)
    if not 1 <= cfg["batch"] or not 1 <= cfg["epochs"] or not 0 <= cfg["min_epochs"] <= cfg["epochs"]:
        raise ValueError("Invalid batch/epochs/min_epochs")
    rows = read_manifest(args.manifest)
    tr, va = ([r for r in rows if r["split"] == s] for s in ("train", "val"))
    if not tr or not va:
        raise ValueError("Train and validation splits must both be nonempty")
    cfg.update(manifest_sha256=sha256(args.manifest), pretrained_sha256=sha256(args.pretrained), upstream=UPSTREAM)
    out = Path(args.out).resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError("Resume into the existing run directory")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    model = pretrained(args.pretrained).to(device)
    groups = [[], []]
    for name, parameter in model.named_parameters():
        groups[int(parameter.ndim < 2 or name.endswith(".bias"))].append(parameter)
    opt = torch.optim.AdamW(
        [dict(params=groups[0], weight_decay=cfg["weight_decay"]), dict(params=groups[1], weight_decay=0)],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    steps = math.ceil(len(order_for(tr, 1, cfg)) / cfg["batch"])

    def factor(step):
        warm, total = cfg["warmup_epochs"] * steps, cfg["epochs"] * steps
        if step < warm:
            return 0.01 + 0.99 * step / max(1, warm - 1)
        progress = min(1, (step - warm) / max(1, total - warm - 1))
        return cfg["final_lr_ratio"] + (1 - cfg["final_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    early = dict(significant=-1.0, best=-1.0, fdr=1.0, count=0, epoch=0)
    start, step = 1, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck["config"] != cfg:
            raise ValueError("Resume config, pretrained file or dataset changed")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        start, step, early = ck["epoch"] + 1, ck["global_step"], ck["early"]
    write_json(out / "config.json", cfg)
    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        order = order_for(tr, epoch, cfg)
        loader = DataLoader(
            Samples(tr, cfg["seed"], epoch),
            batch_size=cfg["batch"],
            sampler=order,
            num_workers=cfg["workers"],
            generator=torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch),
        )
        for x, y, _ in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x.to(device)), y.to(device))
            if loss is not None:
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                scaler.step(opt)
                scaler.update()
            sched.step()
            step += 1
        metrics = validate(model, va, device, cfg["batch"], cfg["workers"])
        score, fdr = metrics["macro_iou"], metrics["false_drivable"]
        if score > early["significant"] + cfg["min_delta"]:
            early.update(significant=score, count=0)
        else:
            early["count"] += 1
        best = score > early["best"] or (score == early["best"] and fdr < early["fdr"])
        if best:
            early.update(best=score, fdr=fdr, epoch=epoch)
        ck = dict(
            model=model.state_dict(),
            yaml=model.yaml,
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch,
            global_step=step,
            early=early.copy(),
            config=cfg,
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            ),
            sampler=dict(completed_epoch=epoch, next_epoch=epoch + 1, seed=cfg["seed"]),
            weights_source="live FP32, no EMA",
        )
        targets = ["last.pt"] + (["best.pt"] if best else [])
        if cfg["save_period"] > 0 and epoch % cfg["save_period"] == 0:
            targets.append(f"epoch{epoch:03}.pt")
        for name in targets:
            temporary = out / (name + ".part")
            torch.save(ck, temporary)
            temporary.replace(out / name)
        write_json(out / f"metrics{epoch:03}.json", metrics)
        print(json.dumps(dict(epoch=epoch, **metrics, patience_counter=early["count"])), flush=True)
        if epoch >= cfg["min_epochs"] and early["count"] >= cfg["patience"]:
            break
    write_json(
        out / "result.json",
        dict(
            last_epoch=max(start - 1, epoch if start <= cfg["epochs"] else 0),
            best=early,
            validation="OpenCV bilinear both logits, strict foreground > background",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="JSON overriding the documented recipe")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", help="Full checkpoint; retain the same config and run directory")
    main(p.parse_args())
