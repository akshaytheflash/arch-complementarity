"""Step 02 — in-distribution inference (the foundation of every later table).

Writes, for every usable model, `logits.npz` containing:
    logits : (N, 1000) float32
    feat   : (N, C)    float32   (penultimate representation, for the probes)
    y      : (N,)      int64     ground-truth ImageNet class index
    paths  : (N,)      unicode   (for pairing across steps)

N is a stratified 10,000-image subset (50 per class x 200 classes -> capped by class
size), the same subset for every model, so all pairwise comparisons are on identical
images. Reproducible via SEED.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets as D
import models as M
import nb

N = 10_000
SEED = 0


def main(out_dir: str) -> dict:
    torch.backends.cudnn.benchmark = True
    val_root = D.find_wnid_root(D.kaggle_download("tusonggao/imagenet-validation-dataset"))

    ck = json.load(open(os.path.join(out_dir, "01_checkpoints", "checkpoints.json")))
    usable = [k for k, v in ck["models"].items() if v.get("usable")]
    print("usable models:", usable)
    if len(usable) < 2:
        raise RuntimeError("fewer than 2 usable models; fix checkpoints before continuing")

    meta = {}
    for name in usable:
        spec = M.REGISTRY[name]
        tf, _ = M.transform_for(spec)
        ds = D.WnidFolder(val_root, tf, limit=N, seed=SEED)
        loader = DataLoader(ds, batch_size=64, num_workers=2, pin_memory=True,
                            persistent_workers=True)
        t0 = time.time()
        mdl = M.load(spec)
        r = M.infer(mdl, loader)
        y = np.array([ds.items[i][1] for i in range(len(ds))], dtype=np.int64)
        acc = float((r["logits"].argmax(1) == y).mean() * 100)
        np.savez_compressed(os.path.join(out_dir, "02_inference", f"{name}.npz"),
                            logits=r["logits"], feat=r["feat"], y=y,
                            paths=np.array([p for p, _ in ds.items]))
        meta[name] = {"n": int(len(y)), "top1": acc,
                      "seconds": round(time.time() - t0, 1)}
        print(f"{name:14s} n={len(y)} top1={acc:6.2f}  ({meta[name]['seconds']}s)", flush=True)
        del mdl
        torch.cuda.empty_cache()

    nb.save("02_inference", "meta.json", meta)
    return meta
