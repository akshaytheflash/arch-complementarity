"""Step 04 — corruption sensitivity.

15 corruption types x 5 severities, applied to a stratified 2,000-image subset of the
ImageNet validation images, with the *same deterministic seed per image* for every model,
so ViT and VMamba see byte-identical corrupted pixels.

Writes one npz per (corruption, severity, model). Step 08 computes the sensitivity table,
the architecture-specific degradation profiles, and the corruption-rank correlation.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import corruption as C
import datasets as D
import models as M
import nb

N = 2_000
SEED = 0
# Every corruption is expensive; the two headline architectures are always included.
FORCE_MODELS = ("deit-small", "vmamba-tiny")


def main(out_dir: str) -> dict:
    torch.backends.cudnn.benchmark = True
    val_root = D.find_wnid_root(D.kaggle_download("tusonggao/imagenet-validation-dataset"))

    ck = json.load(open(os.path.join(out_dir, "01_checkpoints", "checkpoints.json")))
    usable = [k for k, v in ck["models"].items() if v.get("usable")]
    models = [m for m in FORCE_MODELS if m in usable]
    for m in usable:
        if m not in models and m in ("vit-small", "vmamba-small"):
            models.append(m)
    print("corruption models:", models)

    out_dir = os.path.join(out_dir, "04_corruption")
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    for name in models:
        spec = M.REGISTRY[name]
        tf, _ = M.transform_for(spec)
        base = D.WnidFolder(val_root, tf, limit=N, seed=SEED)
        items = base.items
        for cname in C.CORRUPTIONS:
            for sev in (1, 3, 5):
                t0 = time.time()
                ds = D.CorruptedRaw(items, tf, cname, sev, seed=SEED)
                loader = DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True)
                mdl = M.load(spec)
                r = M.infer(mdl, loader)
                y = np.array([lab for _, lab in items], dtype=np.int64)
                acc = float((r["logits"].argmax(1) == y).mean() * 100)
                np.savez_compressed(os.path.join(out_dir, f"{cname}__s{sev}__{name}.npz"),
                                    logits=r["logits"], y=y)
                meta.setdefault(f"{cname}__s{sev}", {})[name] = {"top1": acc,
                                                                 "n": int(len(y)),
                                                                 "seconds": round(time.time() - t0, 1)}
                print(f"{cname:20s} s{sev} {name:14s} top1={acc:6.2f}", flush=True)
                del mdl
                torch.cuda.empty_cache()
    nb.save("04_corruption", "meta.json", meta)
    return meta
