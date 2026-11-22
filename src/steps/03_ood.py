"""Step 03 — out-of-distribution inference on ImageNet-A / -R / -Sketch / -V2.

Same protocol as step 02 (stratified subset, same preprocessing, same storage format) so
that OOD numbers are directly comparable to the in-distribution numbers and to each other.
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

SETS = {
    "A":      ("paultimothymooney/natural-adversarial-examples-imageneta", None),
    "R":      ("mauricioalvarezj/imagenet-r", None),
    "sketch": ("wanghaohan/imagenetsketch", None),
}
N_PER_SET = 2_000
SEED = 0


def main(out_dir: str) -> dict:
    torch.backends.cudnn.benchmark = True
    ck = json.load(open(os.path.join(out_dir, "01_checkpoints", "checkpoints.json")))
    usable = [k for k, v in ck["models"].items() if v.get("usable")]
    print("usable models:", usable)

    meta = {}
    for set_name, (ref, _) in SETS.items():
        try:
            root = D.find_wnid_root(D.kaggle_download(ref))
        except Exception as e:
            print(f"{set_name}: download failed {e}")
            meta[set_name] = {"error": str(e)}
            continue
        if root is None:
            meta[set_name] = {"error": "no wnid root"}
            continue
        for name in usable:
            spec = M.REGISTRY[name]
            tf, _ = M.transform_for(spec)
            ds = D.WnidFolder(root, tf, limit=N_PER_SET, seed=SEED)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=64, num_workers=2, pin_memory=True)
            t0 = time.time()
            mdl = M.load(spec)
            r = M.infer(mdl, loader)
            y = np.array([ds.items[i][1] for i in range(len(ds))], dtype=np.int64)
            acc = float((r["logits"].argmax(1) == y).mean() * 100)
            os.makedirs(os.path.join(out_dir, "03_ood"), exist_ok=True)
            np.savez_compressed(os.path.join(out_dir, "03_ood", f"{set_name}__{name}.npz"),
                                logits=r["logits"], feat=r["feat"], y=y)
            meta.setdefault(set_name, {})[name] = {"n": int(len(y)), "top1": acc,
                                                   "seconds": round(time.time() - t0, 1)}
            print(f"{set_name:6s} {name:14s} n={len(y):5d} top1={acc:6.2f}", flush=True)
            del mdl
            torch.cuda.empty_cache()

    nb.save("03_ood", "meta.json", meta)
    return meta
