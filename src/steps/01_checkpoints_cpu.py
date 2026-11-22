"""Step 01-CPU — checkpoint verification on the CPU (no GPU quota needed).

Same gate as `01_checkpoints` but with a smaller stratified subset (600 images) and the
two headline architectures only, so it completes in minutes on a Colab CPU runtime.

    ViT  : deit-small  (22.1M, published 79.8)
    SSM  : vmamba-tiny (31.0M, published 82.5)

The gate is identical in logic: a model that does not reproduce its published top-1
within TOLERANCE percentage points is marked unusable and excluded downstream.
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

TOPK = 600
TOLERANCE = 1.5
MODELS = ("deit-small", "vmamba-tiny")


def main(out_dir: str) -> dict:
    torch.set_num_threads(os.cpu_count() or 2)
    val_root = D.find_wnid_root(D.kaggle_download("tusonggao/imagenet-validation-dataset"))
    print("val root:", val_root)

    results = {}
    for name in MODELS:
        spec = M.REGISTRY[name]
        rec = {"published_top1": spec.published_top1, "params_m": spec.params_m,
               "family": spec.family}
        t0 = time.time()
        try:
            tf, cfg = M.transform_for(spec)
            ds = D.WnidFolder(val_root, tf, limit=TOPK, seed=0)
            loader = DataLoader(ds, batch_size=16, num_workers=2)
            mdl = M.load(spec, device="cpu")
            r = M.infer(mdl, loader, device="cpu")
            y = np.array([ds.items[i][1] for i in range(len(ds))], dtype=np.int64)
            top1 = float((r["logits"].argmax(1) == y).mean() * 100)
            rec.update({"measured_top1": top1, "n": int(len(ds)),
                        "delta": top1 - spec.published_top1,
                        "usable": abs(top1 - spec.published_top1) <= TOLERANCE,
                        "seconds": round(time.time() - t0, 1)})
            np.savez_compressed(os.path.join(out_dir, "01_checkpoints", f"{name}.npz"),
                                logits=r["logits"], feat=r["feat"], y=y,
                                paths=np.array([p for p, _ in ds.items]))
            print(f"{name:14s} top1={top1:6.2f} (published {spec.published_top1}, "
                  f"delta {rec['delta']:+.2f}) usable={rec['usable']} "
                  f"{rec['seconds']}s", flush=True)
            del mdl
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            rec.update({"error": f"{type(e).__name__}: {e}", "usable": False,
                        "traceback": tb[-1500:],
                        "seconds": round(time.time() - t0, 1)})
            print(f"{name:14s} FAILED {type(e).__name__}: {e}", flush=True)
            print(tb[-1200:], flush=True)
        results[name] = rec
        nb.save("01_checkpoints", "checkpoints.json", {"models": results,
                                                       "subset": TOPK,
                                                       "tolerance_pp": TOLERANCE,
                                                       "device": "cpu"})

    usable = [k for k, v in results.items() if v.get("usable")]
    print("USABLE:", usable, flush=True)
    return {"usable": usable,
            "top1": {k: v.get("measured_top1") for k, v in results.items()}}
