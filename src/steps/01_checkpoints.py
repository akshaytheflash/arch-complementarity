"""Step 1 — checkpoint verification.

Every model must reproduce its published ImageNet-1K top-1 within 1.0 pp on a
stratified 2,000-image validation subset. A model that fails is excluded from all
downstream analyses (recorded in `checkpoints.json` with `usable: false`).

This is the control that protects every later claim: if a VMamba checkpoint does not
reproduce 82.5 %, then any "architecture difference" we measure could just be a broken
weight load, and the honest response is to say so rather than to report the comparison.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import datasets as D
import models as M
import nb

TOPK = 2000          # stratified subset size for the gate
TOLERANCE = 1.0      # percentage points


def main(out_dir: str) -> dict:
    results = {}
    torch.backends.cudnn.benchmark = True

    val_root = D.find_wnid_root(D.kaggle_download("tusonggao/imagenet-validation-dataset"))
    print("val root:", val_root)

    for name, spec in M.REGISTRY.items():
        t0 = time.time()
        try:
            tf, cfg = M.transform_for(spec)
            ds = D.WnidFolder(val_root, tf, limit=TOPK, seed=0)
            loader = DataLoader(ds, batch_size=64, num_workers=2, pin_memory=True)
            mdl = M.load(spec)
            r = M.infer(mdl, loader)
            y = np.array([ds.items[i][1] for i in range(len(ds))])
            pred = r["logits"].argmax(1)
            top1 = float((pred == y).mean() * 100)
            top5 = float(np.mean([y[i] in np.argsort(-r["logits"][i])[:5]
                                  for i in range(len(y))]) * 100)
            rec.update({"measured_top1": top1, "measured_top5": top5,
                        "n": int(len(ds)), "delta": top1 - spec.published_top1,
                        "usable": abs(top1 - spec.published_top1) <= TOLERANCE,
                        "preprocessing": {k: str(v) for k, v in cfg.items()},
                        "seconds": round(time.time() - t0, 1)})
            print(f"{name:14s} top1={top1:6.2f} (published {spec.published_top1:5.1f}, "
                  f"delta {rec['delta']:+.2f})  usable={rec['usable']}", flush=True)
            del mdl
            torch.cuda.empty_cache()
        except Exception as e:
            rec.update({"error": f"{type(e).__name__}: {e}", "usable": False,
                        "seconds": round(time.time() - t0, 1)})
            print(f"{name:14s} FAILED {e}", flush=True)
        results[name] = rec
        nb.save("01_checkpoints", "checkpoints.json",
                {"any_top1": {"runs": 1}, "models": results, "subset": TOPK,
                 "tolerance_pp": TOLERANCE})

    usable = [k for k, v in results.items() if v.get("usable")]
    print("USABLE MODELS:", usable)
    return {"usable": usable, "all": {k: v.get("measured_top1") for k, v in results.items()}}
