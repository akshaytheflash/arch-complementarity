"""Step 05 — explanation divergence (occlusion sensitivity).

For a 1,000-image stratified subset, occlude a 7x7 grid of 32x32 grey patches and record
the drop in the ground-truth logit. The resulting 7x7 map is a faithful, architecture
agnostic explanation (it needs no gradient, no method-specific assumptions).

Step 08 measures:
  * per-image Pearson / SSIM between the ViT and the VMamba map, split by whether both
    models were correct or they disagreed;
  * inner-50% vs border-ring saliency mass (background dependency);
  * whether the two architectures' maps are more similar on correct-both than on
    disagree examples.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import datasets as D
import models as M
import nb

N = 1_000
GRID = 7
PATCH = 32
SEED = 0


def occlusion_maps(model, imgs, y, device="cuda", grid=GRID, patch=PATCH):
    """imgs: (B,3,224,224) tensor. Returns (B, grid, grid) map of -Δlogit(y)."""
    B = imgs.shape[0]
    base = model(imgs)["logits"]
    base_logit_y = base.gather(1, y.view(-1, 1)).squeeze(1)
    maps = np.zeros((B, grid, grid), dtype=np.float32)
    for gi in range(grid):
        for gj in range(grid):
            occ = imgs.clone()
            y0 = int(gi * 224 / grid)
            x0 = int(gj * 224 / grid)
            y1 = min(224, y0 + patch)
            x1 = min(224, x0 + patch)
            occ[:, :, y0:y1, x0:x1] = 0.0
            with torch.no_grad():
                lg = model(occ)["logits"].gather(1, y.view(-1, 1)).squeeze(1)
            maps[:, gi, gj] = (base_logit_y - lg).float().cpu().numpy()
    return maps


def main(out_dir: str) -> dict:
    val_root = D.find_wnid_root(D.kaggle_download("tusonggao/imagenet-validation-dataset"))
    ck = json.load(open(os.path.join(out_dir, "01_checkpoints", "checkpoints.json")))
    usable = [k for k, v in ck["models"].items() if v.get("usable")]
    models = [m for m in ("deit-small", "vmamba-tiny", "vit-small", "vmamba-small") if m in usable]
    print("explanation models:", models)
    if len(models) < 2:
        raise RuntimeError("need at least 2 usable models for explanation analysis")

    paths_labels = None
    results = {}
    for name in models:
        spec = M.REGISTRY[name]
        tf, _ = M.transform_for(spec)
        base = D.WnidFolder(val_root, tf, limit=N, seed=SEED)
        if paths_labels is None:
            paths_labels = base.items
        else:
            assert paths_labels == base.items, "subset mismatch between models"
        loader = DataLoader(base, batch_size=32, num_workers=2, pin_memory=True)
        mdl = M.load(spec)
        t0 = time.time()
        all_maps, all_y, all_correct = [], [], []
        for xb, yb in loader:
            xb = xb.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)
            m = occlusion_maps(mdl, xb, yb)
            with torch.no_grad():
                pred = mdl(xb)["logits"].argmax(1)
            all_maps.append(m)
            all_y.append(yb.cpu().numpy())
            all_correct.append((pred == yb).float().cpu().numpy())
        maps = np.concatenate(all_maps)
        y = np.concatenate(all_y)
        correct = np.concatenate(all_correct)
        np.savez_compressed(os.path.join(out_dir, "05_explanations", f"{name}.npz"),
                            maps=maps, y=y, correct=correct)
        results[name] = {"n": int(len(y)), "correct_frac": float(correct.mean()),
                         "seconds": round(time.time() - t0, 1)}
        print(f"{name:14s} n={len(y)} correct={correct.mean():.3f} "
              f"({results[name]['seconds']}s)", flush=True)
        del mdl
        torch.cuda.empty_cache()

    nb.save("05_explanations", "meta.json", results)
    return results
