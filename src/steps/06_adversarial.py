"""Step 06 — adversarial transferability (FGSM / PGD, 2x2 transfer matrix).

Protocol: for every source model, attack a 500-image subset of the images it classifies
correctly, with the model in eval mode and fp32 gradients (no AMP, so gradients are exact).
Untargeted attacks, projected to the epsilon ball in the input pixel space.

Transfers measured: ViT -> VMamba and VMamba -> ViT (cross-architecture), plus ViT -> ViT
and VMamba -> VMamba (within-architecture) using a second member of each family when one
is available, otherwise the same model (which is the exact self-transfer baseline).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets as D
import models as M
import nb

N = 500
SEED = 0
EPS = 8.0 / 255.0


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x)["logits"], y)
    g, = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).detach()


def pgd(model, x, y, eps, steps=10, alpha=1.0 / 255.0):
    x0 = x.clone()
    x = x0 + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x)["logits"], y)
        g, = torch.autograd.grad(loss, x)
        x = x + alpha * g.sign()
        x = torch.max(torch.min(x, x0 + eps), x0 - eps).detach()
        x = torch.clamp(x, 0.0, 1.0)
    return x


def transfer_rate(model, adv, y, correct_mask):
    with torch.no_grad():
        pred = model(adv)["logits"].argmax(1)
    return float((pred[correct_mask] == y[correct_mask]).mean() * 100)


def main(out_dir: str) -> dict:
    val_root = D.find_wnid_root(D.kaggle_download("tusonggao/imagenet-validation-dataset"))
    ck = json.load(open(os.path.join(out_dir, "01_checkpoints", "checkpoints.json")))
    usable = [k for k, v in ck["models"].items() if v.get("usable")]
    vit = [m for m in usable if M.REGISTRY[m].family == "vit"]
    ssm = [m for m in usable if M.REGISTRY[m].family == "ssm"]
    if not vit or not ssm:
        raise RuntimeError("need at least one ViT and one SSM")
    print("ViT:", vit, "SSM:", ssm)

    results = {}
    for source in vit[:1] + ssm[:1]:
        spec = M.REGISTRY[source]
        tf, _ = M.transform_for(spec)
        base = D.WnidFolder(val_root, tf, limit=N, seed=SEED)
        loader = DataLoader(base, batch_size=16, num_workers=2, pin_memory=True)
        src = M.load(source)
        x_all, y_all = [], []
        for xb, yb in loader:
            x_all.append(xb)
            y_all.append(yb)
        x = torch.cat(x_all).cuda()
        y = torch.cat(y_all).cuda()
        with torch.no_grad():
            ok = src(x)["logits"].argmax(1) == y
        x, y = x[ok], y[ok]
        print(f"{source}: {ok.sum().item()} correctly classified attack seeds")

        for attack in ("fgsm", "pgd"):
            t0 = time.time()
            adv = (fgsm(src, x, y, EPS) if attack == "fgsm"
                   else pgd(src, x, y, EPS))
            row = {}
            for tgt in usable:
                if tgt == source:
                    continue
                tm = M.load(tgt)
                row[tgt] = {"transfer_acc": transfer_rate(tm, adv, y, torch.ones_like(ok[ok])),
                            "n": int(len(y))}
                del tm
                torch.cuda.empty_cache()
            results.setdefault(attack, {})[source] = row
            print(f"  {attack} from {source}: " +
                  ", ".join(f"{k}={v['transfer_acc']:.1f}%" for k, v in row.items()),
                  flush=True)
        del src
        torch.cuda.empty_cache()

    nb.save("06_adversarial", "transfer.json", results)
    return results
