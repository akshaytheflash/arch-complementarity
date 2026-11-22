"""Model registry & inference for the ViT vs VMamba complementarity study.

Uniform interface (every wrapper):
    mdl(x: (B,3,H,W) float tensor) -> {"feat": (B,C) float32, "logits": (B,1000) float32}

Design constraints (see docs/research-design.md):
  * Only public ImageNet-1K pretrained weights, no fine-tuning.
  * Preprocessing is taken from `timm.data.resolve_model_data_config` for every model, so
    ViT and VMamba receive the identical resize/crop/normalise pipeline. Any accuracy gap
    is therefore attributable to architecture, not to preprocessing.
  * Two architectures at matched parameter budget:
        vmamba-tiny   (31M, 82.5-82.6 top-1)  <->  deit-small   (22M, 79.8)  *and* vit-small (22M, 81.3)
        vmamba-small  (50M, 83.6 top-1)       <->  vit-small    (22M) *and* swin-tiny (28M)
    so that size is not confounded with family.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import timm


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str          # "vit" | "ssm"
    provider: str        # "timm" | "vmamba"
    ckpt: str
    params_m: float
    published_top1: float


REGISTRY: dict[str, ModelSpec] = {
    # ---- ViT family ----------------------------------------------------------------
    "deit-small": ModelSpec("deit-small", "vit", "timm",
                            "deit_small_patch16_224.fb_in1k", 22.1, 79.8),
    "vit-small": ModelSpec("vit-small", "vit", "timm",
                           "vit_small_patch16_224.augreg_in1k", 22.1, 81.3),
    "swin-tiny": ModelSpec("swin-tiny", "vit", "timm",
                           "swin_tiny_patch4_window7_224.ms_in1k", 28.3, 81.3),
    "vit-base": ModelSpec("vit-base", "vit", "timm",
                          "vit_base_patch16_224.augreg2_in21k_ft_in1k", 86.6, 85.4),
    # ---- SSM family (VMamba v2, #v2cls release) ---------------------------------
    "vmamba-tiny": ModelSpec("vmamba-tiny", "ssm", "vmamba", "vssm_tiny_0230", 31.0, 82.5),
    "vmamba-small": ModelSpec("vmamba-small", "ssm", "vmamba", "vssm_small_0229", 50.0, 83.6),
}

# Checkpoint URLs (verified via the GitHub releases API, Nov 2026).
VMAMBA_URLS = {
    "vssm_tiny_0230": "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_tiny_0230_ckpt_epoch_262.pth",
    "vssm_small_0229": "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth",
}

# Architecture configs for the three standard VMamba sizes (classification/models/vmamba.py)
VMAMBA_CFG = {
    "vssm_tiny_0230": dict(depths=[2, 2, 9, 2], dims=96),
    "vssm_small_0229": dict(depths=[2, 2, 27, 2], dims=96),
}

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def transform_for(spec: ModelSpec, image_size: int = 224):
    """Identical preprocessing for every model (ImageNet defaults, bicubic, centre crop)."""
    from timm.data import create_transform, resolve_data_config
    if spec.provider == "timm":
        cfg = resolve_data_config({}, model=timm.create_model(spec.ckpt, pretrained=False))
    else:
        cfg = {"input_size": (3, image_size, image_size), "mean": IMAGENET_MEAN,
               "std": IMAGENET_STD, "crop_pct": 1.0, "interpolation": "bicubic"}
    tf = create_transform(**{**cfg, "is_training": False})
    return tf, cfg


# --------------------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------------------
class TimmWrapper(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.net = timm.create_model(spec.ckpt, pretrained=True)
        self.net.eval()

    @torch.no_grad()
    def forward(self, x):
        f = self.net.forward_features(x)
        if f.ndim == 4:                       # BCHW (swin / conv stems)
            f = f.mean(dim=(2, 3))
        elif f.ndim == 3:                     # BNC -> mean over patch tokens
            f = f.mean(dim=1)
        else:
            f = f.flatten(1)
        return {"feat": f.float(), "logits": self.net.forward_head(f).float()}


def download_vmamba(spec: ModelSpec, cache: str = "/content/ckpts") -> str:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, os.path.basename(VMAMBA_URLS[spec.ckpt]))
    if os.path.exists(path) and os.path.getsize(path) > 50_000_000:
        return path
    url = VMAMBA_URLS[spec.ckpt]
    torch.hub.download_url_to_file(url, path, progress=True)
    return path


class VMambaWrapper(nn.Module):
    """Official VSSM (upstream code + official weights), pure-pytorch selective scan."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        from vmamba_pt import build_vssm
        self.spec = spec
        self.net = build_vssm(spec.ckpt, VMAMBA_CFG[spec.ckpt], download_vmamba(spec))
        self.net.eval()

    @torch.no_grad()
    def forward(self, x):
        f = self.net.forward_features(x)
        if f.ndim == 4:
            f = f.mean(dim=(2, 3))
        elif f.ndim == 3:
            f = f.mean(dim=1)
        logits = self.net.head(f)
        return {"feat": f.float(), "logits": logits.float()}


def load(spec_or_name, device="cuda") -> nn.Module:
    spec = spec_or_name if isinstance(spec_or_name, ModelSpec) else REGISTRY[spec_or_name]
    mdl = TimmWrapper(spec) if spec.provider == "timm" else VMambaWrapper(spec)
    return mdl.to(device).eval()


@torch.no_grad()
def infer(model, loader, device="cuda", max_batches=None) -> dict:
    on_cuda = str(device).startswith("cuda")
    feats, logits = [], []
    for i, (xb, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=on_cuda):
            out = model(xb)
        feats.append(out["feat"].float().cpu().numpy())
        logits.append(out["logits"].float().cpu().numpy())
    return {"feat": np.concatenate(feats).astype(np.float32),
            "logits": np.concatenate(logits).astype(np.float32)}
