"""Model registry & inference for the ViT vs VMamba complementarity study.

Uniform interface:
    wrapper(x: (B,3,H,W) tensor) -> {"feat": (B,C), "logits": (B,1000)}

Families:
    vit    - timm pretrained (ImageNet-1K or 21K->1K)
    ssm    - VMamba (VSSM) official checkpoints, run through the official code
    hybrid - (reserved)

Preprocessing comes from `timm.data.resolve_model_data_config` for every model, so ViT
and VMamba see the *same* augmentation-free centre-crop pipeline. This is the control
that keeps the comparison honest: any accuracy gap is architecture, not preprocessing.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import timm


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str            # "vit" | "ssm"
    provider: str          # "timm" | "vmamba"
    ckpt: str              # timm tag or VMamba factory tag
    params_m: float
    published_top1: float


REGISTRY: dict[str, ModelSpec] = {
    "deit-tiny": ModelSpec("deit-tiny", "vit", "timm", "deit_tiny_patch16_224.fb_in1k", 5.7, 72.2),
    "deit-small": ModelSpec("deit-small", "vit", "timm", "deit_small_patch16_224.fb_in1k", 22.1, 79.8),
    "vit-small": ModelSpec("vit-small", "vit", "timm", "vit_small_patch16_224.augreg_in1k", 22.1, 81.3),
    "vit-base": ModelSpec("vit-base", "vit", "timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k", 86.6, 85.4),
    "swin-tiny": ModelSpec("swin-tiny", "vit", "timm", "swin_tiny_patch4_window7_224.ms_in1k", 28.3, 81.3),
    "vmamba-tiny": ModelSpec("vmamba-tiny", "ssm", "vmamba", "vssm_tiny", 30.0, 82.6),
    "vmamba-small": ModelSpec("vmamba-small", "ssm", "vmamba", "vssm_small", 50.0, 83.6),
}


def transform_for(spec: ModelSpec):
    """timms's own preprocessing for any tag, defaults for VMamba which is a Swin recipe."""
    try:
        m = timm.create_model(spec.ckpt, pretrained=False) if spec.provider == "timm" else None
    except Exception:
        m = None
    if m is not None:
        cfg = timm.data.resolve_model_data_config(m)
        return timm.data.create_transform(**cfg, is_training=False), cfg
    # VMamba checkpoints were trained with the Swin recipe (ImageNet stats, 224 crop)
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    cfg = {"input_size": (3, 224, 224), "mean": IMAGENET_DEFAULT_MEAN,
           "std": IMAGENET_DEFAULT_STD, "crop_pct": 1.0, "interpolation": "bicubic"}
    tf = timm.data.create_transform(**{"input_size": cfg["input_size"], "mean": cfg["mean"],
                                       "std": cfg["std"], "crop_pct": cfg["crop_pct"],
                                       "interpolation": cfg["interpolation"]},
                                    is_training=False)
    return tf, cfg


class TimmWrapper(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.net = timm.create_model(spec.ckpt, pretrained=True)
        self.net.eval()

    @torch.no_grad()
    def forward(self, x):
        f = self.net.forward_features(x)
        if f.ndim == 4:
            f = f.mean(dim=(2, 3))
        elif f.ndim == 3:
            f = f.mean(dim=1)             # ViT: patch tokens (pos 0 already dropped by timm)
        else:
            f = f.flatten(1)
        return {"feat": f, "logits": self.net.forward_head(f)}


def vmamba_ckpt(spec: ModelSpec, cache="/content/ckpts") -> Optional[str]:
    os.makedirs(cache, exist_ok=True)
    hits = sorted(glob.glob(os.path.join(cache, spec.ckpt.replace("-", "_") + "*.pth")) +
                  glob.glob(os.path.join(cache, spec.ckpt + "*.pth")))
    return hits[0] if hits else None


def load(spec_or_name, device="cuda") -> nn.Module:
    spec = REGISTRY.get(spec_or_name, spec_or_name)
    if spec.provider == "timm":
        mdl = TimmWrapper(spec)
    else:
        from vmamba_pt import VMambaClassifier
        ck = vmamba_ckpt(spec)
        if ck is None:
            raise FileNotFoundError(
                f"no VMamba checkpoint for {spec.ckpt} in /content/ckpts "
                f"(expected files like {spec.ckpt.replace('-', '_')}_*.pth)")
        mdl = VMambaClassifier(spec.ckpt, ck)
    return mdl.to(device).eval()


@torch.no_grad()
def infer(model: nn.Module, loader, device="cuda") -> dict:
    feats, logits = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                out = model(xb)
            feats.append(out["feat"].float().cpu().numpy())
            logits.append(out["logits"].float().cpu().numpy())
    return {"feat": np.concatenate(feats).astype(np.float32),
            "logits": np.concatenate(logits).astype(np.float32)}
