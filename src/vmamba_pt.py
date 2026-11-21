"""VMamba (VSSM) classifier wrapper around the *official* upstream implementation.

We deliberately do not re-implement the selective-scan kernel. Upstream ships
`kernels/selective_scan` with a CUDA extension; when that extension is absent upstream
falls back to `selective_scan_ref` in pure PyTorch (a Python loop over L). On a T4 this
is slow but exact, which is what we need: the experiment is inference-only over ~10k
images x 5 datasets, and every checkpoint is used identically across protocols.

The fallback is enabled by setting `selective_scan_cuda` to a shim that forwards to
`selective_scan_ref`. `attach_pure_torch_fallback()` below installs that shim *before*
the upstream modules are imported (the upstream `selective_scan_interface` does
`try: import selective_scan_cuda ... except: pass` and then checks for the module).
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn as nn


ROOT = os.environ.get("VMAMBA_ROOT", "/content/vmamba")


def _ensure_path():
    for p in (f"{ROOT}/classification", ROOT, f"{ROOT}/kernels/selective_scan"):
        if p not in sys.path:
            sys.path.insert(0, p)


def attach_pure_torch_fallback():
    """Install a `selective_scan_cuda` module that defers to the reference scan.

    Upstream calls:
        selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        selective_scan_cuda.fwd(...) with `delta_softplus` positional
        selective_scan_cuda.oflex(...) and .nrow(...) (fast paths, same signature)
    We implement the three used variants by forwarding to `selective_scan_ref`.
    """
    _ensure_path()
    from selective_scan import selective_scan_interface as ssi  # upstream module

    ref = ssi.selective_scan_ref

    shim = types.ModuleType("selective_scan_cuda")

    def fwd(u, delta, A, B, C, D=None, z=None, delta_bias=None,
            delta_softplus=False, return_last_state=False):
        return ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

    shim.fwd = fwd
    shim.nrow = fwd
    shim.ndstate = fwd
    shim.oflex = fwd
    sys.modules["selective_scan_cuda"] = shim
    # Make upstream `SelectiveScanFn` see the attribute set by its module-level import.
    for name in ("selective_scan_cuda", "selective_scan_cuda_core",
                 "selective_scan_cuda_nrow", "selective_scan_cuda_oflex",
                 "selective_scan_cuda_ndstate"):
        sys.modules[name] = shim
        setattr(ssi, name, shim)
    return ssi


def build_vssm(tag: str) -> nn.Module:
    """`tag` matches an upstream factory, e.g. 'vssm_tiny', 'vssm_small', 'vssm_base'."""
    _ensure_path()
    attach_pure_torch_fallback()
    from vmamba import VSSM
    import vmamba as vm

    factory = getattr(vm, tag, None)
    if factory is not None:
        return factory()
    # Explicit construction for the standard three sizes (ImageNet-1K, 224px)
    cfg = {
        "vssm_tiny": dict(depths=[2, 2, 9, 2], dims=96),
        "vssm_small": dict(depths=[2, 2, 27, 2], dims=96),
        "vssm_base": dict(depths=[2, 2, 27, 2], dims=128),
    }[tag]
    return VSSM(patch_size=4, in_chans=3, num_classes=1000, ssm_ratio=2.0,
                depths=cfg["depths"], dims=cfg["dims"], downsample_version="v3",
                patchembed_version="v2", drop_rate=0.0, drop_path_rate=0.0)


class VMambaClassifier(nn.Module):
    """Wraps a VSSM so that `.forward_features` + `.head` match the ViT interface."""

    def __init__(self, tag: str, checkpoint_path: str | None = None):
        super().__init__()
        self.net = build_vssm(tag)
        self.tag = tag
        if checkpoint_path:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd = {k: v for k, v in sd.items() if not k.startswith("head.weight_norm")}
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            self.load_report = {"missing": list(missing)[:20],
                                "unexpected": list(unexpected)[:20]}
        self.net.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_feat: bool = True) -> dict:
        f = self.net.forward_features(x)          # (B, C, H, W)
        if f.ndim == 4:
            f = f.mean(dim=(2, 3))
        elif f.ndim == 3:
            f = f.mean(dim=1)
        logits = self.net.head(f)
        return {"feat": f, "logits": logits}
