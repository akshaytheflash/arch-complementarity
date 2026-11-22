"""Builds a VMamba VSSM using the official upstream implementation.

The upstream repo's `classification/models/csms6s.py` ships a pure-pytorch
`selective_scan_torch`, selected automatically when no CUDA extension is importable
(`WITH_SELECTIVESCAN_* = False`). That is the code path used on a T4, so the forward
pass is the authors' own reference implementation -- no fork.

The notebook setup step clones `https://github.com/MzeroMiko/VMamba` into $VMAMBA_ROOT.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.environ.get("VMAMBA_ROOT", "/content/vmamba")


def _ensure_importable() -> None:
    for p in (os.path.join(ROOT, "classification"), ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    # upstream imports fvcore only for its FLOPs counter; stub it when absent
    try:
        import fvcore  # noqa: F401
    except Exception:
        import types

        stub = types.ModuleType("fvcore")
        stub_nn = types.ModuleType("fvcore.nn")
        stub_nn.FlopCountAnalysis = lambda *a, **k: None
        stub_nn.flop_count_str = lambda *a, **k: ""
        stub_nn.flop_count = lambda *a, **k: {}
        stub_nn.parameter_count = lambda *a, **k: {}
        stub.nn = stub_nn
        sys.modules["fvcore"] = stub
        sys.modules["fvcore.nn"] = stub_nn


def build_vssm(tag: str, cfg: dict, checkpoint_path: str) -> nn.Module:
    _ensure_importable()
    from vmamba import VSSM

    net = VSSM(
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_ratio=2.0,
        depths=cfg["depths"],
        dims=cfg["dims"],
        downsample_version="v3",
        patchembed_version="v2",
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer="ln",
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k: v for k, v in state.items() if not k.endswith("head.weight_norm")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"VMamba {tag}: {len(missing)} missing keys, e.g. {list(missing)[:8]}")
    net.load_report = {"missing": 0, "unexpected": list(unexpected)[:8]}
    net.eval()
    return net
