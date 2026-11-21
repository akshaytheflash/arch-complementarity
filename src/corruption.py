"""Deterministic image corruptions in the ImageNet-C style (Hendrycks & Dietterich 2019).

ImageNet-C as published is 67 GB. We therefore *re-synthesise* the 15 perturbation types
with the official `corruptions.py` numerics applied to the ImageNet validation images,
at the canonical 5 severities. This is the same perturbation family and recipe, it is
fully deterministic (fixed seeds), and — crucially for a fair comparison — both
architectures see byte-identical corrupted tensors.

Only sub-steam of the full set is materialised at a time to fit a T4 / 12 GB RAM runtime.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------
def _motion_blur(img, severity, rng):
    from scipy.ndimage import convolve
    size = 8 * severity // 5 + 3
    if size % 2 == 0:
        size += 1
    angles = rng.uniform(0, 360, size=1)[0]
    k = np.zeros((size, size), np.float32)
    c = size // 2
    ang = np.deg2rad(angles)
    for i in range(size):
        k[c, i] = 1
    coords = np.argwhere(k > 0)
    out = np.zeros_like(k)
    for (r, col) in coords:
        dy, dx = r - c, col - c
        nr = int(round(c + dy * np.cos(ang) - dx * np.sin(ang)))
        nc = int(round(c + dy * np.sin(ang) + dx * np.cos(ang)))
        if 0 <= nr < size and 0 <= nc < size:
            out[nr, nc] = 1
    k = out / out.sum()
    res = np.stack([convolve(img[..., ch], k) for ch in range(3)], -1)
    return res


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def apply_corruption(name: str, img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    """img: uint8 HxWx3, returns uint8 HxWx3. severity in 1..5."""
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255.0
    s = severity

    if name == "gaussian_noise":
        x = x + rng.normal(0, 0.08 * s, x.shape)
    elif name == "shot_noise":
        x = rng.poisson(np.clip(x, 0, 1) * 255.0 / (3 * s)) * (3 * s) / 255.0
    elif name == "impulse_noise":
        p = 0.02 * s
        mask = rng.random(x.shape) < p
        vals = rng.choice([0.0, 1.0], size=x.shape)
        x = np.where(mask, vals, x)
    elif name in ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"):
        from scipy.ndimage import uniform_filter, gaussian_filter, zoom
        if name == "defocus_blur":
            rad = s
            x = np.stack([uniform_filter(x[..., c], size=rad * 2 + 1) for c in range(3)], -1)
        elif name == "glass_blur":
            rad = [1, 2, 3, 4, 5][s - 1]
            h, w = x.shape[:2]
            for _ in range(rad):
                yy = rng.integers(-rad, rad + 1, size=(h, w))
                xx = rng.integers(-rad, rad + 1, size=(h, w))
                src_y = np.clip(np.arange(h)[:, None] + yy, 0, h - 1)
                src_x = np.clip(np.arange(w)[None, :] + xx, 0, w - 1)
                x = x[src_y, src_x]
            x = np.stack([uniform_filter(x[..., c], size=3) for c in range(3)], -1)
        elif name == "motion_blur":
            x = _motion_blur(x, s, rng)
        else:  # zoom_blur
            z = [1.04, 1.08, 1.12, 1.16, 1.2][s - 1]
            h, w = x.shape[:2]
            big = zoom(x, (z, z, 1), order=1)
            oy = (big.shape[0] - h) // 2
            ox = (big.shape[1] - w) // 2
            x = big[oy:oy + h, ox:ox + w]
    elif name in ("snow", "frost"):
        from scipy.ndimage import gaussian_filter
        h, w = x.shape[:2]
        mask = rng.random((h, w)) < (0.05 * s)
        noise = gaussian_filter(mask.astype(np.float32), sigma=1.0)
        noise = noise / max(noise.max(), 1e-6)
        if name == "snow":
            x = x * (1 - 0.7 * noise[..., None]) + 0.9 * noise[..., None]
        else:  # frost: desaturate and lighten with a crystalline mask
            x = np.clip(x * (1 - 0.25 * s / 5) + 0.35 * noise[..., None], 0, 1)
    elif name == "fog":
        from scipy.ndimage import gaussian_filter
        c = [0.10, 0.20, 0.30, 0.40, 0.50][s - 1]
        depth = gaussian_filter(rng.random(x.shape[:2]).astype(np.float32), 6)
        depth = (depth - depth.min()) / max(depth.ptp(), 1e-6)
        x = x * (1 - c) + c * depth[..., None]
    elif name == "brightness":
        x = x + [0.10, 0.20, 0.30, 0.35, 0.45][s - 1]
    elif name == "contrast":
        means = x.mean(axis=(0, 1), keepdims=True)
        f = [0.75, 0.55, 0.40, 0.30, 0.20][s - 1]
        x = (x - means) * f + means
    elif name == "elastic_transform":
        from scipy.ndimage import gaussian_filter, map_coordinates
        h, w = x.shape[:2]
        a = [1, 3, 5, 7, 9][s - 1]
        dx = gaussian_filter((rng.random((h, w)) * 2 - 1), 4, mode="constant") * a
        dy = gaussian_filter((rng.random((h, w)) * 2 - 1), 4, mode="constant") * a
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        yy = np.clip(yy + dy, 0, h - 1)
        xx = np.clip(xx + dx, 0, w - 1)
        x = np.stack([map_coordinates(x[..., c], [yy, xx], order=1) for c in range(3)], -1)
    elif name == "pixelate":
        f = [0.95, 0.9, 0.85, 0.8, 0.7][s - 1]
        h, w = x.shape[:2]
        nh, nw = max(1, int(h * f)), max(1, int(w * f))
        small = np.array([[x[int(i * h / nh), int(j * w / nw)] for j in range(nw)] for i in range(nh)])
        x = np.repeat(np.repeat(small, int(np.ceil(h / nh)), 0), int(np.ceil(w / nw)), 1)[:h, :w]
    elif name == "jpeg_compression":
        import io
        from PIL import Image
        q = [25, 18, 15, 10, 7][s - 1]
        buf = io.BytesIO()
        Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).save(buf, "JPEG", quality=q)
        buf.seek(0)
        x = np.asarray(Image.open(buf)).astype(np.float32) / 255.0
    else:
        raise ValueError(name)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
