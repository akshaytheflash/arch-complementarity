"""Dataset loaders for every protocol in the study.

All datasets are obtained from Kaggle (the runtime has Kaggle credentials only; there
is no ImageNet download key). The five validation-family datasets plus the
background-only set are enough for every experiment in the design document:

    val      ImageNet-1K validation            tusonggao/imagenet-validation-dataset
    A        ImageNet-A (natural adversarial) paultimothymooney/natural-adversarial-examples-imageneta
    R        ImageNet-R (renditions)           mauricioalvarezj/imagenet-r
    sketch   ImageNet-Sketch                   wanghaohan/imagenetsketch
    bg       ImageNet-Bg (background only)     xiangexiang/imagenet-bg-ospcoop-cvpr2025
    corrupt  synthesised on the fly from `val`  (see src/corruption.py)
"""
from __future__ import annotations

import os
import random
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA = "/content/data"
SYNSET_TO_IDX = None   # filled by load_synsets()


# ---------------------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------------------
def kaggle_download(ref: str, root: str = DATA) -> str:
    """Download + unzip a Kaggle dataset. Returns the extraction directory."""
    import kaggle  # noqa: F401  (ensures the package is importable)

    slug = ref.replace("/", "__")
    dest_dir = os.path.join(root, slug)
    os.makedirs(dest_dir, exist_ok=True)
    marker = os.path.join(dest_dir, ".done")
    if os.path.exists(marker):
        return dest_dir

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(ref, path=dest_dir, unzip=True, quiet=False)
    open(marker, "w").write("ok")
    return dest_dir


# ---------------------------------------------------------------------------------------
# ImageNet class bookkeeping
# ---------------------------------------------------------------------------------------
def imagenet_class_index() -> dict[int, str]:
    """{0..999 -> n########} using the standard ImageNet class order."""
    try:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        return {i: info.index_to_synset(i) for i in range(1000)}
    except Exception:
        # Hungarian order fallback from the widely used mapping file
        import json
        path = os.path.join(os.path.dirname(__file__), "data", "imagenet_class_index.json")
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v[0] for k, v in raw.items()}


def load_synsets() -> dict[str, int]:
    global SYNSET_TO_IDX
    if SYNSET_TO_IDX is None:
        SYNSET_TO_IDX = {v: k for k, v in imagenet_class_index().items()}
    return SYNSET_TO_IDX


# ---------------------------------------------------------------------------------------
# Folder datasets (val / A / R / sketch / bg) — all are <split>/<wnid>/<file>.jpg
# ---------------------------------------------------------------------------------------
def find_image_root(extracted: str) -> str:
    """Locate the deepest directory whose children are mostly wnid directories."""
    best, best_score = extracted, -1
    for dirpath, dirnames, filenames in os.walk(extracted):
        if not dirnames:
            continue
        wnid = sum(1 for d in dirnames if len(d) == 9 and d.startswith("n") and d[1:].isdigit())
        score = wnid
        if score > best_score:
            best, best_score = dirpath, score
    return best


class FolderImages(Dataset):
    """Deterministic subset of a wnid-structured ImageNet-style folder tree."""

    def __init__(self, root: str, transform, limit: int | None = None, seed: int = 0,
                 shuffle: bool = False):
        self.root = root
        self.transform = transform
        synsets = load_synsets()
        items = []
        for wnid in sorted(os.listdir(root)):
            if not wnid.startswith("n"):
                continue
            d = os.path.join(root, wnid)
            if not os.path.isdir(d):
                continue
            label = synsets.get(wnid)
            if label is None:
                continue
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    items.append((os.path.join(d, fn), label))
        if limit is not None and len(items) > limit:
            rng = random.Random(seed)
            if shuffle:
                rng.shuffle(items)
            else:
                # stratified: keep proportionally to class size, preserving class balance
                items = _stratified(items, limit, rng)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _stratified(items, limit, rng):
    by_cls = {}
    for it in items:
        by_cls.setdefault(it[1], []).append(it)
    out = []
    keys = sorted(by_cls)
    per = max(1, limit // len(keys))
    for k in keys:
        v = by_cls[k]
        rng.shuffle(v)
        out.extend(v[:per])
    return out[:limit]


class BackgroundOnly(Dataset):
    """ImageNet-Bg: the same wnid structure but with the objects removed/replaced.

    Filename = ILSVRC2012_val_00000123.JPEG; label comes from the val annotation order
    (ILSVRC2012_val_00000001 is the first val image). ImageNet-Bg images keep the same
    val index, so labels are recoverable from `val_index_to_label`.
    """

    def __init__(self, root: str, transform, val_index_to_label: dict[int, int], limit=None):
        self.root, self.transform = root, transform
        files = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    files.append(os.path.join(dp, fn))
        files.sort()
        pairs = []
        for p in files:
            base = os.path.basename(p)
            try:
                idx = int(base.split("_")[-1].split(".")[0])
            except ValueError:
                continue
            if idx in val_index_to_label:
                pairs.append((p, val_index_to_label[idx]))
        if limit:
            pairs = pairs[:limit]
        self.items = pairs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y


class Corrupted(Dataset):
    """Applies a corruption to the *same* underlying images (stratified val subset)."""

    def __init__(self, base: FolderImages, corruption: str, severity: int, seed=0,
                 cache_dir: str | None = None):
        self.base, self.name, self.severity, self.seed = base, corruption, severity, seed
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        path, label = self.base.items[i]
        if self.cache_dir:
            cp = os.path.join(self.cache_dir, self.name, str(self.severity),
                              os.path.basename(path))
            if os.path.exists(cp):
                img = np.asarray(Image.open(cp).convert("RGB"))
            else:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                img = np.asarray(Image.open(path).convert("RGB"))
                from . import corruption  # late import to keep loaders light
                img = corruption.apply_corruption(self.name, img, self.severity,
                                                  seed=self.seed + i)
                Image.fromarray(img).save(cp, quality=95)
        else:
            img = np.asarray(Image.open(path).convert("RGB"))
            from . import corruption
            img = corruption.apply_corruption(self.name, img, self.severity,
                                              seed=self.seed + i)
        return self.base.transform(img), label


def val_order_labels(root: str) -> dict[int, int]:
    """ILSVRC2012_val_NNNNNNNN -> class index, from a wnid folder tree."""
    syn = load_synsets()
    out = {}
    for wnid in os.listdir(root):
        d = os.path.join(root, wnid)
        if not os.path.isdir(d) or wnid not in syn:
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                try:
                    idx = int(fn.split("_")[-1].split(".")[0])
                except ValueError:
                    continue
                out[idx] = syn[wnid]
    return out
