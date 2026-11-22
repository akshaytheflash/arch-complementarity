"""Dataset loaders for every protocol in the study.

Sources are Kaggle datasets (the runtime has Kaggle credentials; there is no
ImageNet-download key). Each is wnid-folder structured, so a single loader covers them:

    val      ImageNet-1K validation          tusonggao/imagenet-validation-dataset
    A        ImageNet-A                      paultimothymooney/natural-adversarial-examples-imageneta
    R        ImageNet-R                      mauricioalvarezj/imagenet-r
    sketch   ImageNet-Sketch                 wanghaohan/imagenetsketch
    v2       ImageNetV2                      (see find_val_v2)

ImageNet-Bg (background-only images) uses the *val* labels and is loaded by filename
index, see `BackgroundOnly`.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = "/content/data"


# --------------------------------------------------------------------------------------
# Kaggle
# --------------------------------------------------------------------------------------
def kaggle_download(ref: str, root: str = DATA_ROOT) -> str:
    """Download + extract a Kaggle dataset once. Returns the extraction directory."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest = os.path.join(root, ref.replace("/", "__"))
    marker = os.path.join(dest, ".complete")
    if os.path.exists(marker):
        return dest
    os.makedirs(dest, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(ref, path=dest, unzip=True, quiet=False)
    open(marker, "w").write("ok")
    return dest


# --------------------------------------------------------------------------------------
# ImageNet class bookkeeping
# --------------------------------------------------------------------------------------
def synset_to_index() -> dict[str, int]:
    """{'n01440764': 0, ...} in the standard ImageNet class order used by all our models.

    timm's `ImageNetInfo` API has changed across versions (index_to_synset / index_to_label
    / label_to_index), so several strategies are tried. A final fallback reads the
    ordering straight from the validation folder names against a shipped mapping.
    """
    # 1) timm >= 1.0: ImageNetInfo with index_to_label + label_to_index
    try:
        from timm.data import ImageNetInfo
        ii = ImageNetInfo()
        for meth in ("index_to_synset",):
            if hasattr(ii, meth):
                return {getattr(ii, meth)(i): i for i in range(1000)}
        # label -> index, then map label -> synset via the human-readable name table
        if hasattr(ii, "label_to_index") and hasattr(ii, "index_to_label"):
            return {_label_to_synset(ii.index_to_label(i)): i for i in range(1000)}
    except Exception:
        pass

    # 2) timm's bundled imagenet class-index json
    try:
        import json
        from timm.data import imagenet
        path = getattr(imagenet, "__file__", "")
        for name in ("_imagenet_info.py", "class_map.json"):
            p = os.path.join(os.path.dirname(path), name)
            if os.path.exists(p):
                data = json.load(open(p))
                if isinstance(data, dict) and len(data) >= 1000:
                    out = {}
                    for k, v in data.items():
                        syn = v[0] if isinstance(v, (list, tuple)) else str(v)
                        if syn.startswith("n"):
                            out[syn] = int(k)
                    if len(out) >= 1000:
                        return out
    except Exception:
        pass

    # 3) shipped fallback mapping next to this file
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    fb = os.path.join(here, "data", "imagenet_class_index.json")
    if os.path.exists(fb):
        data = json.load(open(fb))
        return {v[0]: int(k) for k, v in data.items()}

    raise RuntimeError("could not resolve the ImageNet synset->index mapping")


_LABEL_TO_SYNSET: dict[str, str] | None = None


def _label_to_synset(label: str) -> str:
    """Map a timm human-readable class label to its wnid using the shipped table."""
    global _LABEL_TO_SYNSET
    if _LABEL_TO_SYNSET is None:
        import json
        here = os.path.dirname(os.path.abspath(__file__))
        fb = os.path.join(here, "data", "imagenet_class_index.json")
        data = json.load(open(fb))
        _LABEL_TO_SYNSET = {v[1]: v[0] for v in data.values()}
        _LABEL_TO_SYNSET.update({v[1].replace("_", " "): v[0] for v in data.values()})
    return _LABEL_TO_SYNSET.get(label, "")


def find_wnid_root(path: str) -> str | None:
    """Deepest directory whose immediate children are (mostly) wnid folders."""
    best, best_n = None, 0
    for dirpath, dirnames, _ in os.walk(path):
        n = sum(1 for d in dirnames if len(d) == 9 and d.startswith("n") and d[1:].isdigit())
        if n > best_n:
            best, best_n = dirpath, n
    return best


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
class WnidFolder(Dataset):
    """Any <wnid>/<image> tree. Deterministically stratified-subsampled when `limit`."""

    def __init__(self, root: str, transform, limit: int | None = None, seed: int = 0):
        self.transform = transform
        syn = synset_to_index()
        by_class: dict[int, list[str]] = {}
        for wnid in sorted(os.listdir(root)):
            d = os.path.join(root, wnid)
            if not os.path.isdir(d) or wnid not in syn:
                continue
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".jpeg", ".jpg", ".png", ".JPEG", ".JPG")):
                    by_class.setdefault(syn[wnid], []).append(os.path.join(d, fn))
        items: list[tuple[str, int]] = []
        if limit is not None and sum(len(v) for v in by_class.values()) > limit:
            rng = random.Random(seed)
            keys = sorted(by_class)
            per = max(1, limit // max(1, len(keys)))
            for k in keys:
                files = by_class[k][:]
                rng.shuffle(files)
                items.extend((f, k) for f in files[:per])
            items.sort(key=lambda t: t[0])          # deterministic global order
        else:
            for k in sorted(by_class):
                items.extend((f, k) for f in by_class[k])
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), y


class ValIndexed(Dataset):
    """ImageNet-Bg style: filenames carry the ILSVRC2012_val index, labels come from val."""

    def __init__(self, root: str, transform, index_to_label: dict[int, int],
                 limit: int | None = None):
        self.transform = transform
        pairs = []
        for dirpath, _, fns in os.walk(root):
            for fn in fns:
                if not fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue
                try:
                    idx = int(os.path.basename(fn).split("_")[-1].split(".")[0])
                except ValueError:
                    continue
                if idx in index_to_label:
                    pairs.append((os.path.join(dirpath, fn), index_to_label[idx]))
        pairs.sort()
        self.items = pairs[:limit] if limit else pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y


class CorruptedRaw(Dataset):
    """Reads each base image raw, corrupts it, then applies the transform."""

    def __init__(self, paths_labels, transform, name: str, severity: int, seed: int = 0):
        self.items, self.transform = paths_labels, transform
        self.name, self.severity, self.seed = name, severity, seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        from corruption import apply_corruption

        path, y = self.items[i]
        img = np.asarray(Image.open(path).convert("RGB"))
        img = apply_corruption(self.name, img, self.severity, seed=self.seed + i)
        return self.transform(img), y


class Occludable(Dataset):
    """Base dataset that also returns the path, for occlusion / explanation work."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y
