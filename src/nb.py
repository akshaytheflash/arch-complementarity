"""Colab notebook helper: run a named pipeline step in a GPU session and persist results.

Each `notebooks/step_*.ipynb` is self-contained and calls `nb.run("<step>")` at the end.
Results land in `/content/out/<step>/` and are mirrored to Google Drive at
`MyDrive/arch-complementarity/<step>/` so a disconnect costs at most one step.

The *only* thing a notebook must do is: connect to a GPU runtime and Run all.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

REPO_URL = "https://github.com/akshaytheflash/arch-complementarity.git"
REPO = "/content/repo"
SRC = os.path.join(REPO, "src")
OUT = "/content/out"
DRIVE = "/content/drive/MyDrive/arch-complementarity"
VMAMBA = "/content/vmamba"

INSTALL = ["timm==1.0.22", "kaggle", "scikit-image", "fvcore"]


# --------------------------------------------------------------------------------------
def gpu_check() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again.")
    return torch.cuda.get_device_name(0)


def setup() -> None:
    """Install, clone code, mount Drive, expose imports. Idempotent."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *INSTALL], check=False)

    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
    else:
        subprocess.run(["bash", "-lc", f"cd {REPO} && git fetch -q origin main && "
                                       "git reset -q --hard origin/main"], check=False)

    if not os.path.exists(VMAMBA):
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/MzeroMiko/VMamba.git", VMAMBA], check=True)
    os.environ["VMAMBA_ROOT"] = VMAMBA

    if not os.path.exists("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:                      # Drive is optional, not fatal
            print("Drive mount skipped:", e)

    for p in (SRC,):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.makedirs(OUT, exist_ok=True)

    commit = subprocess.run(["bash", "-lc", f"cd {REPO} && git rev-parse --short HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print("repo commit:", commit)
    print("gpu:", gpu_check())
    print("OK_SETUP", flush=True)


def kaggle_creds() -> None:
    """Write Kaggle credentials where the CLI expects them."""
    os.makedirs("/root/.kaggle", exist_ok=True)
    dst = "/root/.kaggle/kaggle.json"
    for cand in ("/content/kaggle.json", os.path.join(DRIVE, "secrets", "kaggle.json")):
        if os.path.exists(cand):
            shutil.copy(cand, dst)
            os.chmod(dst, 0o600)
            print("kaggle creds from", cand)
            return
    if not os.path.exists(dst):
        raise RuntimeError("No Kaggle credentials at /content/kaggle.json")


# --------------------------------------------------------------------------------------
def persist(step: str) -> None:
    """Mirror /content/out/<step> to Drive (best-effort)."""
    src = os.path.join(OUT, step)
    if not os.path.isdir(src) or not os.path.isdir("/content/drive/MyDrive"):
        return
    dst = os.path.join(DRIVE, step)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print("persisted to", dst, os.listdir(dst))


def run(step: str) -> dict:
    """Import `src.steps.<step>` and call its `main()`. Records timings + status."""
    t0 = time.time()
    info = {"step": step, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": False}
    try:
        kaggle_creds()
        mod = __import__(f"steps.{step}", fromlist=["main"])
        result = mod.main(OUT)
        info["ok"] = True
        info["result"] = result
    except Exception:
        info["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        info["seconds"] = round(time.time() - t0, 1)
        os.makedirs(os.path.join(OUT, step), exist_ok=True)
        with open(os.path.join(OUT, step, "status.json"), "w") as f:
            json.dump(info, f, indent=2)
        persist(step)
        print("STATUS", json.dumps({k: v for k, v in info.items() if k != "error"}))
        print("STEP_FINISHED", step, info["ok"], flush=True)
    return info


def save(step: str, name: str, obj) -> str:
    """Saves a JSON-serialisable object under out/<step>/."""
    d = os.path.join(OUT, step)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("wrote", path)
    return path
