"""Colab notebook helper: run a named pipeline step in a GPU session and persist results.

Design constraints learned the hard way:
  * Drive mount can fail on a free account. It must never be fatal — results are written to
    /content/out regardless, and mirrored to Drive *only if* a mount exists.
  * Kaggle credentials are supplied either as a file at /content/kaggle.json or as the
    environment variables KAGGLE_USERNAME / KAGGLE_KEY (set in the notebook).
  * A step must be resumable: if its output file already exists it is skipped.
"""
from __future__ import annotations

import glob
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
DRIVE_CANDIDATES = ("/content/drive/MyDrive", "/content/drive/My Drive")
VMAMBA = "/content/vmamba"

INSTALL = ["timm==1.0.22", "kaggle", "scikit-image", "fvcore"]
DRIVE_ROOT: str | None = None


# --------------------------------------------------------------------------------------
def drive_root() -> str | None:
    """First existing Drive folder, or None."""
    for c in DRIVE_CANDIDATES:
        if os.path.isdir(c):
            return c
    return None


def mount_drive() -> str | None:
    """Try to mount Drive; return its root or None. Never raises."""
    global DRIVE_ROOT
    if DRIVE_ROOT:
        return DRIVE_ROOT
    if drive_root() is None:
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print("Drive not mounted (results stay in /content/out):", str(e)[:120])
    DRIVE_ROOT = drive_root()
    if DRIVE_ROOT:
        print("Drive root:", DRIVE_ROOT)
    return DRIVE_ROOT


def gpu_check(required: bool = True) -> str:
    """Return the accelerator name. CPU is allowed for steps that opt out."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if required:
        raise RuntimeError(
            "No GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again. "
            "Alternatively run a *_cpu step, which works without a GPU.")
    return "cpu"


def setup(require_gpu: bool = True) -> None:
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

    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    os.makedirs(OUT, exist_ok=True)
    if not os.environ.get("SKIP_DRIVE"):
        mount_drive()

    commit = subprocess.run(["bash", "-lc", f"cd {REPO} && git rev-parse --short HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print("repo commit:", commit)
    print("device:", gpu_check(required=require_gpu))
    print("OK_SETUP", flush=True)


def kaggle_creds() -> None:
    """Make Kaggle credentials available to the CLI. Never raises if already present."""
    dst = "/root/.kaggle/kaggle.json"
    if os.path.exists(dst):
        return
    os.makedirs("/root/.kaggle", exist_ok=True)

    for cand in ("/content/kaggle.json",):
        if os.path.exists(cand):
            shutil.copy(cand, dst)
            os.chmod(dst, 0o600)
            print("kaggle creds from", cand)
            return
    d = drive_root()
    if d:
        for cand in glob.glob(os.path.join(d, "**", "kaggle.json"), recursive=True):
            shutil.copy(cand, dst)
            os.chmod(dst, 0o600)
            print("kaggle creds from", cand)
            return

    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        with open(dst, "w") as f:
            json.dump({"username": user, "key": key}, f)
        os.chmod(dst, 0o600)
        print("kaggle creds from environment")
        return

    raise RuntimeError(
        "No Kaggle credentials. Put kaggle.json in /content/ or set "
        "KAGGLE_USERNAME and KAGGLE_KEY in the notebook before running.")


# --------------------------------------------------------------------------------------
def persist(step: str) -> None:
    d = drive_root()
    src = os.path.join(OUT, step)
    if not d or not os.path.isdir(src):
        return
    dst = os.path.join(d, "arch-complementarity", step)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f))
    print("persisted to", dst)


def run(step: str) -> dict:
    t0 = time.time()
    info = {"step": step, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": False}
    os.makedirs(os.path.join(OUT, step), exist_ok=True)
    try:
        kaggle_creds()
        import importlib.util
        module_file = os.path.join(SRC, "steps", f"{step}.py")
        if not os.path.exists(module_file):
            raise FileNotFoundError(module_file)
        spec = importlib.util.spec_from_file_location(f"arch_comp_step_{step}", module_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        info["result"] = mod.main(OUT)
        info["ok"] = True
    except Exception:
        info["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        info["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(OUT, step, "status.json"), "w") as f:
            json.dump(info, f, indent=2, default=str)
        persist(step)
        print("STATUS", json.dumps({k: v for k, v in info.items() if k != "error"},
                                   default=str))
        print("STEP_FINISHED", step, info["ok"], flush=True)
    return info


def save(step: str, name: str, obj) -> str:
    d = os.path.join(OUT, step)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("wrote", path)
    return path
