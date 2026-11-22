# HANDOFF — resume this project on a new account

Last updated: 22 Nov 2026. Repo commit: `edb8c96`

## Read this first

The project is **fully scaffolded and has produced zero results**. All code, the research
design, and the verification gate are on GitHub. Nothing of value lived in the Colab
runtime, so switching accounts loses nothing.

**Blocker:** Colab free GPU quota for account `akshaytheflash@gmail.com` was exhausted on
22 Nov 2026 ("You cannot currently connect to a GPU due to usage limits in Colab").

## Resume in 4 steps (new account)

1. Sign in to <https://colab.research.google.com> with the new account.
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. Download a fresh Kaggle token (kaggle.com → Settings → Create New Token) and upload
   `kaggle.json` to `/content/` in the Colab file browser.
4. Open this URL and click **Run all**:
   `https://colab.research.google.com/github/akshaytheflash/arch-complementarity/blob/main/notebooks/step_01_checkpoints.ipynb`

The notebook pulls code from GitHub itself — nothing needs uploading.

## What must pass before any science is done

`step_01_checkpoints` is a gate, not an experiment. It runs every model over a stratified
2,000-image ImageNet-val subset and requires each to reproduce its published top-1 within
**1.0 pp**. Any model that fails is marked `usable: false` and is excluded downstream.

Do not proceed to later steps until at least one ViT and one VMamba are `usable: true`.

Expected values (224px, ImageNet-1K, no fine-tuning):

| name | published top-1 | params |
|---|---|---|
| deit-small | 79.8 | 22.1M |
| vit-small | 81.3 | 22.1M |
| swin-tiny | 81.3 | 28.3M |
| vit-base | 85.4 | 86.6M |
| vmamba-tiny | 82.5 | 31.0M |
| vmamba-small | 83.6 | 50.0M |

## Compute budget (stratified 10k protocol, per your decision)

| step | est. T4 time |
|---|---|
| 01 checkpoints (2k images × 6 models) | ~8 min |
| 02 inference: 4 models × 10k val | ~25 min |
| 03 OOD: 4 models × (A, R, Sketch, V2) | ~35 min |
| 04 corruption: 15 types × 5 sev × 2 models × 2k | ~60 min |
| 05 explanations: occlusion 7×7 × 2 models × 1k | ~45 min |
| 06 adversarial: FGSM/PGD transfer, 2×2 matrix | ~40 min |
| 07 probes: error predictability | ~10 min |
| 08 analysis + figures | ~5 min |

Total ≈ 3.5 GPU-hours. Free Colab gives ~12 h/day per account but disconnects on idle.

## Actual credentials / resources (verified working)

- **GitHub token** (`repo` scope): stored in `~/.config/gh/hosts.yml`, account
  `akshaytheflash`. Repo is public, so Colab needs no token.
- **Kaggle**: `KGAT` / `273a8cc49a7dc8057fc3881a3075454d` (in `~/.kaggle/kaggle.json`).
  Verified working against the Kaggle API. The new Colab account needs its own token —
  any valid Kaggle account can download the datasets (all are public).
- **Datasets** (public Kaggle, all verified to exist with the exact structures the
  loaders expect):

| role | Kaggle ref | size |
|---|---|---|
| ImageNet-1K val | `tusonggao/imagenet-validation-dataset` | 6.7 GB |
| ImageNet-A | `paultimothymooney/natural-adversarial-examples-imageneta` | 0.7 GB |
| ImageNet-R | `mauricioalvarezj/imagenet-r` | 2.1 GB |
| ImageNet-Sketch | `wanghaohan/imagenetsketch` | 15.2 GB |
| ImageNet-Bg (background only) | `xiangexiang/imagenet-bg-ospcoop-cvpr2025` | 2.6 GB |

- **Checkpoints**: all URLs verified live (see `src/models.py::VMAMBA_URLS`). ViT/Swin via
  `timm`, VMamba v2 via the `#v2cls` GitHub release. VMamba source cloned from
  `MzeroMiko/VMamba`; its pure-PyTorch `selective_scan_torch` is used (no CUDA compile).

## If GPU stays unavailable

- **Kaggle Notebooks** — same Kaggle key works, 30 GPU-h/week, no extra token. Needs
  `src/nb.py` to write to `/kaggle/working` instead of Drive (~20 min port).
- **CPU fallback** — Colab CPU works today. Drop to a 2k-image pilot; VMamba's Python-loop
  selective scan makes it 60–100× slower, so budget accordingly.

## What is deliberately NOT done

No results, no tables, no paper. The 6-page IEEE paper cannot be drafted before the runs:
all five research questions (ensemble effectiveness, confidence-vs-disagreement,
adversarial transfer, explanation divergence, OOD complementarity) are empirical.
Writing it now would require inventing effect sizes.
