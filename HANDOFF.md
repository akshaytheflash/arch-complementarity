# HANDOFF — resume this project on a new account

Last updated: 22 Nov 2026. Repo: `akshaytheflash/arch-complementarity` (public).

## Status in one line

**Everything needed to run the full experiment suite is on GitHub. Zero results exist yet.**

## Why it stopped

Colab free-GPU quota for `akshaytheflash@gmail.com` was exhausted on 22 Nov 2026 with
"You cannot currently connect to a GPU due to usage limits in Colab". Switching account
gives a fresh quota.

## Resume (new account), 4 steps

1. Sign in to <https://colab.research.google.com>.
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. Upload a Kaggle token (`kaggle.json`, from kaggle.com → Settings → Create New Token)
   into `/content/` via the file browser.
4. Open **one** of these and click **Run all**:

| notebook | what it does | T4 time |
|---|---|---|
| `notebooks/step_01_checkpoints.ipynb` | **GATE** — every model must reproduce its published top-1 within 1.0 pp or it is excluded | ~8 min |
| `notebooks/step_02_inference.ipynb` | 10k stratified ImageNet-val, logits + features, 4–6 models | ~25 min |
| `notebooks/step_03_ood.ipynb` | ImageNet-A / -R / -Sketch | ~35 min |
| `notebooks/step_04_corruption.ipynb` | 15 corruptions × 3 severities × 2–4 models | ~60 min |
| `notebooks/step_05_explanations.ipynb` | occlusion maps 7×7, 1k images | ~45 min |
| `notebooks/step_06_adversarial.ipynb` | FGSM + PGD transfer matrix | ~40 min |
| `notebooks/step_07_probes.ipynb` | error-predictability probes (CPU) | ~10 min |
| `notebooks/step_08_analysis.ipynb` | all tables + figures + zip to Drive | ~5 min |
| `notebooks/run_all.ipynb` | everything above, one session | ~3.5 h |

URLs are of the form
`https://colab.research.google.com/github/akshaytheflash/arch-complementarity/blob/main/notebooks/<name>.ipynb`

**Run step 01 first, alone.** If it reports `usable: true` for fewer than one ViT and one
SSM, stop and report — everything downstream is meaningless on a broken checkpoint.

## Files in the repo

```
HANDOFF.md                    this file
README.md                     project overview, layout, reproduce commands
docs/research-design.md       frozen design: hypotheses H1–H9, protocols, failure modes
src/models.py                 model registry (6 matched checkpoints) + inference
src/vmamba_pt.py              official VSSM loader (pure-pytorch selective scan)
src/datasets.py               Kaggle loaders (val / A / R / Sketch / background-only)
src/corruption.py             15 corruption types x 5 severities, deterministic
src/metrics.py                overlap, AURC/AUROC, ECE, DeLong, bootstrap, McNemar, Holm
src/nb.py                     Colab harness: setup, GPU check, Drive checkpointing
src/steps/01_checkpoints.py   the verification gate
src/steps/02_inference.py     ID logits + features
src/steps/03_ood.py           OOD logits
src/steps/04_corruption.py    corruption accuracy
src/steps/05_explanations.py  occlusion maps
src/steps/06_adversarial.py   adversarial transfer
src/steps/07_probes.py        error predictability
src/steps/08_analysis.py      all tables + figures (writes results/)
notebooks/step_*.ipynb        one per step
notebooks/run_all.ipynb       orchestrator
```

## Resources verified working

**GitHub** — public repo, no token needed to clone. Pushing uses the token in
`~/.config/gh/hosts.yml` (account `akshaytheflash`).

**Kaggle** — `KGAT` / `273a8cc49a7dc8057fc3881a3075454d`. Public datasets, so any valid
Kaggle account works:

| role | ref | size |
|---|---|---|
| ImageNet-1K val | `tusonggao/imagenet-validation-dataset` | 6.7 GB |
| ImageNet-A | `paultimothymooney/natural-adversarial-examples-imageneta` | 0.7 GB |
| ImageNet-R | `mauricioalvarezj/imagenet-r` | 2.1 GB |
| ImageNet-Sketch | `wanghaohan/imagenetsketch` | 15.2 GB |
| ImageNet-Bg (background only) | `xiangexiang/imagenet-bg-ospcoop-cvpr2025` | 2.6 GB |

**Checkpoints** — all URLs verified live, listed in `src/models.py::VMAMBA_URLS`:

| name | published top-1 | params | family |
|---|---|---|---|
| deit-small | 79.8 | 22.1M | ViT |
| vit-small | 81.3 | 22.1M | ViT |
| swin-tiny | 81.3 | 28.3M | ViT/hybrid |
| vit-base | 85.4 | 86.6M | ViT |
| vmamba-tiny | 82.5 | 31.0M | SSM |
| vmamba-small | 83.6 | 50.0M | SSM |

VMamba source is cloned from `MzeroMiko/VMamba`; its own pure-PyTorch
`selective_scan_torch` path is used, so no CUDA extension compilation is required.

## If GPU stays unavailable

- **Kaggle Notebooks** — the same Kaggle key works and gives 30 GPU-h/week. Only change
  needed: `src/nb.py` writes to `/kaggle/working` instead of Drive.
- **CPU** — Colab CPU works today. VMamba's Python-loop selective scan makes it 60–100×
  slower; a 2k-image pilot is the only sensible CPU scope.

## Known caveats in the current code (fixed on first run-through, or before)

- `metrics.aurc` uses `np.trapezoid` (numpy ≥ 2.0). Colab ships numpy ≥ 2.0; on an older
  numpy it must be `np.trapz`.
- `CorruptedRaw` applies corruption in `__getitem__`; with `num_workers=2` that is fine,
  but each image is corrupted repeatedly per epoch — this is a single-epoch step, so OK.
- Step 04 runs severities 1, 3, 5 only (not all 5) to fit the budget; change `for sev in
  (1, 3, 5)` to `(1, 2, 3, 4, 5)` for the full ImageNet-C protocol.

## Deliberately not done

No results, no tables, no figures, no paper. All five research questions are empirical and
must be answered from the runs above. Writing the 6-page IEEE paper before the numbers
exist would require inventing effect sizes, which the project's integrity rules forbid.
