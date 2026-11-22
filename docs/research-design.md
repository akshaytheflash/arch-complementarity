# Research Design — Architectural Inductive Bias as a Complementarity Signal

Repo: `arch-complementarity` (working dir `vit-vmamba-ensemble/`)
Date: 21 Nov 2026
Status: design frozen before experiments; results section will be written only after runs.

---

## 1. Positioning against prior work (novelty)

Three literatures that are adjacent but do **not** answer the question we ask:

| Prior work | What it does | What it does not do |
|---|---|---|
| Lee, *Consensus vs. Controversy: Mapping the Decision Space Where Architectures Diverge*, CVPR 2026 | 12 timm models over {CNN (7), ViT (3), MLP (2)}; top-10% disagreement tail vs consensus tail; agreement matrices; controversy score; OOD on ImageNet-A/R | **No SSM/Mamba family** (`grep -c mamba` on the paper text = 0); disagreement is defined as the *std of max-probability across models*, i.e. a confidence-dispersion statistic, not a functional disagreement; never tests whether disagreement is a *usable selective-prediction* signal, never does explanation analysis, never does adversarial transfer, never decomposes ensemble error. Notes "the framework ... remains statistical. It does not definitively attribute controversy to texture vs. shape bias, global vs. local information" — i.e. leaves explanation-level analysis as explicit future work. |
| Yin et al., *Bridging Human and Model Attention: CNN, Mamba, ViT with Gaze-Based Validation*, arXiv 2610.12703 (Oct 2026) | Explainer-attribution alignment with eye tracking on CUB / MIMIC-CXR / KDEF; a small ViT+Mamba ensemble on CUB / KDEF | Explanations compared to *human gaze*, not to each other; no occlusion-style faithfulness or background-forcing; no error overlap, OOD/corruption, adversarial, adversarial transfer, or selective prediction. |
| Terjek & Jalali, *Do VLMs Need Vision Transformers? Evaluating SSMs as Vision Encoders*, arXiv 2603.19209 (2026) | VMamba vs ViT vs MaxViT vs MambaVision as frozen backbones inside VLMs | VLM-centric (VQA/grounding); no classification-level error, corruption, OOD, adversarial or explanation analysis; no complementarity analysis. |
| VM-BeautyNet, arXiv 2510.16220 (2025) | ViT + Mamba ensemble for facial beauty regression, SCUT-FBP5500 | Application paper; no error decomposition; variance is tiny (MAE 0.22–0.21) so ensemble gains are near-noise; no conditions stated for *when* it fails. |
| MambaOut / Yu & Wang, CVPR 2025 | Shows SSM token-mixers under-perform gated-conv on ImageNet/COCO/ADE20K under *matched training* | Answers "is Mamba needed?" at aggregate accuracy; does not ask whether SSM architectures carry a *distinct* error signature that is exploitable via complementarity. |

**The gap.** Two independent 2026 works (Lee; Yin et al.) each approach one half of the question — Lee provides a statistical disagreement framework but excludes SSMs and explanation analysis; Yin et al. provide explanation analysis including Mamba but not complementarity statistics, robustness, or error analysis. **No work puts an SSM** (`VMamba`) **and a plain ViT on a matched protocol and asks whether the architectural difference produces (i) complementary errors, (ii) a disagreement signal that beats confidence for selective prediction, (iii) the same or different adversarial perturbations (transferability), (iv) the same or different spatial reliance, (v) complementary failures under distribution shift.** VM-BeautyNet asserts "ViT and Mamba are complementary" on a single subjective regression task; nobody has measured *where* and *when* that complementarity actually exists and when it fails.

**Defensive framing (to keep the contribution honest).** *ViT is a universal approximator of any CNN / SSM token mixer under sufficient heads and width* (Yun et al. 2020; Cordonnier et al. ICLR 2020; Trockman & Kolter 2023; late 2025 surveys). So any error-complementarity we observe cannot be a statement about *capacity*; it must be a statement about **what is learned under a given data distribution, optimizer, and training budget** — i.e., about **inductive bias as a regulariser**, not about expressivity. The paper must say that plainly (this is what reviewers will ask first).

---

## 2. Hypotheses

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Cross-architecture ensembles (ViT + VMamba) give larger accuracy gains per member than matched within-architecture ensembles. | Pair all 2-model subsets of 4 ID models on ImageNet-val, compare member-relative gain to hypergeometric-null (and permutation test) for cross-arch vs same-arch pairs. |
| H2 | Error overlap between ViT and VMamba is *positive* (errors are correlated far above chance, because both are strong models trained on the same data) but *lower* than between two same-family models. | Jaccard index of error sets; together with the "disagreement-is-not-accuracy" control (H2b). |
| H3 | Functionally, the disagreement signal (`1[max p_ViT == max p_VMamba]` or Jensen-Shannon distance between posteriors) gives better selective-prediction AUC than the best member's confidence, *and this advantage is concentrated on complementarity-relevant subsets* (controversial tail, corruptions where the two architectures differ in sensitivity). | AURC, AUROC(correct), risk at 20% coverage, pairwise DeLong test. |
| H4 | Architecture-specific corruption sensitivity is **not** the same profile: the ViT degrades more smoothly, the VMamba more jaggedly, on the SAME 15-corruption × 5-severity protocol. | Per-corruption error differences, paired bootstrap, and the correlation of corruption-error *ranks* between models. |
| H5 | Background dependency differs: occlusion error at the image centre minus that in the border ring is larger for the VMamba than for the ViT (or vice versa, whichever the data says). | Occlusion on a 7×7 grid; Δ = inner/BG error gap; per-model CIs; two-architecture paired test on the same images. |
| H6 | Adversarial *transferability* is **lower than within-architecture** transfer for matched models: PGD on the ViT fools the VMamba substantially less often than PGD on another ViT fools the ViT. And **a transfer-attack is a more powerful complementarity probe** than random corruption. | FGSM and PGD on ImageNet-1K-val (200–500 images); transferability matrix (2×2); paired comparison at matched ε. |
| H7 | Explanation structure: when both models are correct, their occlusion/RISE maps are *spatially similar but not identical* — shared structure (real objects) plus architecture-specific saliency — and on *disagreement* examples the differing regions concentrate on the background / peripheral context. | Per-image SSIM / Pearson between the two maps (correct-both vs disagree); split each map into "inner 50% / object" vs "border ring"; also a "same-object-mask" control (the annotated mask, where we have one). |
| H8 | Error *predictability*: a feature-fusion head is not needed; a *single* probe on the ViT's (or the VMamba's) internal features predicts *the other model's* correctness at AUC_A→B > the target model's own confidence AUC. | Add a small "rich-disagreement" analysis: linear probe of "will B be wrong?" on A's penultimate features vs AUCs of A and B's own confidence. |
| H9 | OOD complementarity: failure sets are more complementary on corrupted data than on ID (OOD), and the ensemble gain from *adding the second architecture* is larger under corruption than on clean ID. | Per-corruption paired analysis; ensemble oracle difference; split by "both-correct/both-wrong/A-only/B-only". |

Note H1, H2, H5, H7, H8 do not assume a sign; the experiments choose the sign.

---

## 3. Experiments (protocols)

### 3.1 Models

`models.yaml` selects four pretrained checkpoints — required: at least 2 ViTs, 2 SSMs; all 224×224 ImageNet-1K.

The choice must be **matched in size and accuracy** across architectures or "ViT vs VMamba" collapses into "big model vs small model". Targets (top-1, 224, IN-1K, from prep downloaded with the verified numbers):

| Slot | Model | Params | Top-1 | Role |
|---|---|---|---|---|
| `vit-s` | `vit_small_patch16_224.augreg_in1k` | 22M | 81.3 | ViT-small |
| `vit-b` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 86M | 85.4 | ViT-base (secondary; compare with `vit-base-in1k` 84.5) |
| `ssm-t` | `VMamba-T` (`vssm_tiny` s1l8) | 30M | 82.6 | VMamba, ~ViT-S scale, slightly higher acc. |
| `ssm-s` | `VMamba-S` (s2l15) | 50M | 83.6 | VMamba, ~ViT-S/DeiT-S scale |
| `ssm-b` | `VMamba-B` (s2l15) | 89M | 83.9 | VMamba, ~ViT-B scale |
| `deit-s` | `deit_small_patch16_224.fb_in1k` | 22M | 79.8 | matched same-family ViT pair for the "same-arch vs cross-arch" contrast |

The `vmamba` code must be integrated with the local `selective_scan` CUDA extension or fallback path; the fallback with PyTorch SDPA may be used if compilation fails (document which).

### 3.2 Data

| Dataset | Role | Size |
|---|---|---|
| ImageNet-1K-val | In-distribution (ID); everything is anchored here | 50,000 (subset 10,000 for adversarial & explanation) |
| ImageNet-C (Hendrycks & Dietterich 2019) | Corruption sensitivity (15 × 5 severities) — reuse their corrupted-image dataset on a fixed subset to save time | 50,000 (colab: 5,000) per (corruption, severity) |
| ImageNet-A (Hendrycks et al. 2021) | Natural OOD | 7,500 |
| ImageNet-R (Hendrycks et al. 2021) | Rendition OOD | 30,000 |
| ImageNet-Sketch (Wang et al. 2019) | Sketch OOD | 50,000 |
| ImageNet-V2 (Recht et al. 2019) | Showcase | 10,000 |

ID | 5,000 for the main protocol, 50,000 for the final headline table if time permits.

### 3.3 Protocol per experiment

1. **Inference**: per image, store logits for every model on every dataset (`pred_<model>_<dataset>.npz`). Compute accuracies; cross-check against published top-1 (a sanity check; if a checkpoint is missing, log which and continue with the others; never report a made-up number).
2. **Ensemble**: enumerate all 2-model subsets × {prob-average, logit-average} and the matched 4-model full ensemble.
3. **Disagreement vs confidence (selective prediction)**: for each pair, for each confidence definition (max softmax on each model, average of the two), for each disagreement definition (hard disagreement = argmax differs; Jensen-Shannon distance; 1 − cosine), compute AURC, AUROC, and risk@coverage ∈ {.10, .20, .30, .50, .70, .90}. Statistical test: DeLong (for AUCs) with 10,000 bootstrap on the per-pair difference on the *same* images. **Fully partition by ID/OOD/corruption.**
4. **Error overlap**: for every model pair, error-overlap = Jaccard(errorA, errorB); complementarity = 1 − Jaccard. Chance model (independence) = p_A p_B / (p_A + p_B − p_A p_B). Δ over chance, with bootstrap CI. Same-family vs cross-family.
5. **Occlusion explanations**: 7×7 grid, 32×32 grey patch; for every model INT its predicted class and evaluate Δlogit(ground truth) + Δmax; normalise to a probability map. Statistics: (a) Pearson/SSIM between per-image ViT & VMamba maps; (b) "inner 50% vs border 20% ring" mass statistic; (c) on correct-both vs disagree examples; (d) on background-forced (ImageNet-9 style) examples, per-class or per-superclass.
6. **Adversarial transferability**: FGSM ε∈{1,2,4,8}/255 and PGD-10 ε=4/255 α=1/255; for each source model (2), attack 200–500 images it classifies correctly; evaluate transfer to the other; compute targeted & untargeted transfer; 2×2 confusion; also transfer to a *third* model not used as source. Bootstrap CIs.
7. **Probes (H8)**: penultimate features on the 5,000-image ID set (an average-pooled feature), logistic probe with 5-fold CV: target = B-wrong. Compare with B's own confidence AUC.
8. **Repetitions**: everything stochastic uses ≥ 3 seeds where seeds exist; prompt says `n=...`.

### 3.4 Statistical analysis

- All confidence intervals: percentile 10,000 bootstrap over images (not over checkpoints), paired where the paired comparison is meaningful.
- Pairwise comparisons: DeLong test for AUC; paired permutation for accuracy; McNemar's for error overlap; Holm–Bonferroni.
- Effect sizes reported with every p-value; "no difference" conclusions require a TOST non-inferiority statement at a stated δ (5% accuracy, or 0.02 AUC).
- Pre-registration: the design above is frozen; all failed runs, OOMs and abandoned configs to be recorded in `logs/`.

---

## 4. Failure conditions (when does the ensemble fail?)

We will explicitly test and report:
- **Correlated training already removed diversity**: if error-trap sets are tiny (only ~1–2% of images), ensembles are a *compute* waste.
- **Complementarity does not survive adversarial**: if architectural-transfer ≈ within-arch-transfer.
- **Explanation disagreement is not a signal**: if "both correct yet disagree" maps are as similar as random.
- **Sign reversal across datasets**: if the VMamba wins on natural images and the ViT on stylised ones, the "complementarity direction" is dataset-specific.
- **Compute**: report ensemble cost (2× inference, 2× memory) alongside the gain so that the cost/benefit is visible.

---

## 5. Acceptance criteria

- [ ] ≥ 15 sources reviewed. 
- [ ] Checkpoints loaded and accuracies match published within 0.5% (else the checkpoint is wrong / preprocessing is wrong).
- [ ] ID ≥ 5,000; corruption 15 × 5 × ≥ 1000 (subset) → ≥ 3,000 images per pair-selective analysis.
- [ ] OOD: at least ImageNet-A, -R, -Sketch, -V2, with at least 1,000 images each.
- [ ] Every number in the paper traces to a script in the repo.
- [ ] 6-page IEEE paper compiled; template matches `Full-Paper-template.docx`.
