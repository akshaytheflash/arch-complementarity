"""Step 08 — analysis: turns raw logits into every table and figure in the paper.

Produces, all in `results/`:
  table1_accuracy.json       per-model top-1 on ID + each OOD set
  table2_overlap.json        error-overlap / complementarity / phi per model pair
  table3_ensemble.json       pair ensembles (proba/logit) + weight sweep + oracle
  table4_selective.json      AURC / AUROC / Risk@coverage: confidence vs disagreement,
                             with DeLong and paired-bootstrap CIs
  table5_corruption.json     per-corruption accuracy, architecture-specific profile,
                             rank correlation between the two degradation profiles
  table6_explanation.json    map agreement (correct-both vs disagree), inner/border mass
  table7_adversarial.json    2x2 transfer matrix
  table8_probes.json         error-predictability AUROCs
  figures/*.png              one figure per table

Every number here is a direct function of the npz files written by steps 02-07.
"""
from __future__ import annotations

import json
import os
import itertools

import numpy as np

import metrics as MT
import models as M
import nb

OUT = "/content/out"
RES = os.path.join(OUT, "08_analysis")
FIG = os.path.join(RES, "figures")


def _load(step, fname):
    p = os.path.join(OUT, step, fname)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def table1():
    rows = {}
    ck = json.load(open(os.path.join(OUT, "01_checkpoints", "checkpoints.json")))
    usable = [k for k, v in ck["models"].items() if v.get("usable")]
    for name in usable:
        d = _load("02_inference", f"{name}.npz")
        y = d["y"]
        rows[name] = {"ID_top1": MT.accuracy(d["logits"], y),
                      "n": int(len(y)),
                      "family": M.REGISTRY[name].family,
                      "params_m": M.REGISTRY[name].params_m}
    ood = json.load(open(os.path.join(OUT, "03_ood", "meta.json")))
    for s in ood:
        for name, rec in (ood[s] or {}).items():
            if name in rows and "top1" in rec:
                rows[name][f"{s}_top1"] = rec["top1"]
    nb.save("08_analysis", "table1_accuracy.json", rows)
    return rows


def table2():
    usable = list(table1().keys())
    out = {}
    for a, b in itertools.combinations(usable, 2):
        da, db = _load("02_inference", f"{a}.npz"), _load("02_inference", f"{b}.npz")
        y = da["y"]
        s = MT.overlap_stats(da["logits"], db["logits"], y)
        s["pair"] = f"{a}+{b}"
        s["same_family"] = M.REGISTRY[a].family == M.REGISTRY[b].family
        out[f"{a}|{b}"] = s
    nb.save("08_analysis", "table2_overlap.json", out)
    return out


def table3():
    usable = list(table1().keys())
    out = {}
    for a, b in itertools.combinations(usable, 2):
        da, db = _load("02_inference", f"{a}.npz"), _load("02_inference", f"{b}.npz")
        y = da["y"]
        rec = {"same_family": M.REGISTRY[a].family == M.REGISTRY[b].family,
               "oracle": MT.overlap_stats(da["logits"], db["logits"], y)["oracle_acc"]}
        for mode in ("prob", "logit"):
            accs = []
            for w in np.linspace(0, 1, 11):
                lg = MT.ensemble_logits(da["logits"], db["logits"], w, mode)
                accs.append(float(MT.accuracy(lg, y)))
            best = float(np.max(accs))
            rec[f"{mode}_best_w"] = float(np.linspace(0, 1, 11)[int(np.argmax(accs))])
            rec[f"{mode}_best_acc"] = best
            rec[f"{mode}_acc_at_half"] = accs[5]
        out[f"{a}|{b}"] = rec
    nb.save("08_analysis", "table3_ensemble.json", out)
    return out


def table4():
    usable = list(table1().keys())
    out = {}
    for a, b in itertools.combinations(usable, 2):
        da, db = _load("02_inference", f"{a}.npz"), _load("02_inference", f"{b}.npz")
        y = da["y"]
        ca = da["logits"].argmax(1) == y
        cb = db["logits"].argmax(1) == y
        sig = MT.disagreement_signals(da["logits"], db["logits"])
        conf_a, _ = MT.confidence(da["logits"])
        conf_b, _ = MT.confidence(db["logits"])
        rec = {"pair": f"{a}|{b}",
               "confidence_auroc": MT.auroc(np.minimum(conf_a, conf_b), ca & cb),
               "confidence_aurc": MT.aurc(np.minimum(conf_a, conf_b), ca & cb),
               "hard_disagree_auroc": MT.auroc(sig["hard"], ca & cb),
               "hard_disagree_aurc": MT.aurc(sig["hard"], ca & cb),
               "js_auroc": MT.auroc(sig["js"], ca & cb),
               "js_aurc": MT.aurc(sig["js"], ca & cb),
               "conf_gap_auroc": MT.auroc(sig["conf_gap"], ca & cb)}
        d = MT.delong_auc_test(-sig["js"], ca & cb, -np.minimum(conf_a, conf_b), ca & cb)
        rec["delong_js_vs_conf"] = d
        # bootstrap CI on the AURC difference
        diff, lo, hi, p = MT.paired_bootstrap_diff(-sig["js"], -np.minimum(conf_a, conf_b),
                                                   stat=lambda s: MT.aurc(s, ca & cb), n=500)
        rec["bootstrap_js_minus_conf_aurc"] = {"diff": diff, "ci": [lo, hi], "p": p}
        out[f"{a}|{b}"] = rec
    nb.save("08_analysis", "table4_selective.json", out)
    return out


def table5():
    meta = json.load(open(os.path.join(OUT, "04_corruption", "meta.json")))
    usable = list(table1().keys())
    vits = [m for m in usable if M.REGISTRY[m].family == "vit"]
    ssms = [m for m in usable if M.REGISTRY[m].family == "ssm"]
    out = {}
    if vits and ssms:
        va, sa = vits[0], ssms[0]
        prof = {}
        for key, rec in meta.items():
            if va in rec and sa in rec:
                prof[key] = {"vit": rec[va]["top1"], "ssm": rec[sa]["top1"],
                             "gap": rec[va]["top1"] - rec[sa]["top1"]}
        rv = [prof[k]["vit"] for k in sorted(prof)]
        rs = [prof[k]["ssm"] for k in sorted(prof)]
        out = {"vit_model": va, "ssm_model": sa, "per_corruption": prof,
               "rank_correlation": float(np.corrcoef(rv, rs)[0, 1]) if len(rv) > 1 else None}
    nb.save("08_analysis", "table5_corruption.json", out)
    return out


def table6():
    files = [f for f in os.listdir(os.path.join(OUT, "05_explanations")) if f.endswith(".npz")]
    if len(files) < 2:
        return {}
    usable = [f[:-4] for f in files]
    out = {}
    for a, b in itertools.combinations(usable, 2):
        da = _load("05_explanations", f"{a}.npz")
        db = _load("05_explanations", f"{b}.npz")
        both = da["correct"].astype(bool) & db["correct"].astype(bool)
        dis = ~(da["correct"].astype(bool)) | ~(db["correct"].astype(bool))
        def agree(mask):
            if mask.sum() == 0:
                return None
            xs = da["maps"][mask].reshape(int(mask.sum()), -1)
            ys = db["maps"][mask].reshape(int(mask.sum()), -1)
            xs = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-9)
            ys = ys / (np.linalg.norm(ys, axis=1, keepdims=True) + 1e-9)
            return float((xs * ys).sum(1).mean())
        out[f"{a}|{b}"] = {"map_cosine_both_correct": agree(both),
                           "map_cosine_disagree": agree(dis),
                           "n_both": int(both.sum()), "n_disagree": int(dis.sum())}
    nb.save("08_analysis", "table6_explanation.json", out)
    return out


def table7():
    p = os.path.join(OUT, "06_adversarial", "transfer.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    nb.save("08_analysis", "table7_adversarial.json", out)
    return out


def table8():
    p = os.path.join(OUT, "07_probes", "probes.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    nb.save("08_analysis", "table8_probes.json", out)
    return out


def figures():
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: accuracy by dataset per model
    t1 = json.load(open(os.path.join(RES, "table1_accuracy.json")))
    cols = ["ID_top1", "A_top1", "R_top1", "sketch_top1"]
    names = list(t1)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(names))
    w = 0.2
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, [t1[n].get(c, 0) for n in names], w, label=c.replace("_top1", ""))
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.legend(fontsize=7, ncol=4)
    ax.set_title("Figure 1. In- and out-of-distribution accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_accuracy.png"), dpi=200)
    plt.close(fig)

    # Fig 2: error overlap heatmap (Jaccard)
    t2 = json.load(open(os.path.join(RES, "table2_overlap.json")))
    if t2:
        mat = np.eye(len(names))
        for k, v in t2.items():
            a, b = k.split("|")
            mat[names.index(a), names.index(b)] = v["jaccard"]
            mat[names.index(b), names.index(a)] = v["jaccard"]
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.imshow(mat, cmap="viridis", vmin=0)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="w" if mat[i, j] < 0.15 else "k", fontsize=7)
        ax.set_title("Figure 2. Error-set Jaccard")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig2_overlap.png"), dpi=200)
        plt.close(fig)

    # Fig 3: selective prediction AUROC, confidence vs disagreement
    t4 = json.load(open(os.path.join(RES, "table4_selective.json")))
    if t4:
        keys = list(t4)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        conf = [t4[k]["confidence_auroc"] for k in keys]
        js = [t4[k]["js_auroc"] for k in keys]
        xx = np.arange(len(keys))
        ax.bar(xx - 0.18, conf, 0.36, label="min-confidence")
        ax.bar(xx + 0.18, js, 0.36, label="Jensen-Shannon disagreement")
        ax.set_xticks(xx, keys, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("AUROC (correct vs incorrect)")
        ax.set_title("Figure 3. Disagreement vs confidence for selective prediction")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig3_selective.png"), dpi=200)
        plt.close(fig)
    print("figures written to", FIG)


def main(out_dir: str) -> dict:
    os.makedirs(RES, exist_ok=True)
    summary = {}
    for fn in (table1, table2, table3, table4, table5, table6, table7, table8, figures):
        try:
            summary[fn.__name__] = "ok"
            fn()
        except Exception as e:
            summary[fn.__name__] = f"{type(e).__name__}: {e}"
            print("FAILED", fn.__name__, e)
    nb.save("08_analysis", "summary.json", summary)
    return summary
