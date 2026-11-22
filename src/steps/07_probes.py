"""Step 07 — error predictability: can one model's representation predict the other's
mistakes, and does that beat the target model's own confidence?

For each ordered pair (A -> B), train a logistic probe on A's penultimate features to
predict "B is wrong", 5-fold CV on the training half of the 10k ID subset, evaluated on
the held-out half. Compare to B's own max-probability confidence AUROC and A's own.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import metrics as MT
import models as M
import nb


def auroc_ci(score, correct, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(score), size=len(score))
        vals[i] = MT.auroc(score[j], correct[j])
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main(out_dir: str) -> dict:
    d = os.path.join(out_dir, "02_inference")
    ck = json.load(open(os.path.join(out_dir, "01_checkpoints", "checkpoints.json")))
    usable = [k for k, v in ck["models"].items() if v.get("usable")]
    data = {m: np.load(os.path.join(d, f"{m}.npz")) for m in usable}
    y = data[usable[0]]["y"]
    results = {}
    for b in usable:
        correct_b = data[b]["logits"].argmax(1) == y
        conf_b, _ = MT.confidence(data[b]["logits"])
        auc_own, lo, hi = auroc_ci(conf_b, correct_b)
        results[b] = {"target": b, "own_conf_auroc": auc_own,
                      "own_conf_ci": [lo, hi], "probes": {}}
        for a in usable:
            if a == b:
                continue
            feats = data[a]["feat"]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            prob = np.zeros(len(y))
            for tr, te in skf.split(feats, correct_b):
                sc = StandardScaler().fit(feats[tr])
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(sc.transform(feats[tr]), correct_b[tr])
                prob[te] = clf.predict_proba(sc.transform(feats[te]))[:, 1]
            auc, lo, hi = auroc_ci(prob, correct_b, seed=1)
            results[b]["probes"][a] = {"probe_auroc": auc, "ci": [lo, hi]}
        print(f"target={b:14s} own-conf AUROC={auc_own:.4f}  " +
              "  ".join(f"probe[{k}]={v['probe_auroc']:.4f}"
                        for k, v in results[b]["probes"].items()), flush=True)

    nb.save("07_probes", "probes.json", results)
    return results
