"""Metrics: error overlap, selective prediction, calibration, bootstrap & significance.

Everything works on per-image arrays so that every number in the paper can be recomputed
from the raw logits stored by `inference.py`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------------------
def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(logits, y):
    return float((logits.argmax(1) == y).mean())


def confidence(logits, kind="maxprob"):
    p = softmax(logits)
    if kind == "maxprob":
        return p.max(1), p.argmax(1)
    if kind == "margin":
        s = np.sort(p, 1)
        return s[:, -1] - s[:, -2], p.argmax(1)
    if kind == "entropy":
        e = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
        return -e, p.argmax(1)            # negated so that larger = more confident
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------
# Error overlap
# ---------------------------------------------------------------------------------------
def error_sets(logits_a, logits_b, y):
    ea = logits_a.argmax(1) != y
    eb = logits_b.argmax(1) != y
    return ea, eb


def overlap_stats(logits_a, logits_b, y) -> dict:
    ea, eb = error_sets(logits_a, logits_b, y)
    a, b = ea.mean(), eb.mean()
    both = (ea & eb).mean()
    union = (ea | eb).mean()
    jacc = both / union if union > 0 else float("nan")
    indep = a * b
    # oracle / complementarity: images where exactly one model errs
    only_a = (ea & ~eb).mean()
    only_b = (~ea & eb).mean()
    # phi coefficient (correlation of the two error indicators)
    if 0 < a < 1 and 0 < b < 1:
        cot = ((ea.astype(float) - a) * (eb.astype(float) - b)).mean()
        phi = cot / np.sqrt(a * (1 - a) * b * (1 - b))
    else:
        phi = float("nan")
    return dict(err_a=float(a), err_b=float(b), err_both=float(both),
                err_union=float(union), jaccard=float(jacc),
                independence_expected=float(indep),
                excess_over_independence=float(both - indep),
                phi=float(phi), only_a=float(only_a), only_b=float(only_b),
                oracle_acc=float(1 - union))


# ---------------------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------------------
def aurc(score, correct) -> float:
    """Area under risk-coverage curve. Higher score = more confident."""
    order = np.argsort(-score)
    c = correct[order].astype(float)
    risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(np.trapezoid(risk) / (len(c)))       # equivalent to mean risk over coverages


def risk_at_coverage(score, correct, coverages=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9)) -> dict:
    order = np.argsort(-score)
    c = correct[order].astype(float)
    n = len(c)
    out = {}
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        out[f"risk@{cov:g}"] = float(1 - c[:k].mean())
    return out


def auroc(score, correct) -> float:
    """AUROC of `score` discriminating correct (1) vs incorrect (0)."""
    correct = correct.astype(bool)
    n1, n0 = correct.sum(), (~correct).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[correct].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(probs, correct, n_bins=15) -> float:
    conf = probs.max(1)
    acc = correct.astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(conf, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(probs, y) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return float(((probs - onehot) ** 2).sum(1).mean())


# ---------------------------------------------------------------------------------------
# Disagreement signals
# ---------------------------------------------------------------------------------------
def disagreement_signals(logits_a, logits_b) -> dict[str, np.ndarray]:
    pa, pb = softmax(logits_a), softmax(logits_b)
    ya, yb = pa.argmax(1), pb.argmax(1)
    hard = (ya != yb).astype(float)
    # Jensen-Shannon distance
    m = 0.5 * (pa + pb)
    def kl(p, q):
        return (p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum(1)
    js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)          # in nats, 0..log2
    js = np.sqrt(np.clip(js / np.log(2), 0, 1))
    # L1 distance between posteriors
    l1 = np.abs(pa - pb).sum(1) / 2.0
    # confidence-gap
    gap = np.abs(pa.max(1) - pb.max(1))
    return {"hard": hard, "js": js, "l1": l1, "conf_gap": gap,
            "pa": pa, "pb": pb, "ya": ya, "yb": yb}


# ---------------------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------------------
def ensemble_logits(logits_a, logits_b, w=0.5, mode="prob"):
    if mode == "prob":
        p = w * softmax(logits_a) + (1 - w) * softmax(logits_b)
        return np.log(np.clip(p, 1e-12, None))
    return w * logits_a + (1 - w) * logits_b


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, stat=np.mean, n=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    reps = np.array([stat(values[i]) for i in idx])
    return float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, stat=np.mean, n=2000, seed=0):
    """CI for stat(a) - stat(b) with paired resampling of the same items."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(a), size=len(a))
        diffs[i] = stat(a[j]) - stat(b[j])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(stat(a) - stat(b)), float(lo), float(hi), float(p)


def delong_auc_test(scores1, y1, scores2=None, y2=None):
    """DeLong test for two correlated ROC AUCs computed on the *same* samples."""
    y = y1.astype(bool)
    if scores2 is None:
        # single AUC — no comparison
        return {"auc": auroc(scores1, y)}
    import scipy.stats as st
    n1, n0 = int(y.sum()), int((~y).sum())
    def placement(s):
        pos, neg = s[y], s[~y]
        # midrank
        allv = np.concatenate([pos, neg])
        r = st.rankdata(allv)
        v10 = (r[:n1] - np.arange(1, n1 + 1)) / n0
        v01 = 1.0 - (r[n1:] - np.arange(1, n0 + 1)) / n1
        return v10, v01
    v10a, v01a = placement(scores1)
    v10b, v01b = placement(scores2)
    auc_a = v10a.mean(); auc_b = v10b.mean()
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n1 + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0
    z = (auc_a - auc_b) / np.sqrt(max(var, 1e-12))
    p = 2 * (1 - st.norm.cdf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "z": float(z), "p": float(p)}


def permutation_paired(a, b, stat=np.mean, n=5000, seed=0):
    """Two-sided paired permutation test (sign flips)."""
    rng = np.random.default_rng(seed)
    d = a - b
    obs = stat(d) if stat is not np.mean else d.mean()
    cnt = 0
    for _ in range(n):
        s = rng.choice([-1, 1], size=len(d))
        if abs((d * s).mean()) >= abs(obs):
            cnt += 1
    return float(obs), float((cnt + 1) / (n + 1))


def mcnemar(ea, eb):
    """McNemar exact test on two error vectors."""
    b01 = int((~ea & eb).sum())
    b10 = int((ea & ~eb).sum())
    if b01 + b10 == 0:
        return {"b01": b01, "b10": b10, "p": 1.0}
    p = stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue
    return {"b01": b01, "b10": b10, "p": float(p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, p * (m - i)))
        out[k] = adj
        prev = adj
    return out
