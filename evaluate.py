"""
Stage 4: honest evaluation, with calibration.

WHY CALIBRATION MATTERS HERE
----------------------------
Everything reported so far has been a RANKING metric. PR-AUC and ROC-AUC
are invariant to any monotone transform of the scores, so a model can top
both while its "0.9" means nothing like 90%. That is fine for choosing
which objects to follow up and useless for saying how confident we are.

Worse, our best number (PR-AUC 0.961) came from a RANK AVERAGE. Ranks are
not probabilities at all - the output is uniform on [0,1] by construction.
It ranks well and cannot be read as confidence.

So: fit calibrators on VAL, apply to TEST, and measure whether the
resulting numbers mean what they say. Fitting on test and reporting on
test would be circular, which is why train_cnn.py and compare_models.py
now save val predictions.

METRICS
-------
  PR-AUC, ROC-AUC          ranking quality (what we had)
  Brier score              squared error of the probability itself
  ECE                      expected calibration error, binned
  reliability curve        predicted vs observed frequency
  recall @ fixed precision the operational number - project.md's Stage 0
                           reports 72.5% recall at 95% precision
  recall @ fixed FPR       the same question from the FP side
  confusion by planet size small planets are the hard, interesting case
"""

import json

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

H5 = "views.h5"


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def platt(vp, vy, tp):
    """Logistic recalibration fitted on val, applied to test."""
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(vp).reshape(-1, 1), vy)
    return lr.predict_proba(_logit(tp).reshape(-1, 1))[:, 1]


def isotonic(vp, vy, tp):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(vp, vy)
    return iso.predict(tp)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece(p, y, bins=15):
    """Expected calibration error: average |confidence - accuracy| over
    equal-width bins, weighted by bin population."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


# ----------------------------------------------------------------------
# Operating points
# ----------------------------------------------------------------------

def recall_at_precision(y, p, target):
    pr, rc, _ = precision_recall_curve(y, p)
    ok = pr >= target
    return float(rc[ok].max()) if ok.any() else 0.0


def recall_at_fpr(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


# ----------------------------------------------------------------------

def main():
    c = np.load("cnn_test_preds.npz")
    r = np.load("rf_test_preds.npz")
    cv = np.load("cnn_val_preds.npz")
    rv = np.load("rf_val_preds.npz")
    assert np.array_equal(c["test_idx"], r["test_idx"])
    assert np.array_equal(cv["val_idx"], rv["val_idx"])

    ti, vi = c["test_idx"], cv["val_idx"]
    y = c["y_true"].astype(int)
    vy = cv["y_true"].astype(int)
    seeds = [0, 1, 2]
    cp = np.mean([c[f"seed{s}"] for s in seeds], axis=0)
    rp = np.mean([r[f"seed{s}"] for s in seeds], axis=0)
    cvp = np.mean([cv[f"seed{s}"] for s in seeds], axis=0)
    rvp = np.mean([rv[f"seed{s}"] for s in seeds], axis=0)

    # Ensemble: rank average on test, and the SAME construction on val so
    # the calibrator sees the same kind of input it will be applied to.
    ens = (rankdata(cp) + rankdata(rp)) / (2 * len(cp))
    vens = (rankdata(cvp) + rankdata(rvp)) / (2 * len(cvp))

    with h5py.File(H5, "r") as f:
        snr = f["snr"][:][ti]
        names = np.array(f["name"].asstr()[:])[ti]
    band = np.digitize(snr, [3.0, 7.0])

    koi = pd.read_csv("koi_cumulative.csv").drop_duplicates("kepoi_name") \
            .set_index("kepoi_name")
    prad = koi.reindex(names)["koi_prad"].to_numpy(dtype=float)

    models = {
        "CNN": (cvp, cp),
        "RF": (rvp, rp),
        "ensemble(rank)": (vens, ens),
    }

    print("=" * 74)
    print("RANKING QUALITY  (invariant to monotone rescaling)")
    print("=" * 74)
    print(f"{'model':<16}{'PR-AUC':>9}{'ROC-AUC':>10}"
          f"{'R@P=0.95':>11}{'R@P=0.99':>11}{'R@FPR=0.01':>12}")
    for nm, (_, tp) in models.items():
        print(f"{nm:<16}{average_precision_score(y, tp):>9.4f}"
              f"{roc_auc_score(y, tp):>10.4f}"
              f"{recall_at_precision(y, tp, 0.95):>11.4f}"
              f"{recall_at_precision(y, tp, 0.99):>11.4f}"
              f"{recall_at_fpr(y, tp, 0.01):>12.4f}")
    print("\nproject.md Stage 0 reference: 72.5% recall at 95% precision")

    print("\n" + "=" * 74)
    print("CALIBRATION  (does 0.9 mean 90%?)   calibrator fitted on VAL")
    print("=" * 74)
    print(f"{'model':<16}{'method':<12}{'Brier':>9}{'ECE':>9}{'PR-AUC':>9}")
    results = {}
    for nm, (vp, tp) in models.items():
        row = {}
        for meth, fn in [("raw", None), ("platt", platt), ("isotonic", isotonic)]:
            q = tp if fn is None else fn(vp, vy, tp)
            row[meth] = dict(brier=float(brier_score_loss(y, q)),
                             ece=ece(q, y),
                             pr_auc=float(average_precision_score(y, q)),
                             rel=reliability(q, y))
            print(f"{nm if meth=='raw' else '':<16}{meth:<12}"
                  f"{row[meth]['brier']:>9.4f}{row[meth]['ece']:>9.4f}"
                  f"{row[meth]['pr_auc']:>9.4f}")
            # Platt, not isotonic. Isotonic is monotone NON-DECREASING: it
            # maps many distinct scores onto identical values, and those
            # ties destroy fine-grained ranking. Measured here it cost
            # PR-AUC on every model (ensemble 0.9607 -> 0.9541). Platt is
            # strictly monotone, so it fixes calibration for free.
            if fn is not None and meth == "platt":
                results.setdefault(nm, {})["cal_test"] = q.tolist()
        results.setdefault(nm, {}).update(row)
        print()

    best = min(models, key=lambda m: results[m]["platt"]["brier"])
    print(f"lowest Brier after Platt: {best}  "
          f"(Brier {results[best]['platt']['brier']:.4f}, "
          f"ECE {results[best]['platt']['ece']:.4f}, "
          f"PR-AUC preserved at {results[best]['platt']['pr_auc']:.4f})")

    print("=" * 74)
    print(f"RELIABILITY - {best}, Platt (predicted vs observed)")
    print("=" * 74)
    print(f"{'predicted':>10}{'observed':>10}{'n':>7}")
    for pm, om, n in results[best]["platt"]["rel"]:
        bar = "#" * int(round(om * 40))
        print(f"{pm:>10.3f}{om:>10.3f}{n:>7}  {bar}")

    print("\n" + "=" * 74)
    print("BY SNR BAND  (Platt-calibrated)")
    print("=" * 74)
    print(f"{'band':<10}{'n':>6}{'planets':>9}"
          + "".join(f"{m:>16}" for m in models))
    for bi, nm in [(0, "snr<3"), (1, "snr3-7"), (2, "snr>7")]:
        m = band == bi
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        cells = ""
        for mod in models:
            q = np.array(results[mod]["cal_test"])
            cells += f"{average_precision_score(y[m], q[m]):>16.4f}"
        print(f"{nm:<10}{m.sum():>6}{int(y[m].sum()):>9}{cells}")

    print("\n" + "=" * 74)
    print("BY PLANET RADIUS  (Platt-calibrated ensemble)")
    print("=" * 74)
    q = np.array(results[best]["cal_test"])
    print(f"{'R_p (Re)':<14}{'n':>6}{'planets':>9}{'PR-AUC':>9}"
          f"{'recall@0.5':>12}{'FP rate':>10}")
    for lo, hi, lbl in [(0, 1.5, "<1.5"), (1.5, 3, "1.5-3"),
                        (3, 8, "3-8"), (8, 1e9, ">8")]:
        m = np.isfinite(prad) & (prad >= lo) & (prad < hi)
        if m.sum() < 15 or len(np.unique(y[m])) < 2:
            continue
        pred = q[m] >= 0.5
        rec = float((pred & (y[m] == 1)).sum() / max((y[m] == 1).sum(), 1))
        fpr_ = float((pred & (y[m] == 0)).sum() / max((y[m] == 0).sum(), 1))
        print(f"{lbl:<14}{m.sum():>6}{int(y[m].sum()):>9}"
              f"{average_precision_score(y[m], q[m]):>9.4f}"
              f"{rec:>12.3f}{fpr_:>10.3f}")

    out = {m: {k: v for k, v in d.items() if k != "cal_test"}
           for m, d in results.items()}
    with open("eval_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez("calibrated_test_preds.npz", test_idx=ti, y_true=y,
             **{m.replace("(", "_").replace(")", ""):
                np.array(results[m]["cal_test"]) for m in models})
    print("\n[done] wrote eval_results.json, calibrated_test_preds.npz")


if __name__ == "__main__":
    main()
