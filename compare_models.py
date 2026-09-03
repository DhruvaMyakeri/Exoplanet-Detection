"""
Stage 4 item 3: CNN vs Random Forest on the SAME group split and the SAME
test set.

WHY THIS SCRIPT EXISTS
----------------------
project.md gives the RF baseline as PR-AUC 0.947, and Stage 3 asks the CNN
to beat it. But that 0.947 was measured on a different population (all
catalogue rows, before view-building dropped 10.1% of KOIs with a strong
class bias) under a different split. Comparing the CNN's number to it
directly would attribute to the MODEL a gap that could equally come from
the data or the split.

So the RF is refit here on exactly the rows that survived into views.h5,
using exactly splits.npz. Both models then see the same 4057 training
examples and are scored on the same 1387 test examples.

Features and hyperparameters are taken from Koi_stage1.py unchanged,
including its exclusion of koi_fpflag_* and koi_score - those are outputs
of the vetting process that produced the labels and leak it outright.

Feature source is koi_cumulative.csv because koi_ephem.csv lacks koi_teq
and koi_insol. The ephemeris columns are irrelevant to the RF.
"""

import json

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

import Koi_stage1 as K

H5 = "views.h5"
SPLITS = "splits.npz"
FEAT_CSV = "koi_cumulative.csv"


def build_matrix():
    """Feature matrix aligned row-for-row with views.h5."""
    with h5py.File(H5, "r") as f:
        names = list(f["name"].asstr()[:])
        y = f["label"][:].astype(int)
        snr = f["snr"][:]
        kepid = f["kepid"][:]

    df = K.add_features(pd.read_csv(FEAT_CSV))
    df = df.drop_duplicates(subset="kepoi_name").set_index("kepoi_name")

    missing = [n for n in names if n not in df.index]
    if missing:
        print(f"[warn] {len(missing)} views have no catalogue row "
              f"(e.g. {missing[:3]}) - dropped from the comparison")

    keep = np.array([n in df.index for n in names])
    sub = df.loc[[n for n in names if n in df.index]]
    X = sub[K.SAFE_FEATURES].to_numpy(dtype=float)
    return X, y[keep], snr[keep], kepid[keep], keep, names


def main():
    X, y, snr, kepid, keep, names = build_matrix()
    s = np.load(SPLITS)

    # splits.npz indexes views.h5 rows; remap onto the kept subset.
    remap = -np.ones(len(keep), dtype=int)
    remap[np.where(keep)[0]] = np.arange(keep.sum())
    idx = {k: remap[s[k]][remap[s[k]] >= 0] for k in ["train", "val", "test"]}

    print(f"[data] {keep.sum()} of {len(keep)} views have catalogue features")
    print(f"[data] train {len(idx['train'])}  val {len(idx['val'])}  "
          f"test {len(idx['test'])}")
    print(f"[data] {len(K.SAFE_FEATURES)} features, "
          f"leaky columns excluded: {K.LEAKY_FEATURES}")

    band = np.digitize(snr, [3.0, 7.0])
    rows = []
    preds = {}
    for seed in [0, 1, 2]:
        # Fit on train only - the same examples the CNN fit on. The CNN
        # additionally used val for early stopping; the RF needs no such
        # set, so giving it train+val would hand it more data, not less
        # bias.
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=600, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=seed)),
        ])
        model.fit(X[idx["train"]], y[idx["train"]])
        p = model.predict_proba(X[idx["test"]])[:, 1]
        t = y[idx["test"]]
        preds[f"seed{seed}"] = p

        strat = {}
        tb = band[idx["test"]]
        for bi, nm in [(0, "snr<3"), (1, "snr3-7"), (2, "snr>7")]:
            sel = tb == bi
            if sel.sum() > 10 and len(np.unique(t[sel])) > 1:
                strat[nm] = float(average_precision_score(t[sel], p[sel]))
        rows.append(dict(seed=seed,
                         pr_auc=float(average_precision_score(t, p)),
                         roc_auc=float(roc_auc_score(t, p)), strat=strat))
        print(f"  [seed {seed}] RF  test PR-AUC {rows[-1]['pr_auc']:.4f}  "
              f"ROC-AUC {rows[-1]['roc_auc']:.4f}")

    pr = np.array([r["pr_auc"] for r in rows])
    rc = np.array([r["roc_auc"] for r in rows])

    print("\n" + "=" * 64)
    print("MATCHED COMPARISON - same split, same test set, same train rows")
    print("=" * 64)
    print(f"RF   TEST PR-AUC  {pr.mean():.4f} +/- {pr.std():.4f}")
    print(f"RF   TEST ROC-AUC {rc.mean():.4f} +/- {rc.std():.4f}")

    try:
        cnn = json.load(open("cnn_results.json"))
        cpr = np.array([c["test"]["pr_auc"] for c in cnn])
        crc = np.array([c["test"]["roc_auc"] for c in cnn])
        print(f"CNN  TEST PR-AUC  {cpr.mean():.4f} +/- {cpr.std():.4f}")
        print(f"CNN  TEST ROC-AUC {crc.mean():.4f} +/- {crc.std():.4f}")
        d = cpr.mean() - pr.mean()
        pooled = np.sqrt(pr.std() ** 2 + cpr.std() ** 2) or 1e-9
        print(f"\ndelta (CNN - RF) PR-AUC: {d:+.4f}   "
              f"({abs(d)/pooled:.1f} pooled sd)")
        print(f"project.md quoted RF baseline: 0.947 "
              f"(different data and split - not comparable)")
        print("\nby SNR band:")
        for b in ["snr<3", "snr3-7", "snr>7"]:
            r = np.array([x["strat"][b] for x in rows if b in x["strat"]])
            c = np.array([x["strat"][b] for x in cnn if b in x["strat"]])
            if len(r) and len(c):
                print(f"  {b:8s} RF {r.mean():.4f}   CNN {c.mean():.4f}   "
                      f"delta {c.mean()-r.mean():+.4f}")
    except FileNotFoundError:
        print("[warn] cnn_results.json not found - run train_cnn.py first")

    np.savez("rf_test_preds.npz", test_idx=idx["test"],
             y_true=y[idx["test"]], **preds)
    with open("rf_results.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n[done] wrote rf_results.json, rf_test_preds.npz")


if __name__ == "__main__":
    main()
