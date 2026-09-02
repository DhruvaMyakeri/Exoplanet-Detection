"""
Stage 1 - Exoplanet vetting baseline on the Kepler KOI catalogue.

Trains a Random Forest to separate CONFIRMED planets from FALSE POSITIVEs
using only physically meaningful measurements - no vetting-pipeline flags.

Usage:
    python koi_stage1.py              # honest baseline
    python koi_stage1.py --leaky      # same model, with vetting flags added
    python koi_stage1.py --plot       # also write pr_curve.png / importance.png

The --leaky run exists so you can see the difference for yourself. Run both.
"""

import argparse
import io
import os

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CACHE = "koi_cumulative.csv"

# Columns we pull. Note what is here and what is deliberately NOT:
#   pulled     : measured transit properties + host star properties
#   pulled but only used with --leaky : koi_fpflag_*, koi_score
#   never used : koi_pdisposition (the pipeline's own guess at the label)
COLUMNS = [
    "kepid", "kepoi_name", "koi_disposition",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_score",
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_impact",
    "koi_model_snr", "koi_teq", "koi_insol", "koi_num_transits",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]


def fetch_koi(cache=CACHE):
    """Download the cumulative KOI table from the NASA Exoplanet Archive."""
    if os.path.exists(cache):
        print(f"[data] using cached {cache}")
        return pd.read_csv(cache)

    query = "select " + ",".join(COLUMNS) + " from cumulative"
    print("[data] querying NASA Exoplanet Archive ...")
    r = requests.get(TAP_URL, params={"query": query, "format": "csv"},
                     timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(cache, index=False)
    print(f"[data] cached {len(df)} rows to {cache}")
    return df


# --------------------------------------------------------------------------
# 2. Physics features
# --------------------------------------------------------------------------

G_CGS = 6.674e-8        # cm^3 g^-1 s^-2
R_SUN_CM = 6.957e10     # cm


def expected_duration_hours(period_days, logg_cgs, srad_solar):
    """
    Transit duration for a central transit on a circular orbit.

        (a/R*)^3 = G rho* P^2 / (3 pi)
        T        = (P / pi) * (R* / a)

    Stellar density comes from surface gravity and radius:
        g = GM/R^2  and  rho = 3M / (4 pi R^3)   =>   rho = 3g / (4 pi G R)

    Check: rho_sun = 1.41 g/cm^3, P = 365.25 d  ->  ~13.0 h, which is
    Earth's real transit duration. If you change this function, re-run
    that check.
    """
    g = 10.0 ** logg_cgs                       # cm/s^2
    R = srad_solar * R_SUN_CM                  # cm
    rho = 3.0 * g / (4.0 * np.pi * G_CGS * R)  # g/cm^3
    P = period_days * 86400.0                  # s
    T = (P / np.pi) * (3.0 * np.pi / (G_CGS * rho * P ** 2)) ** (1.0 / 3.0)
    return T / 3600.0


def add_features(df):
    df = df.copy()
    df["t_exp_hours"] = expected_duration_hours(
        df["koi_period"], df["koi_slogg"], df["koi_srad"])

    # The physics-consistency feature. ~0 for a real planet, far from 0 for
    # a binary caught at the wrong period or a blend on the wrong star.
    ratio = df["koi_duration"] / df["t_exp_hours"]
    df["log_dur_ratio"] = np.log10(ratio.where(ratio > 0))

    # Log-scale everything that spans orders of magnitude, otherwise the
    # split thresholds all pile up near zero.
    df["log_period"] = np.log10(df["koi_period"].clip(lower=1e-3))
    df["log_depth"] = np.log10(df["koi_depth"].clip(lower=1.0))
    df["log_prad"] = np.log10(df["koi_prad"].clip(lower=1e-2))
    df["log_snr"] = np.log10(df["koi_model_snr"].clip(lower=1e-1))
    df["log_insol"] = np.log10(df["koi_insol"].clip(lower=1e-4))
    df["log_ntransits"] = np.log10(df["koi_num_transits"].clip(lower=1))

    # Depth implied by the fitted planet radius vs the depth actually
    # measured. Large mismatch is a dilution/blend signature.
    implied_ppm = (df["koi_prad"] * 6371.0 /
                   (df["koi_srad"] * 695700.0)) ** 2 * 1e6
    ratio_d = df["koi_depth"] / implied_ppm
    df["log_depth_resid"] = np.log10(
        ratio_d.where((implied_ppm > 0) & (df["koi_depth"] > 0)))

    # Belt and braces: any surviving +/-inf becomes NaN for the imputer.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


SAFE_FEATURES = [
    "log_period", "log_depth", "log_prad", "log_snr", "log_insol",
    "log_ntransits", "log_dur_ratio", "log_depth_resid",
    "koi_duration", "koi_impact", "koi_teq",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

LEAKY_FEATURES = [
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_score",
]


# --------------------------------------------------------------------------
# 3. Train / evaluate
# --------------------------------------------------------------------------

def main(leaky=False, plot=False, seed=0):
    raw = fetch_koi()
    df = add_features(raw)

    print("\n[labels] disposition counts:")
    print(df["koi_disposition"].value_counts().to_string())

    # CANDIDATE is not a third class - it means "not yet decided".
    # Hold it out entirely and predict on it at the end.
    train_mask = df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    labelled = df[train_mask].copy()
    candidates = df[~train_mask].copy()

    y = (labelled["koi_disposition"] == "CONFIRMED").astype(int).values
    groups = labelled["kepid"].values          # <- the important bit
    features = SAFE_FEATURES + (LEAKY_FEATURES if leaky else [])
    X = labelled[features].values

    print(f"\n[setup] {len(y)} labelled rows, {y.sum()} confirmed, "
          f"{len(y) - y.sum()} false positives")
    print(f"[setup] {len(np.unique(groups))} distinct host stars")
    print(f"[setup] {len(features)} features"
          + ("  ***LEAKY MODE***" if leaky else ""))

    # Split by HOST STAR, not by row. Multi-planet systems put several KOIs
    # on the same star; a random row split leaks that star's systematics,
    # its stellar parameters, and often its label into both sides.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(splitter.split(X, y, groups))

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )),
    ])
    model.fit(X[tr], y[tr])

    p = model.predict_proba(X[te])[:, 1]
    print(f"\n[eval] ROC-AUC : {roc_auc_score(y[te], p):.4f}")
    print(f"[eval] PR-AUC  : {average_precision_score(y[te], p):.4f}")
    print("\n[eval] at threshold 0.50:")
    print(classification_report(y[te], (p >= 0.5).astype(int),
                                target_names=["false positive", "planet"],
                                digits=3))
    print("[eval] confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y[te], (p >= 0.5).astype(int)))

    # Vetting is a precision game: a follow-up telescope night is expensive,
    # so pick the threshold that buys you 95% precision and report the
    # recall you get for it.
    prec, rec, thr = precision_recall_curve(y[te], p)
    ok = np.where(prec[:-1] >= 0.95)[0]
    if len(ok):
        i = ok[np.argmax(rec[:-1][ok])]
        print(f"\n[eval] at 95% precision: threshold={thr[i]:.3f}, "
              f"recall={rec[i]:.3f}")

    # Permutation importance, not impurity importance. Impurity importance
    # inflates high-cardinality continuous features regardless of whether
    # they help.
    print("\n[importance] permutation importance on the held-out set:")
    imp = permutation_importance(model, X[te], y[te], n_repeats=10,
                                 random_state=seed, n_jobs=-1,
                                 scoring="average_precision")
    order = np.argsort(imp.importances_mean)[::-1]
    for i in order:
        print(f"  {features[i]:<20s} {imp.importances_mean[i]:+.4f} "
              f"+/- {imp.importances_std[i]:.4f}")

    # What does the model think of the undecided ones?
    if len(candidates):
        cp = model.predict_proba(candidates[features].values)[:, 1]
        print(f"\n[candidates] {len(cp)} CANDIDATE rows scored")
        print(f"  p >= 0.9 : {(cp >= 0.9).sum()}")
        print(f"  p <= 0.1 : {(cp <= 0.1).sum()}")
        out = candidates[["kepoi_name", "koi_period", "koi_prad",
                          "koi_depth"]].copy()
        out["planet_prob"] = cp
        out.sort_values("planet_prob", ascending=False).to_csv(
            "candidate_scores.csv", index=False)
        print("  -> candidate_scores.csv")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec)
        ax.axhline(y[te].mean(), ls="--", lw=1, c="grey")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(f"PR curve (AP={average_precision_score(y[te], p):.3f})")
        fig.tight_layout(); fig.savefig("pr_curve.png", dpi=150)

        fig, ax = plt.subplots(figsize=(6, 5))
        k = order[::-1]
        ax.barh([features[i] for i in k], imp.importances_mean[k])
        ax.set_xlabel("drop in average precision when shuffled")
        fig.tight_layout(); fig.savefig("importance.png", dpi=150)
        print("\n[plot] wrote pr_curve.png and importance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaky", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(leaky=a.leaky, plot=a.plot, seed=a.seed)