"""
Build the train/val/test split for Stage 3, once, deterministically.

Two hard requirements from project.md, plus one added after measuring the
data.

1. GROUP ON kepid. Multi-planet systems put several KOIs on one star. A
   random row split leaks that star's systematics and stellar parameters
   across the train/test boundary and invalidates every number downstream.

2. THE TEST SET IS TOUCHED EXACTLY ONCE, at the very end. Model selection
   uses val. This script writes the split to disk so that boundary is a
   file on disk, not a promise in a notebook.

3. STRATIFY BY SNR BAND. Stage 4 reports PR-AUC separately for SNR <3,
   3-7 and >7. A plain GroupShuffleSplit can leave a band too thin in the
   test set to say anything about. Measured on this data the bands are
   n=2441 / 1778 / 1828, so none is rare - but the split must guarantee
   the property rather than get lucky with it.

Stratification is on the JOINT key (label, snr_band) = 6 strata, so class
balance is preserved inside every band. Stratifying on SNR alone could let
the planet fraction drift between splits, which matters because the base
rate is already shifted 36.2% -> 39.7% by the drop bias.

Usage:
    python make_splits.py [--seed 0] [--test 0.2] [--val 0.2]
"""

import argparse
import sys

import h5py
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

H5 = "views.h5"
OUT = "splits.npz"
SNR_EDGES = [3.0, 7.0]          # project.md's Stage 4 bands


def snr_band(snr):
    """0 = SNR<3, 1 = 3-7, 2 = >7."""
    return np.digitize(snr, SNR_EDGES)


def one_fold(strata, groups, n_splits, seed):
    """Return (rest_idx, held_idx) taking a single fold as the held-out set."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    rest, held = next(iter(sgkf.split(np.zeros(len(strata)), strata, groups)))
    return rest, held


def main(seed, test_frac, val_frac):
    with h5py.File(H5, "r") as f:
        kepid = f["kepid"][:]
        label = f["label"][:].astype(int)
        snr = f["snr"][:]
        name = f["name"].asstr()[:]

    n = len(label)
    band = snr_band(snr)
    strata = label * 3 + band              # 6 joint strata

    print(f"[data] {n} views, {len(np.unique(kepid))} stars")
    print(f"[data] SNR bands  <3: {(band==0).sum()}  3-7: {(band==1).sum()}  "
          f">7: {(band==2).sum()}")

    # --- test split -------------------------------------------------------
    n_test = int(round(1 / test_frac))
    rest_i, test_i = one_fold(strata, kepid, n_test, seed)

    # --- val split, carved out of the remainder ---------------------------
    # val_frac is a fraction of the WHOLE set, so convert to a fraction of
    # the remainder before choosing the fold count.
    val_of_rest = val_frac / (1.0 - test_frac)
    n_val = int(round(1 / val_of_rest))
    sub_rest, sub_val = one_fold(strata[rest_i], kepid[rest_i], n_val, seed)
    train_i = rest_i[sub_rest]
    val_i = rest_i[sub_val]

    splits = {"train": train_i, "val": val_i, "test": test_i}

    # --- verification. These are the checks that make the split trustworthy.
    print()
    ok = True

    # 1. every row assigned exactly once
    allidx = np.concatenate(list(splits.values()))
    if len(allidx) != n or len(np.unique(allidx)) != n:
        print(f"[FAIL] partition: {len(allidx)} assigned, {n} rows")
        ok = False
    else:
        print(f"[ok]   partition covers all {n} rows exactly once")

    # 2. THE CRITICAL ONE: no star may appear in two splits
    gsets = {k: set(kepid[v].tolist()) for k, v in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = gsets[a] & gsets[b]
        if shared:
            print(f"[FAIL] {len(shared)} kepids shared between {a} and {b}")
            ok = False
    if ok:
        print("[ok]   no kepid appears in more than one split")

    # 3. every stratum populated everywhere, or Stage 4 cannot report on it
    print()
    hdr = f"{'split':<6}{'n':>6}{'stars':>7}{'planet%':>9}" \
          f"{'SNR<3':>8}{'SNR3-7':>8}{'SNR>7':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k, v in splits.items():
        b = band[v]
        print(f"{k:<6}{len(v):>6}{len(gsets[k]):>7}"
              f"{100*label[v].mean():>8.1f}%"
              f"{(b==0).sum():>8}{(b==1).sum():>8}{(b==2).sum():>8}")
        for bi in range(3):
            if (b == bi).sum() == 0:
                print(f"[FAIL] {k} has no SNR band {bi}")
                ok = False

    # 4. base rate must not drift between splits
    rates = {k: label[v].mean() for k, v in splits.items()}
    spread = max(rates.values()) - min(rates.values())
    print(f"\n[base] planet fraction spread across splits: {spread*100:.2f} pp")
    if spread > 0.03:
        print("[WARN] base rate drifts >3pp between splits")

    if not ok:
        print("\n[FAIL] split is not usable")
        sys.exit(1)

    np.savez(OUT, train=train_i, val=val_i, test=test_i, seed=seed,
             snr_edges=np.array(SNR_EDGES))
    print(f"\n[done] wrote {OUT} (seed={seed})")
    print("[done] test set must be touched exactly once, at the very end")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--val", type=float, default=0.2)
    a = ap.parse_args()
    main(a.seed, a.test, a.val)
