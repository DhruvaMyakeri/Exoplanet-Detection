"""
Sanity-check views.h5 before committing to the full run.

Two questions:
  1. Do the stored arrays actually look like transits, or is the fold
     shifted and we have saved 46 arrays of noise?
  2. Which KOIs got dropped by the quality filters, and are the drops
     biased toward one class?

Usage:
    python inspect_views.py
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5 = "views.h5"
KOI_CSV = "koi_ephem.csv"


def main():
    f = h5py.File(H5, "r")
    name = f["name"].asstr()[:]
    disp = f["disp"].asstr()[:]
    label = f["label"][:]
    depth = f["depth"][:]
    period = f["period"][:]
    G = f["global"]
    L = f["local"]

    print(f"[data] {len(name)} views  "
          f"({int(label.sum())} planet / {int((label == 0).sum())} FP)")

    # ------------------------------------------------------------------
    # 1. Do the views look like transits?
    #
    # Every view is normalised to minimum -1, so the transit MUST sit at
    # the centre bin of the local view. If the deepest bin is scattered
    # uniformly across the array instead, the fold is broken.
    # ------------------------------------------------------------------
    loc = L[:]
    argmin = loc.argmin(axis=1)
    centre = loc.shape[1] // 2
    off = np.abs(argmin - centre)
    print(f"\n[fold] local-view minimum is at bin {centre} +/- ...")
    print(f"[fold] median offset from centre: {np.median(off):.0f} bins "
          f"(of {loc.shape[1]})")
    print(f"[fold] within 10 bins of centre: "
          f"{(off <= 10).mean() * 100:.0f}% of objects")
    print("[fold] EXPECT >80%. If this is near random (~50 bins median),")
    print("[fold] the epoch or time system is wrong and the data is junk.")

    # Out-of-transit scatter: the wings should be flat near 0.
    wings = np.concatenate([loc[:, :30], loc[:, -30:]], axis=1)
    print(f"\n[noise] median |wing| value: {np.median(np.abs(wings)):.3f}")
    print("[noise] EXPECT well under 1.0 - wings are baseline, not transit.")
    snr = f["snr"][:]
    print(f"\n[snr] median {np.median(snr):.1f}, "
          f"{(snr > 5).mean()*100:.0f}% above 5")
    for lo, hi in [(0, 3), (3, 7), (7, 100)]:
        m = (snr >= lo) & (snr < hi)
        if m.sum():
            print(f"[snr] {lo}-{hi}: n={m.sum():3d}  "
                  f"centred={(off[m] <= 10).mean()*100:3.0f}%")

    # ------------------------------------------------------------------
    # 2. Which objects were dropped, and is the drop class-biased?
    # ------------------------------------------------------------------
    koi = pd.read_csv(KOI_CSV)
    koi = koi[koi.koi_disposition.isin(["CONFIRMED", "FALSE POSITIVE"])]
    koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])

    stars_done = set(f["kepid"][:].tolist())
    attempted = koi[koi.kepid.isin(stars_done)]
    got = set(name)
    dropped = attempted[~attempted.kepoi_name.isin(got)]

    print(f"\n[drop] {len(attempted)} KOIs on the stars processed, "
          f"{len(got)} stored, {len(dropped)} dropped "
          f"({len(dropped) / max(len(attempted), 1) * 100:.0f}%)")
    if len(dropped):
        print("\n[drop] by disposition:")
        print(dropped.koi_disposition.value_counts().to_string())
        print(f"\n[drop] median period of dropped:   "
              f"{dropped.koi_period.median():8.2f} d")
        print(f"[drop] median period of kept:      "
              f"{np.median(period):8.2f} d")
        print(f"[drop] median depth of dropped:    "
              f"{dropped.koi_depth.median():8.0f} ppm")
        print(f"[drop] median depth of kept:       "
              f"{np.median(depth[depth > 0]):8.0f} ppm")
        print("\n[drop] longest-period dropped objects:")
        print(dropped.nlargest(5, "koi_period")[
            ["kepoi_name", "koi_disposition", "koi_period",
             "koi_depth"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Plot a few of each class
    # ------------------------------------------------------------------
    n = min(3, int(label.sum()), int((label == 0).sum()))
    if n == 0:
        print("\n[plot] not enough of both classes to plot")
        f.close()
        return

    pl = np.flatnonzero(label == 1)[:n]
    fp = np.flatnonzero(label == 0)[:n]
    fig, ax = plt.subplots(2 * n, 2, figsize=(11, 2.2 * 2 * n))
    for row, (i, kind) in enumerate(
            [(i, "PLANET") for i in pl] + [(i, "FALSE POS") for i in fp]):
        ax[row, 0].plot(np.linspace(-0.5, 0.5, G.shape[1]), G[i], "k-", lw=0.5)
        ax[row, 0].set_ylabel(f"{name[i]}\n{kind}", fontsize=7)
        ax[row, 0].set_title("global" if row == 0 else "", fontsize=8)
        ax[row, 1].plot(np.arange(L.shape[1]), L[i], "k-", lw=0.8)
        ax[row, 1].set_title("local" if row == 0 else "", fontsize=8)
        for a in ax[row]:
            a.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig("views_check.png", dpi=130)
    print("\n[plot] wrote views_check.png")
    print("[plot] every LOCAL panel should show a dip centred at bin 100")
    print("[plot] reaching -1. Planets tend to have flat bottoms; false")
    print("[plot] positives often V-shaped or with a secondary dip in the")
    print("[plot] global view near phase +/-0.25.")
    f.close()


if __name__ == "__main__":
    main()