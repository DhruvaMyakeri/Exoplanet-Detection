"""
Stage 1 - Pull one real Kepler light curve and see a transit with your own eyes.

Target: Kepler-10 (KIC 11904151). Kepler-10b has a ~152 ppm transit on a
~0.84 day period. Invisible in the raw data, unmistakable once folded.

Usage:
    python lc_stage1.py                      # Kepler-10, the guided tour
    python lc_stage1.py --target "Kepler-8"  # try another star
    python lc_stage1.py --pmax 15            # widen the BLS period search

Writes stage1_<target>.png. Open it and look at all four panels.
"""

import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightkurve as lk


# --------------------------------------------------------------------------
# 1. Download and stitch
# --------------------------------------------------------------------------

def load_light_curve(target):
    """
    Kepler observed in ~90 day 'quarters', rolling the spacecraft between
    each one. Every quarter lands the star on a different detector, so the
    raw flux level jumps between quarters. stitch() divides each quarter by
    its own median, which puts them all on a common scale near 1.0.
    """
    print(f"[dl] searching for {target} ...")
    sr = lk.search_lightcurve(target, mission="Kepler", author="Kepler",
                              cadence="long")
    if len(sr) == 0:
        raise SystemExit(f"no Kepler long-cadence data found for {target}")
    print(f"[dl] {len(sr)} quarters found, downloading "
          f"(first run only; cached afterwards) ...")

    t0 = time.time()
    collection = sr.download_all()
    lc = collection.stitch()
    print(f"[dl] done in {time.time() - t0:.0f}s")

    # stitch() uses PDCSAP_FLUX by default: the pipeline has already removed
    # instrumental systematics common to thousands of stars on the same
    # detector. It has NOT removed this star's own variability.
    n0 = len(lc)
    lc = lc.remove_nans()

    # Sigma-clip asymmetrically. A symmetric 5-sigma clip would happily
    # delete transits, because a transit IS a downward outlier. Clip hard
    # above (cosmic rays, flares), barely at all below.
    lc = lc.remove_outliers(sigma_upper=4, sigma_lower=20)
    print(f"[dl] {n0} cadences -> {len(lc)} after cleaning")
    print(f"[dl] baseline: {lc.time.value.max() - lc.time.value.min():.1f} days")
    return lc


# --------------------------------------------------------------------------
# 2. Detrend
# --------------------------------------------------------------------------

def flatten(lc, window_cadences=101):
    """
    Savitzky-Golay filter: fit a low-order polynomial in a sliding window,
    divide it out. Removes anything slower than the window.

    window_cadences is in CADENCES, not days. Kepler long cadence is
    29.4 minutes, so 101 cadences = 2.06 days.

    The rule that matters: window must be MUCH longer than a transit.
    Kepler-10b's transit is ~1.9 hours, so 2.06 days is ~26x longer.
    Safe. Set this to 5 and you will divide the transit away and see
    a flat line with no signal - and nothing will warn you.
    """
    flat = lc.flatten(window_length=window_cadences)
    scatter_ppm = np.std(flat.flux.value) * 1e6
    print(f"[flat] window = {window_cadences} cadences "
          f"({window_cadences * 29.4 / 60 / 24:.2f} days)")
    print(f"[flat] point-to-point scatter: {scatter_ppm:.0f} ppm")
    print(f"[flat] a 152 ppm transit sits at {152 / scatter_ppm:.2f}x "
          f"the scatter of a SINGLE point")
    return flat


# --------------------------------------------------------------------------
# 3. Period search
# --------------------------------------------------------------------------

def find_period(flat, pmin=0.5, pmax=5.0, durations=(0.02, 0.05, 0.1, 0.15),
                ff=500.0):
    """
    Two-pass search.

    Pass 1 is a coarse grid over the full range - fine enough to land on
    the right peak, coarse enough to finish. Pass 2 zooms into a narrow
    window around that peak with a dense explicit grid, which is what
    actually pins the period down.

    This is how real pipelines do it. A single grid fine enough for the
    final answer, spanning the whole range, is unaffordable.
    """
    dur = np.array(durations)

    print(f"[bls] pass 1: coarse scan {pmin}-{pmax} d (ff={ff:.0f}) ...")
    t0 = time.time()
    pg = flat.to_periodogram(method="bls", minimum_period=pmin,
                             maximum_period=pmax, duration=dur,
                             frequency_factor=ff)
    p_coarse = pg.period_at_max_power.value
    print(f"[bls] pass 1 done in {time.time() - t0:.0f}s  "
          f"-> {p_coarse:.5f} d")

    # Pass 2: +/-0.2% around the coarse peak, 20001 samples.
    grid = np.linspace(p_coarse * 0.998, p_coarse * 1.002, 20001)
    print(f"[bls] pass 2: refining over {grid[0]:.5f}-{grid[-1]:.5f} d ...")
    t0 = time.time()
    pg_fine = flat.to_periodogram(method="bls", period=grid, duration=dur)
    print(f"[bls] pass 2 done in {time.time() - t0:.0f}s")

    period = pg_fine.period_at_max_power.value
    epoch = pg_fine.transit_time_at_max_power.value
    duration = pg_fine.duration_at_max_power.value
    depth = pg_fine.depth_at_max_power * 1e6

    print(f"[bls] period   = {period:.6f} d")
    print(f"[bls] epoch    = {epoch:.4f} (BKJD)")
    print(f"[bls] duration = {duration * 24:.2f} h")
    print(f"[bls] depth    = {depth:.0f} ppm")

    # Return the COARSE periodogram for plotting - it spans the full range,
    # so you can see the harmonics. The fine one is a single narrow spike.
    return pg, period, epoch, duration


# --------------------------------------------------------------------------
# 4. Fold and plot
# --------------------------------------------------------------------------

def make_figure(lc, flat, pg, period, epoch, duration, target):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # Panel 1: raw. You will see slow waves. You will NOT see the planet.
    ax[0, 0].plot(lc.time.value, lc.flux.value, "k.", ms=0.4, alpha=0.4)
    ax[0, 0].set_title("1. Raw PDCSAP - starspots dominate")
    ax[0, 0].set_xlabel("time (BKJD, days)")
    ax[0, 0].set_ylabel("normalised flux")

    # Panel 2: BLS power spectrum. One spike = one period the data likes.
    ax[0, 1].plot(pg.period.value, pg.power.value, "k-", lw=0.6)
    ax[0, 1].axvline(period, color="C1", ls="--", lw=1)
    ax[0, 1].set_title(f"2. BLS periodogram - peak at {period:.5f} d")
    ax[0, 1].set_xlabel("trial period (days)")
    ax[0, 1].set_ylabel("BLS power")

    # Panel 3: folded, binned. This is where the planet appears.
    fold = flat.fold(period=period, epoch_time=epoch)
    ax[1, 0].plot(fold.time.value * 24, fold.flux.value, "k.", ms=0.3,
                  alpha=0.15)
    binned = fold.bin(time_bin_size=duration / 8)
    ax[1, 0].plot(binned.time.value * 24, binned.flux.value, "o",
                  color="C1", ms=3)
    ax[1, 0].set_xlim(-duration * 24 * 4, duration * 24 * 4)
    ax[1, 0].set_title("3. Folded on the BLS period (local view)")
    ax[1, 0].set_xlabel("hours from mid-transit")
    ax[1, 0].set_ylabel("normalised flux")

    # Panel 4: the full orbit. Look for a SECOND dip at phase 0.5 -
    # that would be a secondary eclipse, i.e. a binary star, not a planet.
    full = flat.fold(period=period, epoch_time=epoch)
    fb = full.bin(bins=501)
    ax[1, 1].plot(fb.time.value / period, fb.flux.value, "k-", lw=0.7)
    ax[1, 1].axvline(0.5, color="C3", ls=":", lw=1)
    ax[1, 1].axvline(-0.5, color="C3", ls=":", lw=1)
    ax[1, 1].set_title("4. Full phase - check for a secondary eclipse")
    ax[1, 1].set_xlabel("orbital phase")
    ax[1, 1].set_ylabel("normalised flux")

    fig.suptitle(target)
    fig.tight_layout()
    name = f"stage1_{target.replace(' ', '_')}.png"
    fig.savefig(name, dpi=140)
    print(f"\n[plot] wrote {name}")


def main(target, pmin, pmax, window, durations, ff):
    lc = load_light_curve(target)
    flat = flatten(lc, window)
    pg, period, epoch, duration = find_period(flat, pmin, pmax,
                                              durations=durations, ff=ff)
    make_figure(lc, flat, pg, period, epoch, duration, target)

    # Flag a quantised duration: if BLS landed exactly on a grid entry at
    # the edge of the grid, the fit is pinned by the grid, not the data.
    d = np.array(durations)
    if np.isclose(duration, d.min()) or np.isclose(duration, d.max()):
        print(f"\n[warn] duration {duration:.4f} d is at the EDGE of your "
              f"duration grid ({d.min():.3f}-{d.max():.3f} d).")
        print("[warn] widen --durations; the true value may lie outside it.")

    # Compare against the archive. Do NOT take my value on faith - look up
    # the published period for your target and check it yourself.
    if "10" in target:
        published = 0.837491
        err = abs(period - published) / published * 100
        print(f"\n[check] published Kepler-10b period: {published} d")
        print(f"[check] recovered: {period:.6f} d  ({err:.3f}% off)")
        print("[check] if you are off by ~2x or ~0.5x, BLS locked onto a "
              "harmonic - that is normal and worth understanding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Kepler-10")
    ap.add_argument("--pmin", type=float, default=0.5)
    ap.add_argument("--pmax", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=101,
                    help="detrend window in cadences (must be odd)")
    ap.add_argument("--durations", type=float, nargs="+",
                    default=[0.03, 0.045, 0.06, 0.075, 0.09, 0.105, 0.12],
                    help="BLS trial durations in days")
    ap.add_argument("--ff", type=float, default=500.0,
                    help="BLS frequency_factor for the coarse pass; "
                         "raise it if pass 1 is too slow")
    a = ap.parse_args()
    main(a.target, a.pmin, a.pmax, a.window, a.durations, a.ff)