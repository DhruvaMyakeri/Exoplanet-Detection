"""
Phase 2 of 2: build folded views from the local cache written by fetch_all.py.

This does NOT reimplement any of the science. It imports detrend_masked,
make_views, open_store and append from build_views.py unchanged, and only
replaces where the light curve comes from:

    build_views.py        search_lightcurve -> download_all -> stitch
    build_views_local.py  np.load(cache/<kic>.npz)

Everything downstream - masking, Savitzky-Golay detrend, folding, binning,
the depth normalisation, the HDF5 schema - is the same code path.

Because the network is gone, this is pure CPU: 0.66 s/star measured, so
the full catalogue is ~3 minutes across 24 cores instead of ~6 days.
"""

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import build_views as BV          # reuse the validated pipeline

CACHE = "cache"
LOGFILE = "build_views_local.log"


def log(msg):
    line = str(msg)
    print(line, file=sys.__stdout__, flush=True)
    with open(LOGFILE, "a") as fh:
        fh.write(line + "\n")


def _init_worker():
    warnings.filterwarnings("ignore")
    sys.stdout = BV.NullSink()


def load_lc(kepid):
    """Rebuild a lightkurve LightCurve from the cached arrays."""
    import lightkurve as lk
    d = np.load(os.path.join(CACHE, f"{kepid}.npz"))
    return lk.LightCurve(time=d["time"], flux=d["flux"])


def process_star(kepid, rows, window):
    """
    Same contract as build_views.process_star, minus the download.

    remove_outliers is applied here rather than in fetch_all so the cache
    stays raw - if we ever want to change the clipping we do not refetch.
    """
    try:
        lc = load_lc(kepid)
        lc = lc.remove_outliers(sigma_upper=4, sigma_lower=20)
        if len(lc) < 1000:
            return [], f"KIC {kepid}: only {len(lc)} cadences"

        results = []
        for _, r in rows.iterrows():
            try:
                dur_days = r.koi_duration / 24.0
                flat, _ = BV.detrend_masked(lc, r.koi_period, r.koi_time0bk,
                                            dur_days, window)
                v = BV.make_views(flat, r.koi_period, r.koi_time0bk, dur_days)
                if v is None:
                    continue
                g, l, npts, snr = v
                depth = float(r.koi_depth) if np.isfinite(r.koi_depth) else -1.0
                results.append(dict(
                    name=str(r.kepoi_name), kepid=int(kepid),
                    label=int(r.koi_disposition == "CONFIRMED"),
                    disp=str(r.koi_disposition),
                    glob=g, loc=l,
                    period=float(r.koi_period),
                    duration=float(r.koi_duration),
                    depth=depth, npts=npts, snr=float(snr),
                ))
            except Exception as e:
                pass                      # per-KOI failure, not per-star
        return results, None
    except Exception as e:
        return [], f"KIC {kepid}: {type(e).__name__}: {e}"


def main(workers, window, include_candidates, out):
    BV.OUT = out
    koi = pd.read_csv(BV.KOI_CSV)
    keep = ["CONFIRMED", "FALSE POSITIVE"]
    if include_candidates:
        keep.append("CANDIDATE")
    koi = koi[koi.koi_disposition.isin(keep)]
    koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])

    have = {int(x[:-4]) for x in os.listdir(CACHE) if x.endswith(".npz")}
    koi = koi[koi.kepid.isin(have)]
    log(f"[plan] {len(have)} stars cached, {len(koi)} KOIs buildable")

    f = BV.open_store(out)
    done = set(f["name"].asstr()[:]) if f["global"].shape[0] else set()
    if done:
        log(f"[plan] resuming: {len(done)} views already stored")
    koi = koi[~koi.kepoi_name.isin(done)]

    groups = list(koi.groupby("kepid"))
    log(f"[plan] {len(groups)} stars to process, {workers} workers\n")
    if not groups:
        log("[done] nothing to do")
        f.close()
        return

    t0 = time.time()
    n = ok = fail = 0
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker) as ex:
            futs = {ex.submit(process_star, k, g, window): k
                    for k, g in groups}
            for fut in as_completed(futs):
                try:
                    recs, err = fut.result()
                except Exception as e:
                    recs, err = [], f"KIC {futs[fut]}: worker crashed: {e}"
                n += 1
                if err:
                    fail += 1
                    if fail <= 30:
                        log(f"  [fail] {err}")
                else:
                    ok += len(recs)
                    BV.append(f, recs)
                if n % 250 == 0:
                    rate = (time.time() - t0) / n
                    log(f"[{n}/{len(groups)}] {ok} views, {fail} failed | "
                        f"{rate:.3f}s/star | ETA {(len(groups)-n)*rate/60:.1f} min")
    finally:
        nt = f["global"].shape[0]
        lab = f["label"][:]
        log(f"\n[done] {nt} views in {out} "
            f"({int(lab.sum())} planet / {int((lab==0).sum())} FP)")
        log(f"[done] {fail} stars failed, {time.time()-t0:.1f}s elapsed")
        log(f"[done] file size: {os.path.getsize(out)/1e6:.1f} MB")
        f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--window", type=int, default=101)
    ap.add_argument("--include-candidates", action="store_true")
    ap.add_argument("--out", default="views.h5")
    a = ap.parse_args()
    main(a.workers, a.window, a.include_candidates, a.out)
