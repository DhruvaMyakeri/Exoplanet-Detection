"""
Phase 1 of 2: bulk-download every Kepler light curve to a local cache.

WHY THIS EXISTS
---------------
build_views.py interleaved download and compute: fetch a star, fold it,
delete the FITS, fetch the next. Measured on this machine, that split is

    download  47.17 s/star      (99%)
    compute    0.66 s/star      ( 1%)

so the expensive half was thrown away and the cheap half was kept. Any
later change to binning, window length, or the odd/even views of Stage 4
meant re-downloading the entire mission.

This script inverts that. Download once, keep (time, flux) forever,
recompute views in minutes.

WHY S3 AND NOT MAST
-------------------
Measured head to head, same machine, same moment:

    MAST via lightkurve, 12 workers    75.9-90.7 s/star
    S3 via boto3,         1 worker       8.0 s/star

MAST throttles concurrent connections; the public S3 mirror does not.
The layout is deterministic, so we skip search_lightcurve entirely
(that call alone cost 13.2 s/star):

    s3://stpubdata/kepler/public/lightcurves/<KIC[:4]>/<KIC>/

Anonymous unsigned access to the STScI public dataset. No credentials,
no egress cost - Kepler is in AWS's Registry of Open Data.
"""

import argparse
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = "cache"
KOI_CSV = "koi_ephem.csv"
LOGFILE = "fetch_all.log"
BUCKET = "stpubdata"


def log(msg):
    line = str(msg)
    print(line, file=sys.__stdout__, flush=True)
    with open(LOGFILE, "a") as fh:
        fh.write(line + "\n")


def _client():
    """Unsigned S3 client. Built per process - boto3 clients are not
    safe to share across a fork."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(
        signature_version=UNSIGNED,
        max_pool_connections=32,
        retries={"max_attempts": 5, "mode": "adaptive"},
    ))


def _init_worker():
    warnings.filterwarnings("ignore")


def fetch_star(kepid):
    """
    Download one star's long-cadence quarters, stitch, cache (time, flux).

    Returns (kepid, n_cadences, None) or (kepid, 0, error_string).
    """
    out = os.path.join(CACHE, f"{kepid}.npz")
    if os.path.exists(out):
        return kepid, -1, None          # -1 = already cached

    try:
        import lightkurve as lk
        s3 = _client()
        pad = str(kepid).zfill(9)
        prefix = f"kepler/public/lightcurves/{pad[:4]}/{pad}/"

        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        keys = [o["Key"] for o in resp.get("Contents", [])
                if o["Key"].endswith("_llc.fits")]
        if not keys:
            return kepid, 0, "no long-cadence data on S3"

        tmp = tempfile.mkdtemp()
        try:
            # The 15 quarters are independent objects, so fetch them
            # concurrently. This is pure network wait - threads are the
            # right tool and the GIL is irrelevant.
            def grab(key):
                dest = os.path.join(tmp, os.path.basename(key))
                s3.download_file(BUCKET, key, dest)
                return dest

            with ThreadPoolExecutor(max_workers=8) as tp:
                paths = list(tp.map(grab, keys))

            lcs = []
            for p in paths:
                try:
                    lcs.append(lk.read(p))
                except Exception:
                    pass                # one bad quarter must not sink the star
            if not lcs:
                return kepid, 0, "all quarters failed to parse"

            # Chronological order matters: flatten() runs a rolling window
            # over the array, so out-of-order quarters corrupt the trend.
            lcs.sort(key=lambda x: float(np.nanmin(x.time.value)))
            lc = lk.LightCurveCollection(lcs).stitch().remove_nans()

            t = np.asarray(lc.time.value, dtype=np.float64)
            f = np.asarray(lc.flux.value, dtype=np.float32)
            good = np.isfinite(t) & np.isfinite(f)
            t, f = t[good], f[good]
            if len(t) < 1000:
                return kepid, 0, f"only {len(t)} cadences"

            # float64 for time is NOT optional: BKJD runs to ~1590 d and
            # folding needs ~1e-5 d resolution. float32 has ~7 significant
            # digits and would quantise the epoch.
            tmpf = out + ".tmp"
            np.savez(tmpf, time=t, flux=f)
            os.replace(tmpf + ".npz", out)   # atomic - a killed job leaves
            return kepid, len(t), None       # no half-written cache entry
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    except Exception as e:
        return kepid, 0, f"{type(e).__name__}: {e}"


def main(workers, limit, include_candidates):
    os.makedirs(CACHE, exist_ok=True)

    koi = pd.read_csv(KOI_CSV)
    keep = ["CONFIRMED", "FALSE POSITIVE"]
    if include_candidates:
        keep.append("CANDIDATE")
    koi = koi[koi.koi_disposition.isin(keep)]
    koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])

    kepids = sorted(koi.kepid.unique().tolist())
    if limit:
        kepids = kepids[:limit]

    have = {int(x[:-4]) for x in os.listdir(CACHE) if x.endswith(".npz")}
    todo = [k for k in kepids if k not in have]
    log(f"[plan] {len(kepids)} stars needed, {len(have)} cached, "
        f"{len(todo)} to fetch, {workers} workers")
    if not todo:
        log("[done] cache already complete")
        return

    t0 = time.time()
    n = ok = fail = skip = 0
    bytes_est = 0
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(fetch_star, k): k for k in todo}
        for fut in as_completed(futs):
            try:
                kid, ncad, err = fut.result()
            except Exception as e:
                kid, ncad, err = futs[fut], 0, f"worker crashed: {e}"
            n += 1
            if err:
                fail += 1
                if fail <= 40:
                    log(f"  [fail] KIC {kid}: {err}")
            elif ncad == -1:
                skip += 1
            else:
                ok += 1
                bytes_est += ncad * 12
            if n % 100 == 0:
                el = time.time() - t0
                rate = el / n
                log(f"[{n}/{len(todo)}] ok={ok} fail={fail} | "
                    f"{rate:.2f}s/star | {bytes_est/1e9:.1f} GB | "
                    f"ETA {(len(todo)-n)*rate/60:.0f} min")

    el = time.time() - t0
    log(f"\n[done] fetched {ok}, failed {fail}, already-cached {skip}")
    log(f"[done] {el/60:.1f} min total, {el/max(n,1):.2f} s/star")
    log(f"[done] cache: {len(os.listdir(CACHE))} files, "
        f"{sum(os.path.getsize(os.path.join(CACHE,f)) for f in os.listdir(CACHE))/1e9:.1f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-candidates", action="store_true")
    a = ap.parse_args()
    main(a.workers, a.limit, a.include_candidates)
