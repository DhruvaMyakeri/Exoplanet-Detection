"""
Stage 2 - Build the training set.

Turns labelled KOIs into an HDF5 file of fixed-size views:

    global  (N, 2001)   whole folded orbit    -> context, secondary eclipse
    local   (N,  201)   +/-2.5 durations      -> transit shape, U vs V

Both normalised to median 0, minimum -1. Depth stored separately.

Requires koi_ephem.csv (from refetch_koi.py) - the Stage 0 cache does not
contain koi_time0bk, and without the epoch there is no phase zero.

Usage:
    python build_views.py --verify           # mask-convention check
    python build_views.py --limit 50         # trial run
    python build_views.py                    # full run (hours)
    python build_views.py --workers 1        # if threading misbehaves

Resumable: rerun after a crash, it skips what is already in views.h5.
Progress also written to build_views.log.
"""

import argparse
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import lightkurve as lk

GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_HALFWIDTH = 2.5          # in transit durations
OUT = "views.h5"
KOI_CSV = "koi_ephem.csv"
LOGFILE = "build_views.log"

_write_lock = threading.Lock()


# ==========================================================================
# Logging that survives the download layer
# ==========================================================================

def log(msg):
    """
    Write to the ORIGINAL stdout handle plus a file.

    Something in the astroquery/lightkurve download path closes
    sys.stdout when several threads finish at once, after which every
    print() raises ValueError. sys.__stdout__ is the real handle and is
    never touched, so logging through it keeps working.
    """
    try:
        sys.__stdout__.write(str(msg) + "\n")
        sys.__stdout__.flush()
    except Exception:
        pass
    try:
        with open(LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(str(msg) + "\n")
    except Exception:
        pass


class NullSink(io.TextIOBase):
    """
    A stdout replacement that swallows writes and refuses to close.

    We cannot patch the library's own print/tqdm calls, so instead we
    hand them a sink that can never enter the closed state. This is what
    stops one thread's teardown from breaking every other thread.
    """
    def write(self, s):
        return len(s)

    def flush(self):
        pass

    def close(self):
        pass                       # deliberately a no-op

    @property
    def closed(self):
        return False

    def isatty(self):
        return False

def _init_worker():
    """
    Runs once per worker PROCESS.

    Threads share one sys.stdout, and something in astroquery's download
    path closes it - which breaks every other thread. Processes each get
    their own, so the failure cannot propagate. This also silences the
    per-worker download chatter.
    """
    import warnings
    warnings.filterwarnings("ignore")
    sys.stdout = NullSink()
# ==========================================================================
# Binning and normalisation
# ==========================================================================

def bin_mean(x, y, lo, hi, nbins):
    """
    Average y into nbins uniform bins over [lo, hi]. Empty bins -> NaN.

    Mean rather than median: bincount is vectorised and this runs ~7,500
    times. Robustness comes from clipping beforehand instead.
    """
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.digitize(x, edges) - 1
    keep = (idx >= 0) & (idx < nbins) & np.isfinite(y)
    idx, y = idx[keep], y[keep]
    if len(idx) == 0:
        return np.full(nbins, np.nan), np.zeros(nbins, dtype=int)

    counts = np.bincount(idx, minlength=nbins)
    sums = np.bincount(idx, weights=y, minlength=nbins)
    out = np.full(nbins, np.nan)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out, counts


def fill_gaps(v):
    """
    Interpolate empty bins. Long-period planets have few transits, so the
    local view can legitimately have holes. Over 25% empty means the
    object is too sparse to trust - reject it.
    """
    nan = ~np.isfinite(v)
    if nan.mean() > 0.25:
        return None
    if nan.any():
        v = v.copy()
        v[nan] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), v[~nan])
    return v


def robust_scale(l_raw):
    """
    Estimate baseline and depth from the local view's KNOWN geometry
    instead of from its minimum.

    The local view spans +/-2.5 durations across 201 bins, so one
    duration is 201/5 = 40.2 bins. The transit core is the central
    +/-0.5 duration (bins ~80-120); anything beyond +/-1.5 durations
    is safely out of transit.

    Using the minimum instead makes low-SNR objects self-destruct: the
    deepest bin is a noise spike, the view gets divided by noise, and
    the real transit vanishes.
    """
    n = len(l_raw)
    c = n // 2
    per_dur = n / (2.0 * LOCAL_HALFWIDTH)         # bins per duration

    core = l_raw[int(c - 0.5 * per_dur): int(c + 0.5 * per_dur) + 1]
    wings = np.r_[l_raw[:int(c - 1.5 * per_dur)],
                  l_raw[int(c + 1.5 * per_dur):]]
    if len(core) < 5 or len(wings) < 20:
        return None

    base = np.median(wings)
    depth = base - np.median(core)
    scatter = np.std(wings)
    if not np.isfinite(depth) or depth <= 0 or scatter <= 0:
        return None

    # Per-bin significance of the dip. A real transit is many sigma;
    # noise masquerading as a transit is around 1.
    snr = depth / scatter
    return base, depth, snr


def apply_scale(v, base, depth):
    """Baseline -> 0, transit depth -> -1, using a SHARED scale."""
    out = (v - base) / depth
    return np.clip(out, -5.0, 5.0).astype(np.float32)

# ==========================================================================
# Per-object processing
# ==========================================================================

def detrend_masked(lc, period, epoch, duration_days, window=101):
    """
    Detrend with the transits excluded from the trend fit.

    Verified empirically: in lightkurve 2.6.0, mask=True means EXCLUDE
    from the fit. With sigma clipping disabled, masking recovers +646 ppm
    on a 6675 ppm transit - i.e. an unmasked filter eats ~13% of it.

    We mask 2x the catalogue duration. The catalogue value is T_14 (first
    to last contact), but the 29.4-minute integration smears ingress and
    egress wider, and the epoch can drift slightly.
    """
    in_transit = lc.create_transit_mask(period=period,
                                        transit_time=epoch,
                                        duration=duration_days * 2)
    flat = lc.flatten(window_length=window, mask=in_transit,
                      break_tolerance=5)
    return flat, in_transit


def make_views(flat, period, epoch, duration_days):
    """Fold once, crop twice."""
    t = flat.time.value
    f = flat.flux.value
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(t) < 500:
        return None

    phase = ((t - epoch + 0.5 * period) % period) / period - 0.5

    # Asymmetric clip. normalise() divides by the minimum, so one spurious
    # deep point would rescale the whole view; a high spike is harmless.
    med, sd = np.median(f), np.std(f)
    keep = (f < med + 5 * sd) & (f > med - 12 * sd)
    phase, f = phase[keep], f[keep]

    # --- global view: whole orbit, fixed PHASE resolution ---
    # Build both views RAW first - we need the local view to set the scale.
    g, gc = bin_mean(phase, f, -0.5, 0.5, GLOBAL_BINS)
    g = fill_gaps(g)
    if g is None:
        return None

    half = LOCAL_HALFWIDTH * duration_days / period
    if half >= 0.5:                    # transit longer than the orbit
        return None
    l, lcount = bin_mean(phase, f, -half, half, LOCAL_BINS)
    l = fill_gaps(l)
    if l is None:
        return None

    sc = robust_scale(l)
    if sc is None:
        return None
    base, depth, snr = sc

    # Same scale for both views, so they stay comparable to each other.
    g = apply_scale(g, np.median(g), depth)
    l = apply_scale(l, base, depth)
    return g, l, int(gc.sum()), float(snr)


def process_star(kepid, rows, tmproot, window):
    """
    Download one star ONCE, build views for every KOI on it.

    Multi-planet systems share a light curve; per-KOI downloading would
    refetch the same 15 quarters for each planet.
    """
    workdir = tempfile.mkdtemp(dir=tmproot)
    try:
        sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler",
                                  author="Kepler", cadence="long")
        if len(sr) == 0:
            return [], f"KIC {kepid}: no data"

        lc = sr.download_all(download_dir=workdir).stitch()
        lc = lc.remove_nans().remove_outliers(sigma_upper=4, sigma_lower=20)
        if len(lc) < 1000:
            return [], f"KIC {kepid}: only {len(lc)} cadences"

        results = []
        for _, r in rows.iterrows():
            try:
                dur_days = r.koi_duration / 24.0
                flat, _ = detrend_masked(lc, r.koi_period, r.koi_time0bk,
                                         dur_days, window)
                v = make_views(flat, r.koi_period, r.koi_time0bk, dur_days)
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
                log(f"  [skip] {r.kepoi_name}: {type(e).__name__}: {e}")
        return results, None

    except Exception as e:
        return [], f"KIC {kepid}: {type(e).__name__}: {e}"
    finally:
        # THE IMPORTANT LINE. ~90 GB of FITS vs ~67 MB of output.
        shutil.rmtree(workdir, ignore_errors=True)


# ==========================================================================
# HDF5 storage
# ==========================================================================

FIELDS = ["global", "local", "label", "kepid", "period", "duration",
          "depth", "npts", "snr", "name", "disp"]

def open_store(path):
    f = h5py.File(path, "a")
    if "global" not in f:
        vs = h5py.string_dtype()
        f.create_dataset("global", (0, GLOBAL_BINS),
                         maxshape=(None, GLOBAL_BINS), dtype="f4",
                         chunks=(64, GLOBAL_BINS))
        f.create_dataset("local", (0, LOCAL_BINS),
                         maxshape=(None, LOCAL_BINS), dtype="f4",
                         chunks=(64, LOCAL_BINS))
        for k, dt in [("label", "i1"), ("kepid", "i8"), ("period", "f4"),
                      ("duration", "f4"), ("depth", "f4"), ("npts", "i4"),
                      ("snr", "f4")]:
            f.create_dataset(k, (0,), maxshape=(None,), dtype=dt)
        for k in ["name", "disp"]:
            f.create_dataset(k, (0,), maxshape=(None,), dtype=vs)
    return f


def append(f, recs):
    if not recs:
        return
    n0 = f["global"].shape[0]
    n1 = n0 + len(recs)
    for k in FIELDS:
        f[k].resize(n1, axis=0)
    f["global"][n0:n1] = np.stack([r["glob"] for r in recs])
    f["local"][n0:n1] = np.stack([r["loc"] for r in recs])
    for k in ["label", "kepid", "period", "duration", "depth", "npts",
              "snr", "name", "disp"]:
        f[k][n0:n1] = [r[k] for r in recs]
    f.flush()


# ==========================================================================
# Verification
# ==========================================================================

def verify(window):
    """
    Establish the lightkurve mask convention empirically.

    The naive test (mask vs no mask) does NOT discriminate, because
    flatten() runs iterative sigma clipping by default (niters=3,
    sigma=3). On a deep transit every in-transit point is tens of sigma
    below the trend, so the clipper removes them on its own and the mask
    has nothing left to do. Only with clipping OFF must the mask work
    alone.
    """
    koi = pd.read_csv(KOI_CSV)
    cand = koi[(koi.koi_disposition == "CONFIRMED") &
               (koi.koi_depth > 5000) &
               (koi.koi_period < 10)]
    if len(cand) == 0:
        raise SystemExit("no suitable deep confirmed planet in the catalogue")
    r = cand.nlargest(1, "koi_model_snr").iloc[0]

    log(f"[verify] target {r.kepoi_name}  P={r.koi_period:.4f} d  "
        f"catalogue depth={r.koi_depth:.0f} ppm")

    tmp = tempfile.mkdtemp()
    try:
        sr = lk.search_lightcurve(f"KIC {int(r.kepid)}", mission="Kepler",
                                  author="Kepler", cadence="long")
        lc = sr.download_all(download_dir=tmp).stitch()
        lc = lc.remove_nans().remove_outliers(sigma_upper=4, sigma_lower=20)
        dur = r.koi_duration / 24.0

        def measure(flat):
            t, f_ = flat.time.value, flat.flux.value
            ph = ((t - r.koi_time0bk + 0.5 * r.koi_period) % r.koi_period) \
                / r.koi_period - 0.5
            hw = 0.4 * dur / r.koi_period
            din = np.nanmedian(f_[np.abs(ph) < hw])
            dout = np.nanmedian(f_[np.abs(ph) > 3 * hw])
            return (dout - din) * 1e6

        # niters=1, sigma=1e10 -> one pass, nothing ever clipped
        configs = [
            ("no mask, clip ON ", False, 3, 3.0),
            ("masked,  clip ON ", True, 3, 3.0),
            ("no mask, clip OFF", False, 1, 1e10),
            ("masked,  clip OFF", True, 1, 1e10),
        ]
        res = {}
        for label, use_mask, niters, sig in configs:
            kw = dict(window_length=window, niters=niters, sigma=sig)
            if use_mask:
                kw["mask"] = lc.create_transit_mask(
                    period=r.koi_period, transit_time=r.koi_time0bk,
                    duration=dur * 2)
            d = measure(lc.flatten(**kw))
            res[label.strip()] = d
            log(f"[verify] {label}  depth = {d:8.0f} ppm")

        gain = res["masked,  clip OFF"] - res["no mask, clip OFF"]
        log(f"\n[verify] clipping OFF: masking recovers {gain:+.0f} ppm")
        if gain > 100:
            log("[verify] PASS - mask=True means EXCLUDE from the trend fit.")
        elif gain < -100:
            log("[verify] INVERTED - negate the mask in detrend_masked().")
        else:
            log("[verify] INCONCLUSIVE - check the mask flagged anything.")

        n = lc.create_transit_mask(period=r.koi_period,
                                   transit_time=r.koi_time0bk,
                                   duration=dur * 2).sum()
        log(f"[verify] mask flagged {n} / {len(lc)} cadences "
            f"({n / len(lc) * 100:.1f}%)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Driver
# ==========================================================================

def main(limit, workers, window, include_candidates, tmpdir):
    # Library chatter goes to a sink that cannot be closed. Our own log()
    # writes to sys.__stdout__, which nothing touches.
    sys.stdout = NullSink()

    koi = pd.read_csv(KOI_CSV)
    keep = ["CONFIRMED", "FALSE POSITIVE"]
    if include_candidates:
        keep.append("CANDIDATE")
    koi = koi[koi.koi_disposition.isin(keep)]
    koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
    log(f"[plan] {len(koi)} KOIs on {koi.kepid.nunique()} stars")

    f = open_store(OUT)
    done = set(f["name"].asstr()[:]) if f["global"].shape[0] else set()
    if done:
        log(f"[plan] resuming: {len(done)} views already stored")
    koi = koi[~koi.kepoi_name.isin(done)]

    groups = list(koi.groupby("kepid"))
    if limit:
        groups = groups[:limit]
    log(f"[plan] {len(groups)} stars to process, {workers} workers\n")

    t_start = time.time()
    n_done = n_ok = n_fail = 0
    tmproot = tempfile.mkdtemp(dir=tmpdir) if tmpdir else tempfile.mkdtemp()
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker) as ex:
            futs = {ex.submit(process_star, k, g, tmproot, window): k
                    for k, g in groups}
            for fut in as_completed(futs):
                try:
                    recs, err = fut.result()
                except Exception as e:
                    recs, err = [], f"KIC {futs[fut]}: worker crashed: {e}"
                n_done += 1
                if err:
                    n_fail += 1
                    log(f"  [fail] {err}")
                else:
                    n_ok += len(recs)
                    with _write_lock:
                        append(f, recs)
                if n_done % 10 == 0:
                    rate = (time.time() - t_start) / n_done
                    log(f"[{n_done}/{len(groups)}] {n_ok} views, "
                        f"{n_fail} failed  |  {rate:.1f}s/star  "
                        f"ETA {(len(groups) - n_done) * rate / 60:.0f} min")
    finally:
        shutil.rmtree(tmproot, ignore_errors=True)
        try:
            n = f["global"].shape[0]
            lab = f["label"][:]
            log(f"\n[done] {n} views in {OUT} "
                f"({int(lab.sum())} planet / {int((lab == 0).sum())} FP)")
            log(f"[done] {n_fail} stars failed this run")
            log(f"[done] file size: {os.path.getsize(OUT) / 1e6:.1f} MB")
        finally:
            f.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max stars")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--window", type=int, default=101)
    ap.add_argument("--include-candidates", action="store_true")
    ap.add_argument("--tmpdir", default=None,
                    help="where to stage FITS, e.g. D:/PROJECTS/exoplanet/tmp")
    a = ap.parse_args()
    if a.verify:
        verify(a.window)
    else:
        main(a.limit, a.workers, a.window, a.include_candidates, a.tmpdir)