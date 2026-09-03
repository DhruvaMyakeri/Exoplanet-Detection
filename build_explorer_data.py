"""
Pack the held-out test set into a single JSON payload for the explorer.

WHY THE TEST SET ONLY
---------------------
These 1,387 objects are the only ones whose scores are honest out-of-sample
numbers. Showing train or val predictions next to them would mix fitted and
held-out values in one table, which is exactly the confusion the split
exists to prevent.

SIZE
----
Artifacts cap at 16 MB and views.h5 is 61 MB, so the curves are quantised:
the global view is decimated 2001 -> 501 for display, and both views are
stored as int16 at 1/1000 resolution.

int8 was the first attempt and was WRONG. The views are not confined to
[-1.2, +0.3] as the depth normalisation suggests - the pipeline clips at
+/-5, and 5.6% of global cells and 8.9% of local cells exceed |1.27|. int8
at 1/100 silently flattened one cell in fifteen (max error 3.73). int16 at
1/1000 covers +/-32 exactly; the printed reconstruction error confirms it.

Scores are the Platt-calibrated ensemble from evaluate.py - the only
outputs on this page that can be read as probabilities.
"""

import base64
import json

import h5py
import numpy as np
import pandas as pd

DECIM = 4          # 2001 -> 501


def q16(a):
    """Quantise to int16 at 1/1000 resolution. Views span +/-5, so this is
    exact to the third decimal - far finer than a plotted line width."""
    return np.clip(np.round(a * 1000.0), -32767, 32767).astype(np.int16)


def main():
    cal = np.load("calibrated_test_preds.npz")
    c = np.load("cnn_test_preds.npz")
    r = np.load("rf_test_preds.npz")
    ti = cal["test_idx"]
    y = cal["y_true"].astype(int)
    ens = cal["ensemble_rank"]
    cnn = np.mean([c[f"seed{s}"] for s in [0, 1, 2]], axis=0)
    rf = np.mean([r[f"seed{s}"] for s in [0, 1, 2]], axis=0)

    with h5py.File("views.h5", "r") as f:
        G = f["global"][:][ti][:, ::DECIM]
        L = f["local"][:][ti]
        name = np.array(f["name"].asstr()[:])[ti]
        kepid = f["kepid"][:][ti]
        period = f["period"][:][ti]
        dur = f["duration"][:][ti]
        depth = f["depth"][:][ti]
        snr = f["snr"][:][ti]
        npts = f["npts"][:][ti]

    koi = (pd.read_csv("koi_cumulative.csv")
             .drop_duplicates("kepoi_name").set_index("kepoi_name"))
    sub = koi.reindex(name)
    prad = sub["koi_prad"].to_numpy(dtype=float)
    steff = sub["koi_steff"].to_numpy(dtype=float)
    srad = sub["koi_srad"].to_numpy(dtype=float)
    kmag = sub["koi_kepmag"].to_numpy(dtype=float)

    gq, lq = q16(G), q16(L)
    print(f"[quant] global max error {np.abs(gq/1000.0 - G).max():.4f}")
    print(f"[quant] local  max error {np.abs(lq/1000.0 - L).max():.4f}")

    rows = []
    for i in range(len(ti)):
        rows.append(dict(
            n=str(name[i]), k=int(kepid[i]), y=int(y[i]),
            e=round(float(ens[i]), 4), c=round(float(cnn[i]), 4),
            r=round(float(rf[i]), 4),
            p=round(float(period[i]), 4), d=round(float(dur[i]), 3),
            dp=None if not np.isfinite(depth[i]) or depth[i] < 0
               else round(float(depth[i]), 1),
            s=round(float(snr[i]), 2), np_=int(npts[i]),
            pr=None if not np.isfinite(prad[i]) else round(float(prad[i]), 2),
            te=None if not np.isfinite(steff[i]) else int(steff[i]),
            sr=None if not np.isfinite(srad[i]) else round(float(srad[i]), 3),
            km=None if not np.isfinite(kmag[i]) else round(float(kmag[i]), 2),
        ))

    payload = dict(
        meta=dict(n=len(rows), gbins=G.shape[1], lbins=L.shape[1],
                  decim=DECIM,
                  scale=1000,
                  note="Platt-calibrated ensemble on the held-out test set"),
        rows=rows,
        g=base64.b64encode(gq.tobytes()).decode(),
        l=base64.b64encode(lq.tobytes()).decode(),
    )
    js = json.dumps(payload, separators=(",", ":"))
    with open("explorer_data.json", "w") as fh:
        fh.write(js)
    print(f"[done] explorer_data.json  {len(js)/1e6:.2f} MB  ({len(rows)} objects)")


if __name__ == "__main__":
    main()
