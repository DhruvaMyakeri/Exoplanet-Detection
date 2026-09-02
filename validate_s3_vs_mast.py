import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, h5py
import build_views_local as BVL

f = h5py.File("views.h5", "r")
names = list(f["name"].asstr()[:])
kepids = f["kepid"][:]
G = f["global"][:]; L = f["local"][:]; SNR = f["snr"][:]
f.close()

koi = pd.read_csv("koi_ephem.csv")
koi = koi[koi.koi_disposition.isin(["CONFIRMED", "FALSE POSITIVE"])]
koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])

rows = []
for kid, grp in koi[koi.kepid.isin(set(int(k) for k in kepids))].groupby("kepid"):
    recs, err = BVL.process_star(int(kid), grp, 101)
    if err:
        print("  ERR", kid, err); continue
    for r in recs:
        rows.append(r)

new = {r["name"]: r for r in rows}
print(f"MAST views: {len(names)}   rebuilt from S3 cache: {len(new)}")

gd, ld, sd, matched = [], [], [], 0
for i, nm in enumerate(names):
    if nm not in new: continue
    matched += 1
    gd.append(np.nanmax(np.abs(G[i] - new[nm]["glob"])))
    ld.append(np.nanmax(np.abs(L[i] - new[nm]["loc"])))
    sd.append(abs(SNR[i] - new[nm]["snr"]))

gd, ld, sd = np.array(gd), np.array(ld), np.array(sd)
print(f"matched by name: {matched}")
print(f"global  max|diff|:  median {np.median(gd):.2e}   worst {gd.max():.2e}")
print(f"local   max|diff|:  median {np.median(ld):.2e}   worst {ld.max():.2e}")
print(f"snr        |diff|:  median {np.median(sd):.2e}   worst {sd.max():.2e}")
print(f"views identical to 1e-4: {int((gd<1e-4).sum())}/{matched} global, "
      f"{int((ld<1e-4).sum())}/{matched} local")
