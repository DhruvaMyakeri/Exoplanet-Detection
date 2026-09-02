# refetch_koi.py
import io, requests, pandas as pd

COLUMNS = [
    "kepid", "kepoi_name", "kepler_name", "koi_disposition",
    "koi_period", "koi_time0bk", "koi_duration", "koi_depth",
    "koi_prad", "koi_impact", "koi_model_snr", "koi_num_transits",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
q = "select " + ",".join(COLUMNS) + " from cumulative"
print("querying archive ...")
r = requests.get(url, params={"query": q, "format": "csv"}, timeout=180)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text))
df.to_csv("koi_ephem.csv", index=False)

print(f"{len(df)} rows -> koi_ephem.csv")
print("missing koi_time0bk:", df.koi_time0bk.isna().sum())
print(df[["kepoi_name", "koi_period", "koi_time0bk", "koi_duration",
          "koi_depth"]].head().to_string(index=False))