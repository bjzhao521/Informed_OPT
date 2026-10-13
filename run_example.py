"""Run iOPT on one county and write thresholds and event days.

    python run_example.py

Input  example_county.csv   date (YYYYMMDD), tmax_C, rh_mean (%), wind_ms
Output output/thresholds.csv, output/events.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from iopt import iopt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="example_county.csv")
ap.add_argument("--out", default="output")
ap.add_argument("--anchor", type=float, default=34.8,
                help="physical Tmax threshold in deg C (34.8 maize, 33.7 soybean; Zhao et al. 2026)")
ap.add_argument("--season", default="5-9", help="growing-season months")
ap.add_argument("--rarity", nargs="+", type=float, default=[0.01, 0.005, 0.0025, 0.001],
                help="target probabilities p_t, relative to all days")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(a.data, float_precision="round_trip")
df = df.dropna(subset=["tmax_C", "rh_mean", "wind_ms"])
n_all = len(df)                                       # p_t refers to all days
lo, hi = (int(x) for x in a.season.split("-"))
gs = df[((df.date // 100) % 100).between(lo, hi)].reset_index(drop=True)

T = gs.tmax_C.to_numpy(float)
D = 100.0 - gs.rh_mean.to_numpy(float)                # RH deficit, so all tails are upper tails
W = gs.wind_ms.to_numpy(float)
inputs = {"H": [T], "CDH": [T, D], "CHDW": [T, D, W]}
levels = {"opt": 0.0, "half": 0.5, "anchored": 1.0}

thr_rows, ev_rows = [], []
for p_t in a.rarity:
    for name, V in inputs.items():
        for lev, f in levels.items():
            if name == "H" and f > 0:
                continue
            r = iopt(V, p_t, n_total=n_all, anchor=a.anchor, f=f)
            thr_rows.append(dict(
                definition=name, threshold=lev, p_t=p_t, n_days=r["N"],
                tmax_C=round(r["thr"][0], 2),
                rh_below_pct=round(100.0 - r["thr"][1], 2) if len(V) > 1 else np.nan,
                wind_ms=round(r["thr"][2], 2) if len(V) > 2 else np.nan,
                anchor_clipped=r["clip"]))
            sel = gs[r["mask"]]
            ev_rows.append(pd.DataFrame({
                "date": sel.date, "definition": name, "threshold": lev, "p_t": p_t,
                "tmax_C": sel.tmax_C, "rh_mean": sel.rh_mean, "wind_ms": sel.wind_ms}))

TH = pd.DataFrame(thr_rows)
EV = pd.concat(ev_rows, ignore_index=True).sort_values(["definition", "threshold", "p_t", "date"])
TH.to_csv(os.path.join(a.out, "thresholds.csv"), index=False)
EV.to_csv(os.path.join(a.out, "events.csv"), index=False)

pd.set_option("display.width", 200)
print(f"{n_all} days, {len(gs)} in months {lo}-{hi}, anchor {a.anchor} C")
print(TH[["definition", "threshold", "p_t", "n_days", "tmax_C", "rh_below_pct", "wind_ms",
          "anchor_clipped"]].to_string(index=False))
print(f"\nwrote {a.out}/thresholds.csv and {a.out}/events.csv ({len(EV)} rows)")
