"""
126_peer_effects_similarity.py

Q1. How alike are neighbouring districts, and how fast does that fall with distance?

  (a) Crop-mix similarity of every pair of sample units within 600 km:
        sim = 1 - 0.5 * sum_g |s_ig - s_jg|  over the nine crop groups
      (1 = identical mix, 0 = no overlap), and the gap in effective number of crops
      |D1_i - D1_j|, binned by centroid distance (20 km bins), separately for pairs in
      the same state and pairs across a state border. Both windows (1998-2002, 2015-19).
      Andhra Pradesh-Telangana pairs are left out of both lines (a border only from 2014).
  (b) Moran's I (row-standardised queen adjacency, 999 permutations) of D1 and of the
      group shares: 2015-19 level and the 1998-2002 to 2015-19 change. Reported with all
      links, with within-state links only, with cross-border links only, and on values
      with state means removed (all links).

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/126_peer_effects_similarity.py
Outputs (outputs/revision_nov8/peer_effects/):
  q1_similarity_by_distance.csv, q1_moran.csv, q1_similarity_summary.csv, q1_log.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pe_common import OUT, GROUPS, load, moran  # noqa: E402

LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


units, uy, up, P = load()
P = P[P.new_border == 0].copy()
rng = np.random.default_rng(20261107)

# ------------------------------------------------------------------ (a) similarity by distance
BIN = 20
P["dbin"] = (P.dist_km // BIN).astype(int) * BIN + BIN / 2
rows = []
for lab in ["w9802", "w1519"]:
    for kind, sub in [("same_state", P[P.same_state == 1]), ("cross_border", P[P.same_state == 0])]:
        for adjk, s2 in [("all", sub), ("adjacent", sub[sub.adjacent == 1])]:
            g = s2.dropna(subset=[f"sim_{lab}"]).groupby("dbin")
            t = g.agg(n_pairs=(f"sim_{lab}", "size"), sim_mean=(f"sim_{lab}", "mean"),
                      sim_p25=(f"sim_{lab}", lambda x: x.quantile(.25)),
                      sim_p75=(f"sim_{lab}", lambda x: x.quantile(.75)),
                      dD1_mean=(f"dD1_{lab}", "mean"), dD1_median=(f"dD1_{lab}", "median")).reset_index()
            t["window"], t["pair_type"], t["pairs"] = lab, kind, adjk
            rows.append(t)
sd = pd.concat(rows, ignore_index=True)
sd = sd[sd.n_pairs >= 10]
sd.to_csv(os.path.join(OUT, "q1_similarity_by_distance.csv"), index=False)

# summary rows for the text
summ = []
for lab in ["w9802", "w1519"]:
    for nm, m in [("adjacent, same state", (P.adjacent == 1) & (P.same_state == 1)),
                  ("adjacent, across a border", (P.adjacent == 1) & (P.same_state == 0)),
                  ("not adjacent, 60-100 km, same state", (P.adjacent == 0) & P.dist_km.between(60, 100) & (P.same_state == 1)),
                  ("not adjacent, 60-100 km, across a border", (P.adjacent == 0) & P.dist_km.between(60, 100) & (P.same_state == 0)),
                  ("200-300 km, same state", P.dist_km.between(200, 300) & (P.same_state == 1)),
                  ("200-300 km, across a border", P.dist_km.between(200, 300) & (P.same_state == 0)),
                  ("500-600 km, any", P.dist_km.between(500, 600))]:
        s = P.loc[m, [f"sim_{lab}", f"dD1_{lab}", "dist_km"]].dropna()
        summ.append({"window": lab, "pairs": nm, "n_pairs": len(s), "mean_km": s.dist_km.mean(),
                     "sim_mean": s[f"sim_{lab}"].mean(), "sim_median": s[f"sim_{lab}"].median(),
                     "dD1_mean": s[f"dD1_{lab}"].mean()})
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(OUT, "q1_similarity_summary.csv"), index=False)
w("Similarity of crop mix (1 - half L1 over nine groups) and |D1 gap|:")
w(summ.round(3).to_string(index=False))

# distance at which same-state similarity falls to the cross-border adjacent level
for lab in ["w1519"]:
    xb = summ[(summ.window == lab) & (summ.pairs == "adjacent, across a border")].sim_mean.iloc[0]
    c = sd[(sd.window == lab) & (sd.pair_type == "same_state") & (sd.pairs == "all")].sort_values("dbin")
    hit = c[c.sim_mean <= xb]
    w(f"{lab}: same-state pairs first fall to the cross-border-adjacent similarity ({xb:.3f}) at "
      f"{hit.dbin.iloc[0] if len(hit) else 'n/a'} km")

# ------------------------------------------------------------------ (b) Moran's I
import pandas as _pd  # noqa: E402
ed = _pd.read_csv(os.path.join(OUT, "..", "..", "crop_groups", "crop_group_adjacency_edges.csv"))
w2 = up[up.window == "w1519"].set_index("unit_key")
w1 = up[up.window == "w9802"].set_index("unit_key")
st = units.set_index("unit_key").st
VARS = ["D1"] + [f"s_{g}" for g in GROUPS if g != "other"]
mrows = []
for meas in ["level_2015_19", "change_9802_to_1519"]:
    for v in VARS:
        if meas == "level_2015_19":
            x = w2[v].dropna()
        else:
            x = (w2[v] - w1[v]).dropna()
        x = x[x.index.isin(units[units.in_606 == 1].unit_key)]
        x = x[x.index.isin(st.index)]
        keys = list(x.index)
        ix = {k: i for i, k in enumerate(keys)}
        Wb = np.zeros((len(keys), len(keys)))
        for a, b in zip(ed.unit_i, ed.unit_j):
            if a in ix and b in ix:
                Wb[ix[a], ix[b]] = 1
        sarr = st.reindex(keys).values
        same = sarr[:, None] == sarr[None, :]
        # states are present-day: Andhra Pradesh-Telangana links count as cross-border here
        xv = x.values.astype(float)
        xdm = xv - pd.Series(xv).groupby(sarr).transform("mean").values
        for linkset, Wl, vals in [("all links", Wb, xv),
                                  ("within-state links", Wb * same, xv),
                                  ("cross-border links", Wb * (~same), xv),
                                  ("all links, state means removed", Wb, xdm)]:
            r = moran(vals, Wl, nperm=999, rng=rng)
            mrows.append({"measure": meas, "var": v, "links": linkset, **r})
mo = pd.DataFrame(mrows)
mo.to_csv(os.path.join(OUT, "q1_moran.csv"), index=False)
w("")
w("Moran's I:")
w(mo.pivot_table(index=["measure", "var"], columns="links", values="I").round(3).to_string())
w(mo.pivot_table(index=["measure", "var"], columns="links", values="n").to_string())

with open(os.path.join(OUT, "q1_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
