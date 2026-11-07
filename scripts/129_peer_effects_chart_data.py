"""
129_peer_effects_chart_data.py

Chart-ready files for the peer-effects page, from the outputs of scripts 125-128.
Nothing is estimated here except Benjamini-Hochberg q-values over the 14 main
neighbour-minus-ring tests of script 128.

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/129_peer_effects_chart_data.py
Outputs (outputs/revision_nov8/peer_effects/chart_data/):
  v1_similarity_vs_distance.csv / .json     curve, same state vs across a border, two windows
  v2_border_dumbbells.csv / .json           side means and gaps, highlighted borders, both windows
  v3_border_gap_over_time.csv               yearly similarity by pair type; yearly signed gaps on chosen borders
  v4_spread_maps.json                       unit event year + yearly grower status for the map episodes
  v4_polygon_to_unit.csv                    geojson feature (n, s, k) -> unit_key, for joining the maps
  v5_neighbour_vs_placebo.csv / .json       hazard coefficients: neighbours, ring placebo, difference, q
  v6_neighbour_explorer.json                each district, its crop mix, each neighbour's similarity
  v7_moran.csv                              Moran's I by link set
  v8_wave.csv                               contiguity of new growers: observed vs random
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pe_common import OUT, ROOT, GROUPS, load  # noqa: E402

CD = os.path.join(OUT, "chart_data")
os.makedirs(CD, exist_ok=True)
units, uy, up, P = load()


def dump(obj, name):
    with open(os.path.join(CD, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False, default=lambda x: None)


def clean(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), nd) if isinstance(v, (float, np.floating)) else v


def recs(df, nd=4):
    return [{k: clean(v, nd) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, np.integer) else v)
             for k, v in r.items()} for r in df.to_dict("records")]


name_of = units.set_index("unit_key").district.str.title()
st_of = units.set_index("unit_key").st.str.title()

# v1 -------------------------------------------------------------------------------
sd = pd.read_csv(os.path.join(OUT, "q1_similarity_by_distance.csv"))
v1 = sd[(sd.pairs == "all") & (sd.n_pairs >= 30)][["window", "pair_type", "dbin", "n_pairs", "sim_mean",
                                                     "sim_p25", "sim_p75", "dD1_mean"]].copy()
v1["window"] = v1.window.map({"w9802": "1998-2002", "w1519": "2015-19"})
v1 = v1.rename(columns={"dbin": "km"})
for c in ["sim_mean", "sim_p25", "sim_p75"]:
    v1[c] = (v1[c] * 100).round(2)  # points, as in the other chart files
v1.to_csv(os.path.join(CD, "v1_similarity_vs_distance.csv"), index=False)
adj = pd.read_csv(os.path.join(OUT, "q1_similarity_summary.csv"))
dump({"note": "sim = 100 x (1 - half the L1 distance between nine-group shares of GCA), points; reference_points "
              "keep the 0-1 scale; pairs of sample districts "
              "(2011 units) with centroids within 600 km; Andhra Pradesh-Telangana pairs left out",
      "curve": recs(v1), "reference_points": recs(adj)}, "v1_similarity_vs_distance.json")

# v2 -------------------------------------------------------------------------------
bg = pd.read_csv(os.path.join(OUT, "q2_border_gaps.csv"))
v2 = bg[(bg.highlight == 1) & bg["var"].isin(["rice", "wheat", "pulses", "millets", "maize", "cotton", "D1"])
        & bg.window.isin(["w9802", "w1519"])].copy()
v2["window"] = v2.window.map({"w9802": "1998-2002", "w1519": "2015-19"})
v2 = v2[["border", "state_a", "state_b", "var", "window", "n_pairs", "n_units_a", "n_units_b", "mean_a", "mean_b",
         "gap_pairFE", "se_unit", "geo_predicted_gap", "gap_net_of_geography", "se_net", "few_pairs"]]
v2.to_csv(os.path.join(CD, "v2_border_dumbbells.csv"), index=False)
dump({"note": "border districts only (adjacent cross-border pairs). mean_a/mean_b: unit means of the share "
              "(points of GCA; D1 in effective crops) on each side; gap_pairFE: mean within-pair difference A minus "
              "B with SE dyadic-robust by 2011 unit; few_pairs = fewer than 8 pairs, SE unreliable",
      "rows": recs(v2, 3)}, "v2_border_dumbbells.json")

# v3 -------------------------------------------------------------------------------
pt = pd.read_csv(os.path.join(OUT, "q2_pair_type_year_means.csv"))
pt = pt[["ptype", "year", "n_pairs", "sim", "dD1", "g_rice", "g_wheat", "g_pulses", "g_millets"]].rename(
    columns={"ptype": "series"})
pt["kind"] = "adjacent pairs, mean"
ba = pd.read_csv(os.path.join(OUT, "q2_border_annual.csv"))
pick = [("Telangana | Andhra Pradesh", "rice"), ("Telangana | Andhra Pradesh", "cotton"),
        ("Telangana | Karnataka", "cotton"), ("Jharkhand | Odisha", "rice"), ("Chhattisgarh | Odisha", "rice"),
        ("Punjab | Haryana", "rice"), ("Haryana | Rajasthan", "wheat"), ("Madhya Pradesh | Uttar Pradesh", "wheat"),
        ("Madhya Pradesh | Uttar Pradesh", "pulses"), ("West Bengal | Bihar", "wheat")]
bb = pd.concat([ba[(ba.border == b) & (ba["var"] == v)] for b, v in pick])
bb = bb.rename(columns={"border": "series"})
bb["kind"] = "signed border gap, A minus B, points"
pd.concat([pt, bb], ignore_index=True).to_csv(os.path.join(CD, "v3_border_gap_over_time.csv"), index=False)

# v4 -------------------------------------------------------------------------------
ey = pd.read_csv(os.path.join(OUT, "q3_entry_years.csv"))
us = pd.read_csv(os.path.join(OUT, "q3_unit_year_status.csv.gz"))
MAPS = [("chickpea", "surge"), ("soybean", "entry"), ("cotton", "surge"), ("maize", "entry"), ("rabi_maize", "entry")]
v4 = {"note": "one record per sample 2011 unit. event_year: first year of a sustained entry (share crosses 5 points "
              "of GCA, 2 for rabi maize) or surge (share 5 points above own 1998-2000 level); 'established' = "
              "already above the level in 1998-2000. share_by_year in points of GCA, null = unreliable year.",
      "years": list(range(1998, 2020)), "episodes": {}}
for e, m in MAPS:
    d = ey[(ey.episode == e) & (ey["mode"] == m)].set_index("unit_key")
    s = us[(us.episode == e) & (us["mode"] == m)].pivot(index="unit_key", columns="year", values="share")
    out = {}
    for u, r in d.iterrows():
        sh = s.loc[u] if u in s.index else None
        out[u] = {"name": name_of.get(u), "state": st_of.get(u), "status": r.status_1998_2000,
                  "event_year": None if not np.isfinite(r.event_year) else int(r.event_year),
                  "base": clean(r.baseline_share * 100, 2),
                  "share_by_year": None if sh is None else [clean(x * 100, 2) for x in sh.values]}
    v4["episodes"][f"{e}_{m}"] = {"threshold_pts": float(d.threshold.iloc[0] * 100), "units": out}
dump(v4, "v4_spread_maps.json")
pm = pd.read_csv(os.path.join(ROOT, "outputs", "crop_groups", "crop_group_polygon_match.csv"))
pm[["n", "s", "k", "unit_key"]].to_csv(os.path.join(CD, "v4_polygon_to_unit.csv"), index=False)

# v5 -------------------------------------------------------------------------------
H = pd.read_csv(os.path.join(OUT, "q3_hazard.csv"))
h = H[(H.tag == "main") & (H.cluster == "unit")]
rows = []
for (e, m), g in h.groupby(["episode", "mode"]):
    r = {"episode": e, "mode": m}
    for spec, var, lab in [("nbr + ring", "nbr", "nbr"), ("nbr + ring", "ring", "ring"),
                           ("nbr + ring", "nbr minus ring", "diff"), ("nbr split by border", "nbr_same", "nbr_same_state"),
                           ("nbr split by border", "nbr_other", "nbr_other_state"),
                           ("nbr + ring + same-state", "samestate", "same_state_non_nbr")]:
        q = g[(g.spec == spec) & (g["var"] == var)]
        if len(q):
            r[f"{lab}_b"], r[f"{lab}_se"], r[f"{lab}_p"] = q.b.iloc[0], q.se.iloc[0], q.p.iloc[0]
    q = g[(g.spec == "nbr + ring") & (g["var"] == "nbr")]
    r["n_unit_years"], r["n_events"], r["base_rate"] = q.n.iloc[0], q.n_entries.iloc[0], q.mean_entry.iloc[0]
    hs = H[(H.tag == "main") & (H.cluster == "state") & (H.episode == e) & (H["mode"] == m)
           & (H.spec == "nbr + ring") & (H["var"] == "nbr")]
    r["nbr_se_state"] = hs.se.iloc[0] if len(hs) else np.nan
    rows.append(r)
v5 = pd.DataFrame(rows)
# BH over the 14 neighbour-minus-ring tests
p = v5.diff_p.values
o = np.argsort(p)
q = np.empty_like(p)
q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
v5["diff_q_bh14"] = np.minimum(q, 1)
v5.to_csv(os.path.join(CD, "v5_neighbour_vs_placebo.csv"), index=False)
dump({"note": "linear-probability hazard, state-by-year FE, SEs clustered on 2011 unit. Coefficients: change in the "
              "yearly probability of the event when the share of growers among the set goes from 0 to 1. ring = "
              "non-adjacent districts at 0.75-1.5x the mean neighbour distance (about 104 km). q = BH over 14 tests.",
      "rows": recs(v5, 4)}, "v5_neighbour_vs_placebo.json")

# v6 -------------------------------------------------------------------------------
ed = pd.read_csv(os.path.join(ROOT, "outputs", "crop_groups", "crop_group_adjacency_edges.csv"))
A = P[P.adjacent == 1]
exp = {}
for lab, wl in [("w1519", "2015-19"), ("w9802", "1998-2002")]:
    t = up[up.window == lab].set_index("unit_key")
    for u, r in t.iterrows():
        if u not in st_of.index or units.set_index("unit_key").in_606.get(u, 0) != 1:
            continue
        exp.setdefault(u, {"name": name_of.get(u), "state": st_of.get(u),
                           "lon": clean(units.set_index("unit_key").lon.get(u)),
                           "lat": clean(units.set_index("unit_key").lat.get(u)), "mix": {}, "neighbours": []})
        exp[u]["mix"][wl] = {"D1": clean(r.D1, 2), **{g: clean(r[f"s_{g}"] * 100, 1) for g in GROUPS}}
for _, r in A.iterrows():
    for a, b in [(r.unit_i, r.unit_j), (r.unit_j, r.unit_i)]:
        if a in exp:
            exp[a]["neighbours"].append({"unit_key": b, "name": name_of.get(b), "state": st_of.get(b),
                                         "same_state": int(r.same_state), "km": clean(r.dist_km, 0),
                                         "sim_2015_19": clean(r.sim_w1519 * 100, 1),
                                         "sim_1998_2002": clean(r.sim_w9802 * 100, 1),
                                         "D1_gap_2015_19": clean(r.dD1_w1519, 2)})
for u, v in exp.items():
    s = [n["sim_2015_19"] for n in v["neighbours"] if n["sim_2015_19"] is not None]
    v["mean_nbr_sim_2015_19"] = clean(np.mean(s), 1) if s else None
allsim = [v["mean_nbr_sim_2015_19"] for v in exp.values() if v["mean_nbr_sim_2015_19"] is not None]
for v in exp.values():
    if v["mean_nbr_sim_2015_19"] is not None:
        v["pct_rank_mean_nbr_sim"] = clean(np.mean(np.array(allsim) <= v["mean_nbr_sim_2015_19"]) * 100, 0)
dump({"note": "similarity in points (100 = identical nine-group mix); shares in points of GCA; D1 = effective number "
              "of crops (exp Shannon over all APY crops), mean of annual values over reliable years",
      "groups": GROUPS, "districts": exp}, "v6_neighbour_explorer.json")

# v7, v8 ---------------------------------------------------------------------------
mo = pd.read_csv(os.path.join(OUT, "q1_moran.csv"))
mo[["measure", "var", "links", "I", "EI", "sd_perm", "z", "p_perm", "n"]].to_csv(os.path.join(CD, "v7_moran.csv"), index=False)
wv = pd.read_csv(os.path.join(OUT, "q3_wave.csv"))
wv.to_csv(os.path.join(CD, "v8_wave.csv"), index=False)

print(v5[["episode", "mode", "n_events", "nbr_b", "nbr_se", "ring_b", "ring_se", "diff_b", "diff_se", "diff_p",
          "diff_q_bh14", "nbr_same_state_b", "nbr_same_state_se", "nbr_other_state_b", "nbr_other_state_se"]]
      .round(3).to_string(index=False))
print("explorer districts:", len(exp))
for f in sorted(os.listdir(CD)):
    print(f, os.path.getsize(os.path.join(CD, f)))
