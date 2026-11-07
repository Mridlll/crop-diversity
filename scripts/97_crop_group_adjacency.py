"""
97_crop_group_adjacency.py
(Three scripts share the number 97: 97_crop_group_adjacency.py runs inside run_all
 as step "97"; 97_imd_district_rainfall.py and 97_policy_nfsm_omm_msp.py run by hand
 before step 99c. Their inputs do not overlap.)

Queen-contiguity adjacency between the crop-group panel units (2011 / PC11
parent districts, keyed `unit_key` exactly as in scripts/98_crop_group_panel.py),
built from docs/data/districts.geojson.

The geojson holds 735 present-day district polygons. Each carries a property
`k`, the normalised crop-record district key (the `key()` of script 96). The
polygon is followed from that key to its crop-record district and then through
the script 72 crosswalk to its 2011 parent. The 52 polygons with no `k` are
matched by hand to their 2011 parent below (post-2011 districts and renames).
Polygons are dissolved to units, then two units are neighbours when they share
any boundary point (queen).

The geojson was simplified at 1.2 km and rounded to 0.001 degrees for the web,
so shared borders no longer coincide exactly. Contiguity is therefore tested
with a tolerance: each unit is buffered by TOL metres (in an equal-distance
Indian projection) and pairs whose buffers intersect are neighbours. The
neighbour count at 0, 250, 500 and 1000 m is logged so the choice can be seen.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/97_crop_group_adjacency.py

Outputs (outputs/crop_groups/):
  crop_group_adjacency_edges.csv      unit_i, unit_j, both directions, with tolerance tested
  crop_group_adjacency_matrix.csv     square 0/1 matrix, rows and columns = unit_key
  crop_group_adjacency_unmatched.csv  panel units without a polygon, polygons without a unit
  crop_group_polygon_match.csv        every geojson feature -> unit_key, with method
  crop_group_adjacency_log.txt
"""
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
OUT = REPO + "/outputs/crop_groups"
GJ = REPO + "/docs/data/districts.geojson"
os.makedirs(OUT, exist_ok=True)
TOL = 500  # metres

LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# same key() as script 96, which is what wrote `k` into the geojson
ALIAS = {"orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
         "jammu and kashmir": "jammu kashmir", "nct of delhi": "delhi",
         "andaman and nicobar islands": "andaman nicobar islands",
         "andaman and nicobar": "andaman nicobar islands",
         "dadra and nagar haveli": "dadra nagar haveli", "daman and diu": "daman diu",
         "the dadra and nagar haveli and daman and diu": "dadra nagar haveli"}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
    return re.sub(r"\s+", " ", s)


def key(state, dist):
    return ALIAS.get(norm(state), norm(state)) + "|" + norm(dist)


# ------------------------------------------------------------------ unit keys (same rule as script 98)
cw = pd.read_csv(COV + "/district_crosswalk.csv")
cw["unit_key"] = cw["shrug_key"].where(cw["shrug_key"].notna(), cw["div_key"])
cw["k"] = [key(*d.split("|", 1)) for d in cw["div_key"]]
k2unit = {}
for k, u in zip(cw["k"], cw["unit_key"]):
    k2unit.setdefault(k, set()).add(u)
amb = {k: v for k, v in k2unit.items() if len(v) > 1}
if amb:
    w("normalised crop keys pointing at more than one unit: {}".format(amb))
k2unit = {k: sorted(v)[0] for k, v in k2unit.items()}
UNITS = sorted(cw["unit_key"].unique())

# polygons with no `k`: present-day district -> 2011 parent unit
MANUAL = {
    ("Punjab", "Sahibzada Ajit Singh Nag*"): "PUNJAB|SAHIBZADA AJIT SINGH NAGAR",
    ("Punjab", "Sri Muktsar Sahib"): "PUNJAB|MUKTSAR",
    ("Haryana", "Gurugram"): "HARYANA|GURGAON",
    ("Haryana", "Nuh"): "HARYANA|MEWAT",
    ("Uttar Pradesh", "Prayagraj"): "UTTAR PRADESH|ALLAHABAD",
    ("Uttar Pradesh", "Bhadohi"): "UTTAR PRADESH|SANT RAVIDAS NAGAR BHADOHI",
    ("Arunachal Pradesh", "Upper Dibang Valley"): "ARUNACHAL PRADESH|DIBANG VALLEY",
    ("Manipur", "Kamjong"): "MANIPUR|UKHRUL",
    ("Manipur", "Kakching"): "MANIPUR|THOUBAL",
    ("Manipur", "Jiribam"): "MANIPUR|IMPHAL EAST",
    ("Manipur", "Kangpokpi"): "MANIPUR|SENAPATI",
    ("Manipur", "Tengnoupal"): "MANIPUR|CHANDEL",
    ("Manipur", "Noney"): "MANIPUR|TAMENGLONG",
    ("Manipur", "Pherzawl"): "MANIPUR|CHURACHANDPUR",
    ("Mizoram", "Hnahthial"): "MIZORAM|LUNGLEI",
    ("Mizoram", "Saitual"): "MIZORAM|AIZAWL",
    ("Mizoram", "Khawzawl"): "MIZORAM|CHAMPHAI",
    ("Tripura", "Unokoti"): "TRIPURA|NORTH TRIPURA",
    ("West Bengal", "South Twenty Four Pargan*"): "WEST BENGAL|SOUTH TWENTY FOUR PARGANAS",
    ("West Bengal", "North Twenty Four Pargan*"): "WEST BENGAL|NORTH TWENTY FOUR PARGANAS",
    ("West Bengal", "Medinipur West"): "WEST BENGAL|PASCHIM MEDINIPUR",
    ("Gujarat", "Chota Udaipur"): "GUJARAT|VADODARA",
    ("Daman & Diu", "Daman"): "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU|DAMAN",
    ("Daman & Diu", "Diu"): "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU|DIU",
    ("Dadra & Nagar Have", "Dadra & Nagar Haveli"): "DADRA NAGAR HAVELI|DADRA NAGAR HAVELI",
    ("Andhra Pradesh", "Y.S.R."): "ANDHRA PRADESH|YSR KADAPA",
    ("Andaman & Nicobar", "North  & Middle Andaman"): "ANDAMAN NICOBAR ISLANDS|NORTH MIDDLE ANDAMAN",
    ("Andaman & Nicobar", "South Andaman"): "ANDAMAN NICOBAR ISLANDS|SOUTH ANDAMAN",
    ("Andaman & Nicobar", "Nicobars"): "ANDAMAN NICOBAR ISLANDS|NICOBARS",
    ("Telangana", "Jogulamba Gadwal"): "ANDHRA PRADESH|MAHBUBNAGAR",
    ("Telangana", "Rajanna Sircilla"): "ANDHRA PRADESH|KARIMNAGAR",
    ("Telangana", "Ranga Reddy"): "ANDHRA PRADESH|RANGAREDDY",
    ("Telangana", "Warangal Rural"): "ANDHRA PRADESH|WARANGAL",
    ("Telangana", "Warangal Urban"): "ANDHRA PRADESH|WARANGAL",
    ("Telangana", "Yadadri Bhuvanagiri"): "ANDHRA PRADESH|NALGONDA",
    ("Telangana", "Bhadradri Kothagudem"): "ANDHRA PRADESH|KHAMMAM",
    ("Ladakh", "Leh"): "JAMMU KASHMIR|LEH LADAKH",
}
# no crop records at all for these, so no unit: Delhi's 11 districts, Kolkata,
# Lakshadweep, and the two districts of Pakistan-administered Kashmir
NO_UNIT_REASON = {"Delhi": "no Delhi crop records survive script 76 cleaning",
                  "Lakshadweep": "no crop records",
                  ("West Bengal", "Kolkata"): "no crop records (urban)",
                  ("Jammu & Kashmir", "Muzaffarabad"): "Pakistan-administered, no crop records",
                  ("Jammu & Kashmir", "Mirpur"): "Pakistan-administered, no crop records"}
bad_manual = [v for v in MANUAL.values() if v not in UNITS]
assert not bad_manual, bad_manual

# ------------------------------------------------------------------ match polygons
g = gpd.read_file(GJ)[["n", "s", "k", "geometry"]]
w("geojson features: {}".format(len(g)))
# the geojson's `k` sends the Bengaluru Rural polygon to Bengaluru Urban (a fuzzy
# match in script 82); its 2011 parent is Bangalore Rural, so the name wins here
OVERRIDE = {("Karnataka", "Bengaluru Rural"): "KARNATAKA|BANGALORE RURAL"}
assert all(v in UNITS for v in OVERRIDE.values())
units_, how_ = [], []
for _, r in g.iterrows():
    if (r["s"], r["n"]) in OVERRIDE:
        units_.append(OVERRIDE[(r["s"], r["n"])]); how_.append("name_override"); continue
    if isinstance(r["k"], str) and r["k"] in k2unit:
        units_.append(k2unit[r["k"]]); how_.append("k_via_crosswalk"); continue
    if (r["s"], r["n"]) in MANUAL:
        units_.append(MANUAL[(r["s"], r["n"])]); how_.append("manual_parent"); continue
    units_.append(None)
    reason = NO_UNIT_REASON.get((r["s"], r["n"]), NO_UNIT_REASON.get(r["s"], "unresolved"))
    how_.append("no_unit: " + reason)
g["unit_key"] = units_
g["match"] = how_
w("polygon match: " + str(g["match"].value_counts().to_dict()))
g.drop(columns="geometry").to_csv(OUT + "/crop_group_polygon_match.csv", index=False)
assert not (g["match"] == "no_unit: unresolved").any(), g[g["match"] == "no_unit: unresolved"]

# ------------------------------------------------------------------ dissolve to units
gm = g[g["unit_key"].notna()].copy()
gm["geometry"] = gm["geometry"].buffer(0)
U = gm.dissolve(by="unit_key", as_index=False)[["unit_key", "geometry"]]
U = U.to_crs(7755)  # WGS 84 / India NSF LCC, metres
w("units with a polygon: {} of {} panel units".format(len(U), len(UNITS)))


def neighbours(tol):
    B = U.copy()
    if tol > 0:
        B["geometry"] = B.geometry.buffer(tol)
    j = gpd.sjoin(B, B, how="inner", predicate="intersects")
    j = j[j["unit_key_left"] != j["unit_key_right"]]
    return set(zip(j["unit_key_left"], j["unit_key_right"]))


E = {}
for tol in [0, 250, 500, 1000]:
    E[tol] = neighbours(tol)
    deg = pd.Series([a for a, _ in E[tol]]).value_counts().reindex(U["unit_key"]).fillna(0)
    w("tolerance {:>4} m: {:>5} undirected links, mean degree {:.2f}, isolated units {}".format(
        tol, len(E[tol]) // 2, deg.mean(), int((deg == 0).sum())))
edges = E[TOL]
gained = sorted({tuple(sorted(e)) for e in E[TOL] - E[0]})
w("links present at {} m but not at 0 m: {}".format(TOL, len(gained)))
for a, b in gained[:30]:
    w("   {}  --  {}".format(a, b))
extra = sorted({tuple(sorted(e)) for e in E[1000] - E[TOL]})
w("links that 1000 m would add on top of {} m: {}".format(TOL, len(extra)))
for a, b in extra[:30]:
    w("   {}  --  {}".format(a, b))

# shared border length, for weighting if wanted (on the buffered geometry, approximate)
Ui = U.set_index("unit_key")["geometry"]
rows = []
for a, b in sorted(edges):
    inter = Ui[a].buffer(TOL).intersection(Ui[b].buffer(TOL))
    # an intersection strip of width ~2*TOL: length ~ area / (2*TOL)
    rows.append({"unit_i": a, "unit_j": b, "approx_shared_border_km":
                 round(inter.area / (2 * TOL) / 1000, 2),
                 "touch_at_0m": int((a, b) in E[0])})
ed = pd.DataFrame(rows)
ed.to_csv(OUT + "/crop_group_adjacency_edges.csv", index=False)

# square matrix over ALL panel units (units without a polygon are all-zero rows)
M = pd.DataFrame(0, index=UNITS, columns=UNITS, dtype=np.int8)
for a, b in edges:
    M.loc[a, b] = 1
assert (M.values == M.values.T).all()
M.index.name = "unit_key"
M.to_csv(OUT + "/crop_group_adjacency_matrix.csv")

deg = M.sum(axis=1)
iso = sorted(set(U["unit_key"]) & set(deg[deg == 0].index))
nopoly = sorted(set(UNITS) - set(U["unit_key"]))
um = ([{"type": "panel unit without polygon", "key": u, "detail": ""} for u in nopoly]
      + [{"type": "polygon without panel unit", "key": "{} | {}".format(r["s"], r["n"]),
          "detail": r["match"].replace("no_unit: ", "")}
         for _, r in g[g["unit_key"].isna()].iterrows()]
      + [{"type": "unit with polygon but no neighbour", "key": u, "detail": "island"}
         for u in iso])
pd.DataFrame(um).to_csv(OUT + "/crop_group_adjacency_unmatched.csv", index=False)
w("")
w("panel units without a polygon ({}): {}".format(len(nopoly), "; ".join(nopoly)))
w("polygons without a unit ({}): {}".format(
    int(g["unit_key"].isna().sum()),
    "; ".join("{}|{}".format(s, n) for s, n in g.loc[g["unit_key"].isna(), ["s", "n"]].values)))
w("isolated units at {} m ({}): {}".format(TOL, len(iso), "; ".join(iso)))
w("degree distribution at {} m: {}".format(TOL, deg[deg > 0].describe().round(2).to_dict()))
# polygons from different present-day states pulled into one unit, as a sanity check
st_mix = gm.groupby("unit_key")["s"].nunique()
w("units whose polygons span more than one present-day state: {}".format(
    "; ".join("{} ({})".format(u, ", ".join(sorted(gm.loc[gm.unit_key == u, 's'].unique())))
              for u in st_mix[st_mix > 1].index)))
with open(OUT + "/crop_group_adjacency_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print("WROTE adjacency to", OUT)
