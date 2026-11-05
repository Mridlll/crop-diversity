"""
90_market_map_layers.py

District values and state outlines for the map, state and typology scenes on
docs/markets.html.

For each of the 606 districts in the market analysis set (the same set and the
same keys as 89_market_typology.py): district and state names, the ten facility
village shares, D0 (count of crops) and D1 (effective number of crops) from the
corrected indices. State outlines are dissolved from districts.geojson and
simplified, written as rings in lon/lat.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/90_market_map_layers.py

Output:
  docs/data/site_market_map.json
"""
import json
import re

import geopandas as gpd
import pandas as pd

REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"
GEO = REPO + "/docs/data/districts.geojson"
OUT = REPO + "/docs/data/site_market_map.json"

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
    a = ALIAS.get(norm(state), norm(state))
    return a + "|" + norm(dist)


def state_label(s):
    return " ".join(x if x in ("and",) else x.capitalize() for x in s.lower().split())


def title(s):
    return " ".join(x if x in ("and", "of") else x.capitalize() for x in str(s).lower().split())


fp = pd.read_csv(COV + "/final_panel.csv")
mk = pd.read_csv(COV + "/market_covariates.csv")
for t in (fp, mk):
    for c in ["pc11_state_id", "pc11_district_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
n0 = len(fp)
d = fp.merge(mk, on=["pc11_state_id", "pc11_district_id"], how="left")
assert len(d) == n0
d = d[d["in_final"] & d["idx_output_market"].notna()].copy().reset_index(drop=True)
assert len(d) == 606
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
DV = ["D0_richness", "D1_exp_shannon"]
d = d.drop(columns=DV).merge(corr[["district_key"] + DV], on="district_key", how="left")
assert d["D1_exp_shannon"].notna().all()
d["k"] = [key(*v.split("|", 1)) for v in d["div_key"]]
assert d["k"].is_unique

FAC = [("mandi", "m_mandi_vshare"), ("regular market", "m_regular_market_vshare"),
       ("weekly haat", "m_weekly_haat_vshare"), ("fertiliser shop", "m_fert_shop_vshare"),
       ("seed centre", "m_seed_centre_vshare"), ("soil testing", "m_soil_test_vshare"),
       ("custom hiring", "m_custom_hire_vshare"), ("cold storage", "m_storage_vshare"),
       ("farm-gate processing", "m_farmgate_proc_vshare"), ("FPO", "m_fpo_vshare")]

g = gpd.read_file(GEO)
names = dict(zip(g["k"], g["n"]))

dist = {}
for _, r in d.iterrows():
    dist[r["k"]] = {
        "n": names.get(r["k"]) or title(r["div_key"].split("|", 1)[1]),
        "s": state_label(r["state_name"]),
        "f": [None if pd.isna(r[c]) else round(float(r[c]), 4) for _, c in FAC],
        "D0": round(float(r["D0_richness"]), 2),
        "D1": round(float(r["D1_exp_shannon"]), 3),
        "m": bool(r["k"] in names),
    }

# state outlines: dissolve district polygons on the geojson state name
g["geometry"] = g["geometry"].buffer(0)
st = g.dissolve(by="s").reset_index()
st["geometry"] = st["geometry"].simplify(0.02, preserve_topology=True)
rings = []
for _, r in st.iterrows():
    geom = r["geometry"]
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for p in polys:
        if p.area < 0.01:
            continue
        rings.append([[round(x, 3), round(y, 3)] for x, y in p.exterior.coords])

out = {"note": "Ten facility village shares (order as in fac), D0 count of crops and D1 "
               "effective number of crops for the 606 market-analysis districts, keyed as "
               "districts.geojson k; m = has a polygon. state_rings: simplified state outlines.",
       "fac": [a for a, _ in FAC], "districts": dist, "state_rings": rings}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print("districts", len(dist), "on map", sum(v["m"] for v in dist.values()), "rings", len(rings))
