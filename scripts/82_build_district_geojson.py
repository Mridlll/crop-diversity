"""
82_build_district_geojson.py

Builds a simplified district GeoJSON for the site, carrying the corrected
diversity indices and the covariates the maps colour by.

The geometry is simplified and coordinate precision reduced so the file stays
small enough to load on a page. Districts are matched to the analysis by name
using the same normalisation the rest of the pipeline uses.

Output: docs/data/districts.geojson
"""
import json
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd

SHP = r"E:/CEEW Project/Package_Maps_Share_20251120_FINAL/shapefiles/in_district.shp"
REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"
OUT = REPO + "/docs/data/districts.geojson"


def norm(s):
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


ALIAS = {
    "ORISSA": "ODISHA", "PONDICHERRY": "PUDUCHERRY", "UTTARANCHAL": "UTTARAKHAND",
    "JAMMU AND KASHMIR": "JAMMU KASHMIR", "NCT OF DELHI": "DELHI",
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN NICOBAR ISLANDS",
    "ANDAMAN AND NICOBAR": "ANDAMAN NICOBAR ISLANDS",
    "DADRA AND NAGAR HAVELI": "DADRA NAGAR HAVELI",
    "DAMAN AND DIU": "DAMAN DIU",
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA NAGAR HAVELI",
}

print("reading shapefile")
gdf = gpd.read_file(SHP)
print("  {} features, columns: {}".format(len(gdf), list(gdf.columns)[:8]))
SC = "stname" if "stname" in gdf.columns else gdf.columns[0]
DC = "dtname" if "dtname" in gdf.columns else gdf.columns[1]
gdf["k"] = gdf[SC].map(lambda v: ALIAS.get(norm(v), norm(v))) + "|" + gdf[DC].map(norm)

# --- analysis values
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
fp = pd.read_csv(COV + "/final_panel.csv")
mk = pd.read_csv(COV + "/market_covariates.csv")
for t in (fp, mk):
    for c in ["pc11_state_id", "pc11_district_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
fp = fp.merge(mk, on=["pc11_state_id", "pc11_district_id"], how="left")

d = corr.merge(
    fp[["div_key", "irr_share", "ay_src_canal_vshare", "ay_src_ground_vshare",
        "m_weekly_haat_vshare", "m_mandi_vshare", "m_fert_shop_vshare",
        "m_fpo_vshare", "mean_holding_ha", "in_final"]],
    left_on="district_key", right_on="div_key", how="left")
def keyify(v):
    a, b = v.split("|", 1)
    return ALIAS.get(norm(a), norm(a)) + "|" + norm(b)


d["k"] = d["district_key"].map(keyify)
d = d.drop_duplicates("k")
print("  analysis rows: {}".format(len(d)))

# The shapefile carries census-era district names, which are the same names SHRUG
# uses. The verified SHRUG-to-analysis crosswalk therefore does most of the work
# here: match the map to a SHRUG district first, then follow the crosswalk across.
cw = pd.read_csv(COV + "/district_crosswalk.csv").dropna(subset=["shrug_key"])
shrug_to_div = {}
for _, r in cw.iterrows():
    shrug_to_div.setdefault(keyify(r["shrug_key"]), keyify(r["div_key"]))

direct = set(d["k"])
resolved, how = [], []
for k in gdf["k"]:
    if k in direct:
        resolved.append(k); how.append("direct"); continue
    if k in shrug_to_div and shrug_to_div[k] in direct:
        resolved.append(shrug_to_div[k]); how.append("via_crosswalk"); continue
    resolved.append(None); how.append("none")

# fuzzy within state for whatever is left
import difflib
by_state = {}
for k in direct:
    by_state.setdefault(k.split("|")[0], []).append(k.split("|")[1])
for i, (k, r) in enumerate(zip(gdf["k"], resolved)):
    if r is not None:
        continue
    st, dt = k.split("|", 1)
    cand = by_state.get(st, [])
    if not cand:
        continue
    best = difflib.get_close_matches(dt, cand, n=1, cutoff=0.86)
    if best:
        resolved[i] = st + "|" + best[0]
        how[i] = "fuzzy"

gdf["mk"] = resolved
from collections import Counter
print("  match method: {}".format(dict(Counter(how))))

m = gdf.merge(d.rename(columns={"k": "mk"}), on="mk", how="left")
matched = int(m["D1_exp_shannon"].notna().sum())
print("  matched {} of {} map features ({:.1%})".format(
    matched, len(gdf), matched / len(gdf)))
miss = sorted(set(gdf.loc[[r is None for r in resolved], "k"]))
print("  still unmatched ({}): {}".format(len(miss), ", ".join(miss[:18])))

# --- simplify. project to metres so the tolerance is in metres, then back.
print("simplifying geometry")
m = m.to_crs(3857)
m["geometry"] = m["geometry"].simplify(1200, preserve_topology=True)
m = m.to_crs(4326)
m["geometry"] = m["geometry"].buffer(0)

FIELDS = {
    "D0": "D0_richness", "D1": "D1_exp_shannon", "D2": "D2_inv_simpson",
    "E": "evenness_D1_D0", "irr": "irr_share", "canal": "ay_src_canal_vshare",
    "haat": "m_weekly_haat_vshare", "mandi": "m_mandi_vshare",
    "fert": "m_fert_shop_vshare", "fpo": "m_fpo_vshare",
    "hold": "mean_holding_ha", "dom": "dominant_crop", "domsh": "dominant_crop_share",
}

feats = []
for _, r in m.iterrows():
    if r["geometry"] is None or r["geometry"].is_empty:
        continue
    props = {"n": str(r[DC]).title(), "s": str(r[SC]).title()}
    for short, col in FIELDS.items():
        v = r.get(col)
        if isinstance(v, str):
            props[short] = v
        elif v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            props[short] = None
        else:
            props[short] = round(float(v), 4)
    feats.append({"type": "Feature", "properties": props,
                  "geometry": json.loads(
                      gpd.GeoSeries([r["geometry"]], crs=4326).to_json())
                  ["features"][0]["geometry"]})


def trim(o, nd=3):
    if isinstance(o, list):
        return [trim(x, nd) for x in o]
    if isinstance(o, float):
        return round(o, nd)
    return o


for f in feats:
    f["geometry"]["coordinates"] = trim(f["geometry"]["coordinates"])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"type": "FeatureCollection", "features": feats}, fh,
              separators=(",", ":"))
print("WROTE {}  {} features, {:.2f} MB".format(
    OUT, len(feats), os.path.getsize(OUT) / 1e6))
