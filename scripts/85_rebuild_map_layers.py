"""
85_rebuild_map_layers.py

Rebuilds the data behind the four interactive maps on the corrected basis.

The earlier maps were drawn from a first pass at the indices which counted the
crops a district recorded at any point in the period rather than the crops it
grows in a year. This recomputes every layer on the same cleaning the corrected
indices use: bogus state-district pairs dropped, the partial final year dropped,
duplicate rows collapsed, and every index measured within a year and averaged.

Layers rebuilt:
  diversity      D0, D1, D2, evenness, dominant crop and its share, category shares
  calorie        food energy produced per hectare of cropped area, and the quadrant
  food           share of cropped area under crops that feed people
  timeline       D0 and D1 per district per year, 1997-98 to 2019-20

Outputs:
  docs/data/districts.geojson          the choropleth geometry, all layers attached
  docs/data/timeline.json              per district per year, no geometry
"""
import json
import os
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
SHP = r"E:/CEEW Project/Package_Maps_Share_20251120_FINAL/shapefiles/in_district.shp"
REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"
DOCS = REPO + "/docs/data"

KCAL = json.load(open(REPO + "/scripts/_kcal_factors.json", encoding="utf-8"))
print("energy factors for {} crops".format(len(KCAL)))

# ------------------------------------------------------- the corrected cleaning
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df["area"] > 0]

BOGUS = [("Delhi", "Chandigarh"), ("Goa", "Chandigarh"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Chandigarh"),
         ("Delhi", "Surguja"), ("Goa", "Surguja"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Surguja"),
         ("Andaman And Nicobar Islands", "Purulia")]
REMAP = {("Uttar Pradesh", d): "Uttarakhand" for d in
         ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar",
          "Nainital", "Pauri Garhwal", "Pithoragarh"]}
REMAP.update({("Madhya Pradesh", d): "Chhattisgarh" for d in
              ["Bastar", "Bilaspur", "Dhamtari", "Durg", "Jashpur", "Kanker",
               "Korba", "Mahasamund", "Raipur", "Rajnandgaon", "Surguja"]})
REMAP.update({("Bihar", d): "Jharkhand" for d in ["Deoghar", "Dhanbad", "Garhwa"]})

df = df[[p not in BOGUS for p in zip(df["state_name"], df["district_name"])]]
for (s_, d_), new in REMAP.items():
    df.loc[(df["state_name"] == s_) & (df["district_name"] == d_), "state_name"] = new
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df = df[df["year_start"] <= 2019]
df = (df.sort_values("area", ascending=False)
        .drop_duplicates(subset=["state_name", "district_name", "year", "season",
                                 "crop_name"], keep="first"))
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
print("clean rows: {:,}  districts: {}  years: {}".format(
    len(df), df["district_key"].nunique(), df["year_start"].nunique()))

# ------------------------------------------------------------- energy and food
# A crop feeds people if its type does. Fibre, drugs and narcotics, and fodder do
# not, and every other type does.
FOOD_TYPES = {"Cereals", "Pulses", "Oilseeds", "Fruits", "Vegetable", "Sugar", "Spices"}
df["is_food"] = df["crop_type"].isin(FOOD_TYPES)

prod = pd.to_numeric(df["production"], errors="coerce")
area = pd.to_numeric(df["area"], errors="coerce")

# Yields beyond 200 tonnes a hectare are reporting errors for every crop except
# coconut, whose production is counted in nuts rather than tonnes.
is_coco = df["crop_name"].str.strip().str.lower() == "coconut"
bad = (~is_coco) & ((prod / area) > 200)
print("dropped {} rows with a yield above 200 tonnes a hectare".format(int(bad.sum())))
df, prod, area, is_coco = df[~bad], prod[~bad], area[~bad], is_coco[~bad]

# Coconut is reported as a count of nuts. About 150 g of edible meat a nut puts it
# back into tonnes alongside everything else.
tonnes = prod.where(~is_coco, prod * 0.00015)

df["kcal_100g"] = df["crop_name"].map(KCAL)
df["kcal"] = tonnes * 10000.0 * df["kcal_100g"].fillna(0)
print("crops carrying an energy value: {} of {}".format(
    df.loc[df["kcal_100g"].notna(), "crop_name"].nunique(), df["crop_name"].nunique()))
print("food crops: {} of {}".format(
    df.loc[df["is_food"], "crop_name"].nunique(), df["crop_name"].nunique()))

per_yr = df.groupby(["district_key", "year_start"]).agg(
    area=("area", "sum"),
    food_area=("area", lambda s: s[df.loc[s.index, "is_food"]].sum()),
    kcal=("kcal", "sum")).reset_index()
per_yr["kcal_per_ha"] = np.where(per_yr["area"] > 0, per_yr["kcal"] / per_yr["area"], np.nan)
per_yr["food_share"] = np.where(per_yr["area"] > 0, per_yr["food_area"] / per_yr["area"], np.nan)

energy = per_yr.groupby("district_key").agg(
    kcal_per_ha=("kcal_per_ha", "mean"),
    food_share=("food_share", "mean"),
    cropped_area=("area", "mean")).reset_index()
print("kcal per hectare: p10 {:,.0f}  median {:,.0f}  p90 {:,.0f}".format(
    energy["kcal_per_ha"].quantile(.1), energy["kcal_per_ha"].median(),
    energy["kcal_per_ha"].quantile(.9)))

# -------------------------------------------------------------- top crops
share = (df.groupby(["district_key", "crop_name"])["area"].sum()
           / df.groupby("district_key")["area"].sum()).reset_index(name="sh")
share = share.sort_values("sh", ascending=False)
top = (share.groupby("district_key").head(5)
            .groupby("district_key")
            .apply(lambda g: [[r.crop_name, round(float(r.sh), 4)] for r in g.itertuples()])
            .rename("top"))

# ------------------------------------------------------------- corrected indices
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
idx = corr.merge(energy, on="district_key", how="left").merge(top, on="district_key", how="left")

# the quadrant, recomputed on the corrected effective number of crops
med_d1 = idx["D1_exp_shannon"].median()
med_k = idx["kcal_per_ha"].median()
idx["quadrant"] = np.select(
    [(idx["D1_exp_shannon"] >= med_d1) & (idx["kcal_per_ha"] >= med_k),
     (idx["D1_exp_shannon"] < med_d1) & (idx["kcal_per_ha"] >= med_k),
     (idx["D1_exp_shannon"] >= med_d1) & (idx["kcal_per_ha"] < med_k)],
    ["diverse and energy-rich", "concentrated and energy-rich", "diverse and energy-poor"],
    default="concentrated and energy-poor")
print(idx["quadrant"].value_counts().to_string())

# ------------------------------------------------------------------ geometry
gdf = gpd.read_file(SHP)
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


gdf["k"] = [key(a, b) for a, b in zip(gdf["stname"], gdf["dtname"])]
idx["k"] = [key(v.split("|")[0], v.split("|")[1]) for v in idx["district_key"]]
idx = idx.drop_duplicates("k")

cw = pd.read_csv(COV + "/district_crosswalk.csv").dropna(subset=["shrug_key"])
s2d = {}
for _, r in cw.iterrows():
    s2d.setdefault(key(*r["shrug_key"].split("|")), key(*r["div_key"].split("|")))

import difflib
direct = set(idx["k"])
by_state = {}
for k in direct:
    by_state.setdefault(k.split("|")[0], []).append(k.split("|")[1])
res = []
for k in gdf["k"]:
    if k in direct:
        res.append(k); continue
    if k in s2d and s2d[k] in direct:
        res.append(s2d[k]); continue
    st, dt = k.split("|", 1)
    m = difflib.get_close_matches(dt, by_state.get(st, []), n=1, cutoff=0.86)
    res.append(st + "|" + m[0] if m else None)
gdf["mk"] = res

fp = pd.read_csv(COV + "/final_panel.csv")
mkc = pd.read_csv(COV + "/market_covariates.csv")
for t in (fp, mkc):
    for c in ["pc11_state_id", "pc11_district_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
fp = fp.merge(mkc, on=["pc11_state_id", "pc11_district_id"], how="left")
fp["k"] = [key(v.split("|")[0], v.split("|")[1]) for v in fp["div_key"]]
idx = idx.merge(fp[["k", "irr_share", "ay_src_canal_vshare", "m_weekly_haat_vshare",
                    "m_mandi_vshare", "m_fert_shop_vshare", "m_fpo_vshare",
                    "mean_holding_ha"]].drop_duplicates("k"), on="k", how="left")

m = gdf.merge(idx.rename(columns={"k": "mk"}), on="mk", how="left")
print("matched {} of {} map features ({:.1%})".format(
    int(m["D1_exp_shannon"].notna().sum()), len(gdf),
    m["D1_exp_shannon"].notna().sum() / len(gdf)))

m = m.to_crs(3857)
m["geometry"] = m["geometry"].simplify(1200, preserve_topology=True).buffer(0)
m = m.to_crs(4326)

FIELDS = {"D0": "D0_richness", "D1": "D1_exp_shannon", "D2": "D2_inv_simpson",
          "E": "evenness_D1_D0", "yrs": "n_years", "kcal": "kcal_per_ha",
          "food": "food_share", "quad": "quadrant", "dom": "dominant_crop",
          "domsh": "dominant_crop_share", "irr": "irr_share",
          "canal": "ay_src_canal_vshare", "haat": "m_weekly_haat_vshare",
          "mandi": "m_mandi_vshare", "fert": "m_fert_shop_vshare",
          "fpo": "m_fpo_vshare", "hold": "mean_holding_ha", "area": "cropped_area"}
CATS = [c for c in idx.columns if c.startswith("share_")]


def clean(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 4)


def trim(o, nd=3):
    if isinstance(o, list):
        return [trim(x, nd) for x in o]
    return round(o, nd) if isinstance(o, float) else o


feats = []
for _, r in m.iterrows():
    g = r["geometry"]
    if g is None or g.is_empty:
        continue
    pr = {"n": str(r["dtname"]).title(), "s": str(r["stname"]).title()}
    for short, col in FIELDS.items():
        v = r.get(col)
        pr[short] = v if isinstance(v, str) else clean(v)
    for c in CATS:
        pr[c.replace("share_", "c_")] = clean(r.get(c))
    t = r.get("top")
    pr["top"] = t if isinstance(t, list) else None
    geom = json.loads(gpd.GeoSeries([g], crs=4326).to_json())["features"][0]["geometry"]
    geom["coordinates"] = trim(geom["coordinates"])
    feats.append({"type": "Feature", "properties": pr, "geometry": geom})

os.makedirs(DOCS, exist_ok=True)
with open(DOCS + "/districts.geojson", "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": feats}, f, separators=(",", ":"))
print("WROTE districts.geojson  {} features  {:.2f} MB".format(
    len(feats), os.path.getsize(DOCS + "/districts.geojson") / 1e6))

# ------------------------------------------------------------------ timeline
panel = pd.read_csv(DIV + "/district_year_diversity_panel_corrected.csv")
panel["k"] = [key(v.split("|")[0], v.split("|")[1]) for v in panel["district_key"]]
years = sorted(panel["year_start"].unique())
tl = {}
for k, g in panel.groupby("k"):
    g = g.set_index("year_start")
    tl[k] = {"D0": [clean(g["D0_richness"].get(y)) for y in years],
             "D1": [clean(g["D1_exp_shannon"].get(y)) for y in years]}
with open(DOCS + "/timeline.json", "w", encoding="utf-8") as f:
    json.dump({"years": [int(y) for y in years], "districts": tl}, f, separators=(",", ":"))
print("WROTE timeline.json  {} districts  {} years ({} to {})".format(
    len(tl), len(years), years[0], years[-1]))
