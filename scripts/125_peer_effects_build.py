"""
125_peer_effects_build.py

Base data for the district peer-effects page. Builds, on constant 2011 parent units
(the crop-group panel units of script 98, adjacency of script 97):

  1. unit attributes: centroid (EPSG 7755), present-day state (Telangana split out),
     2011 state, in_606, covariates (long-run rainfall mean and CV, elevation, TRI,
     census irrigated share 2001 and 2011)
  2. unit x year panel: nine crop-group shares of annual GCA, crop-level effective
     number of crops D1 = exp(Shannon) over all APY crops, and the crop/season
     series the spread episodes need (cotton, soybean, rabi maize share of rabi
     area, rice, kharif rice)
  3. unit x period table: the 1998-2002 and 2015-19 windows (year_start 1998-2002,
     2015-2019), area-summed shares over reliable years, D1 mean of annual values
  4. pair table: every pair of units with centroids within 600 km, adjacency,
     same-state flag (present-day states), distance, crop-mix similarity in both
     windows, D1 gap, covariate gaps

Analysis window: scripts/_years.py (1998-99 to 2019-20; 1997-98 dropped).
Annual values flagged `flag_annual_unreliable` (script 98) are set to missing;
seasonal values also need `flag_season_unreliable == 0`.

No APCNF/PMDS data. Inputs:
  E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv (raw APY, script 98 cleaning)
  outputs/crop_groups/crop_group_panel.csv, crop_group_child_map.csv, crop_group_units.csv,
     crop_group_adjacency_edges.csv, crop_group_polygon_match.csv, rainfall_district_year.csv
  outputs/revision_nov8/census_irrigation/census_irrigation_panel_wide.csv
  outputs/revision_nov8/irrigation_v2/div_district_year.csv (validation of D1 only)
  docs/data/districts.geojson (read only)

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/125_peer_effects_build.py
Outputs: outputs/revision_nov8/peer_effects/pe_units.csv, pe_unit_year.csv,
         pe_unit_period.csv, pe_pairs.csv.gz, pe_build_log.txt
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _years import FIRST_YEAR, LAST_YEAR  # noqa: E402

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG = os.path.join(ROOT, "outputs", "crop_groups")
OUT = os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects")
os.makedirs(OUT, exist_ok=True)
GROUPS = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton",
          "sugarcane", "other"]
W1 = (1998, 2002)
W2 = (2015, 2019)
MAXKM = 600

LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ------------------------------------------------------------------ raw APY, script 98 cleaning
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name"]:
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
pair = list(zip(df["state_name"], df["district_name"]))
df = df[[p not in BOGUS for p in pair]]
for (s_, d_), new in REMAP.items():
    m = (df["state_name"] == s_) & (df["district_name"] == d_)
    df.loc[m, "state_name"] = new
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
df = df[df["year_start"].between(FIRST_YEAR, LAST_YEAR)]
K = ["district_key", "year", "season", "crop_name"]
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")
w(f"APY rows {FIRST_YEAR}-{LAST_YEAR} after script-98 cleaning: {len(df):,}")

cm = pd.read_csv(os.path.join(CG, "crop_group_child_map.csv"))[["div_key", "unit_key"]]
df = df.merge(cm, left_on="district_key", right_on="div_key", how="left")
w(f"rows without a unit: {df.unit_key.isna().sum()}")
df = df[df.unit_key.notna()]

# crop-level D1 per unit-year (all seasons summed per crop)
cy = df.groupby(["unit_key", "year_start", "crop_name"], as_index=False)["area"].sum()
tot = cy.groupby(["unit_key", "year_start"])["area"].transform("sum")
cy["p"] = cy["area"] / tot
cy["plnp"] = -cy["p"] * np.log(cy["p"])
d1 = cy.groupby(["unit_key", "year_start"]).agg(H=("plnp", "sum"), D0=("crop_name", "nunique")).reset_index()
d1["D1"] = np.exp(d1["H"])

# validation against script 122's district D1 for units with exactly one child
dv = pd.read_csv(os.path.join(ROOT, "outputs", "revision_nov8", "irrigation_v2", "div_district_year.csv"),
                 usecols=["district_key", "year_start", "D1"])
one = cm.groupby("unit_key").div_key.nunique()
one = cm[cm.unit_key.isin(one[one == 1].index)]
chk = dv.merge(one, left_on="district_key", right_on="div_key").merge(
    d1, on=["unit_key", "year_start"], suffixes=("_ref", ""))
chk = chk[chk.year_start.between(FIRST_YEAR, LAST_YEAR)]
w(f"D1 validation vs script 122 (single-child units): {len(chk)} unit-years, "
  f"max abs diff {np.abs(chk.D1 - chk.D1_ref).max():.2e}")

# crop and season series for the spread episodes
def crop_area(mask, name):
    a = df[mask].groupby(["unit_key", "year_start"])["area"].sum().rename(name)
    return a


SEAS_K = df.season.isin(["Kharif", "Autumn", "Winter"])
SEAS_R = df.season.eq("Rabi")
series = pd.concat([
    crop_area(df.crop_name.eq("Soyabean"), "a_soybean"),
    crop_area(df.crop_name.eq("Cotton(Lint)"), "a_cotton"),
    crop_area(df.crop_name.eq("Maize") & SEAS_R, "a_maize_rabi"),
    crop_area(df.crop_name.eq("Rice") & SEAS_K, "a_rice_kharif"),
    crop_area(df.crop_name.eq("Rice") & (SEAS_R | df.season.eq("Summer")), "a_rice_dry"),
    crop_area(SEAS_R, "a_rabi_total"),
    crop_area(df.crop_name.isin(["Rapeseed &Mustard"]), "a_mustard"),
    crop_area(df.crop_name.isin(["Gram"]), "a_gram"),
    crop_area(df.crop_name.isin(["Arhar/Tur"]), "a_tur"),
    crop_area(df.crop_name.isin(["Moong(Green Gram)"]), "a_moong"),
    crop_area(df.crop_name.isin(["Potato"]), "a_potato"),
], axis=1).fillna(0).reset_index()

# ------------------------------------------------------------------ crop-group panel (annual) + flags
pa = pd.read_csv(os.path.join(CG, "crop_group_panel.csv"),
                 usecols=["unit_key", "state", "year_start", "season", "gca_ha", "season_area_ha",
                          "flag_annual_unreliable", "flag_season_unreliable", "in_606"]
                 + [f"area_ha_{g}" for g in GROUPS] + [f"share_gca_{g}" for g in GROUPS])
ann = pa[pa.season == "annual"].copy()
rab = pa[pa.season == "rabi"][["unit_key", "year_start", "flag_season_unreliable"]].rename(
    columns={"flag_season_unreliable": "flag_season_unreliable_rabi"})
ann = ann[ann.year_start.between(FIRST_YEAR, LAST_YEAR)]
uy = ann.merge(d1[["unit_key", "year_start", "D1", "D0"]], on=["unit_key", "year_start"], how="left")
uy = uy.merge(series, on=["unit_key", "year_start"], how="left").merge(rab, on=["unit_key", "year_start"], how="left")
for c in series.columns:
    if c.startswith("a_"):
        uy[c] = uy[c].fillna(0)
uy["sh_soybean"] = uy.a_soybean / uy.gca_ha
uy["sh_cotton"] = uy.a_cotton / uy.gca_ha
uy["sh_rice"] = uy.share_gca_rice
uy["sh_maize_rabi_of_rabi"] = np.where(uy.a_rabi_total > 0, uy.a_maize_rabi / uy.a_rabi_total, np.nan)
uy["sh_maize_rabi"] = uy.a_maize_rabi / uy.gca_ha
uy["sh_rice_kharif"] = uy.a_rice_kharif / uy.gca_ha
uy["sh_mustard"] = uy.a_mustard / uy.gca_ha
uy["sh_gram"] = uy.a_gram / uy.gca_ha
uy["sh_tur"] = uy.a_tur / uy.gca_ha
uy["sh_moong"] = uy.a_moong / uy.gca_ha
uy["sh_potato"] = uy.a_potato / uy.gca_ha
uy["reliable"] = (uy.flag_annual_unreliable == 0).astype(int)
uy["reliable_rabi"] = ((uy.flag_annual_unreliable == 0) & (uy.flag_season_unreliable_rabi == 0)).astype(int)
w(f"unit-years {FIRST_YEAR}-{LAST_YEAR}: {len(uy)}, reliable {uy.reliable.sum()}, "
  f"D1 missing {uy.D1.isna().sum()}")

# ------------------------------------------------------------------ units: polygons, centroid, present-day state
units = pd.read_csv(os.path.join(CG, "crop_group_units.csv"))[["unit_key", "state", "district", "in_606"]]
gj = gpd.read_file(os.path.join(ROOT, "docs", "data", "districts.geojson"))[["n", "s", "geometry"]]
pm = pd.read_csv(os.path.join(CG, "crop_group_polygon_match.csv"))[["n", "s", "unit_key"]]
gj = gj.merge(pm, on=["n", "s"], how="left")
gj = gj[gj.unit_key.notna()].copy()
gj["geometry"] = gj.geometry.buffer(0)
gjm = gj.to_crs(7755)
gjm["a"] = gjm.area
# present-day state = state holding most of the unit's polygon area
st_now = gjm.groupby(["unit_key", "s"]).a.sum().reset_index().sort_values("a").drop_duplicates("unit_key", keep="last")
st_now = st_now.set_index("unit_key").s
U = gjm.dissolve(by="unit_key", as_index=False)[["unit_key", "geometry"]]
U["area_km2"] = U.area / 1e6
cen = U.geometry.centroid
U["cx"], U["cy"] = cen.x, cen.y
cll = gpd.GeoSeries(cen, crs=7755).to_crs(4326)
U["lon"], U["lat"] = cll.x.round(4), cll.y.round(4)
units = units.merge(U.drop(columns="geometry"), on="unit_key", how="left")
units["state_now"] = units.unit_key.map(st_now).str.upper().str.replace("&", "AND", regex=False)
units["state_now"] = units["state_now"].fillna(units["state"])
w(f"units {len(units)}, with polygon {units.cx.notna().sum()}; 606-units with polygon "
  f"{units[units.in_606 == 1].cx.notna().sum()}")
w("units whose present-day state differs from 2011 state: "
  + "; ".join(f"{u} -> {s}" for u, s, s0 in units[["unit_key", "state_now", "state"]].values
             if isinstance(s, str) and s.replace(" ", "") != s0.replace(" ", "")
             and not (s0.startswith("ANDAMAN") or "DADRA" in s0 or "JAMMU" in s0)))

# rainfall long-run means: district-level IMD mean / sd (1990-2019 base), area-weighted to units
rf = pd.read_csv(os.path.join(CG, "rainfall_district_year.csv"),
                 usecols=["district_key", "year", "rain_jjas", "rain_annual", "rain_jfmam_next", "area_km2"])
rf = rf.merge(cm, left_on="district_key", right_on="div_key").drop_duplicates(["unit_key", "district_key", "year"])
rf = rf[rf.year.between(1990, 2019)]
def wmean(d, c):
    ok = d[c].notna()
    return np.average(d.loc[ok, c], weights=d.loc[ok, "area_km2"].fillna(1)) if ok.any() else np.nan
ry = rf.groupby(["unit_key", "year"]).apply(lambda d: pd.Series({
    "r_ann": wmean(d, "rain_annual"), "r_jjas": wmean(d, "rain_jjas"),
    "r_rabi": wmean(d, "rain_jfmam_next")})).reset_index()
rstat = ry.groupby("unit_key").agg(rain_ann_mean=("r_ann", "mean"), rain_ann_sd=("r_ann", "std"),
                                   rain_jjas_mean=("r_jjas", "mean"), rain_rabi_mean=("r_rabi", "mean"),
                                   n_rain_years=("r_ann", "count")).reset_index()
rstat["rain_ann_cv"] = rstat.rain_ann_sd / rstat.rain_ann_mean
units = units.merge(rstat, on="unit_key", how="left")

ci = pd.read_csv(os.path.join(ROOT, "outputs", "revision_nov8", "census_irrigation", "census_irrigation_panel_wide.csv"))
ci = ci[["shrug_key", "elevation_mean_m", "tri_mean", "irr_share_2001", "irr_share_2011",
         "district_usable_2001", "district_usable_2011"]].rename(columns={"shrug_key": "unit_key"})
ci.loc[ci.district_usable_2001 != True, "irr_share_2001"] = np.nan  # noqa: E712
ci.loc[ci.district_usable_2011 != True, "irr_share_2011"] = np.nan  # noqa: E712
ci["irr_share"] = ci.irr_share_2011.fillna(ci.irr_share_2001)
units = units.merge(ci.drop(columns=["district_usable_2001", "district_usable_2011"]), on="unit_key", how="left")
for c in ["rain_ann_mean", "rain_ann_cv", "elevation_mean_m", "tri_mean", "irr_share"]:
    w(f"  covariate {c}: present for {units.loc[units.in_606 == 1, c].notna().sum()} of 605 sample units")
units.to_csv(os.path.join(OUT, "pe_units.csv"), index=False)

uy = uy.merge(units[["unit_key", "state_now"]], on="unit_key", how="left")
keep_cols = (["unit_key", "state", "state_now", "year_start", "in_606", "gca_ha", "reliable", "reliable_rabi",
              "D1", "D0"] + [f"share_gca_{g}" for g in GROUPS] + [f"area_ha_{g}" for g in GROUPS]
             + [c for c in uy.columns if c.startswith("sh_")] + ["a_rabi_total"])
uy[keep_cols].to_csv(os.path.join(OUT, "pe_unit_year.csv"), index=False)

# ------------------------------------------------------------------ window table
rel2 = uy[uy.reliable == 1]
rows = []
for lab, (a, b) in [("w9802", W1), ("w1519", W2)]:
    d = rel2[rel2.year_start.between(a, b)]
    g = d.groupby("unit_key")
    t = pd.DataFrame({"n_years": g.size(), "gca_mean": g.gca_ha.mean(), "D1": g.D1.mean()})
    for gr in GROUPS:
        t["s_" + gr] = g[f"area_ha_{gr}"].sum() / g.gca_ha.sum()
    for c in ["soybean", "cotton", "mustard", "gram", "tur", "moong", "potato", "maize_rabi"]:
        t["s_" + c] = g["a_" + c].sum() / g.gca_ha.sum()
    t["window"] = lab
    rows.append(t.reset_index())
up = pd.concat(rows, ignore_index=True)
up = up[up.n_years >= 2]
assert np.allclose(up[[f"s_{g}" for g in GROUPS]].sum(1), 1, atol=1e-6)
up.to_csv(os.path.join(OUT, "pe_unit_period.csv"), index=False)
w(f"window table: " + ", ".join(f"{k} {v} units" for k, v in up.window.value_counts().items()))

# ------------------------------------------------------------------ pair table
ed = pd.read_csv(os.path.join(CG, "crop_group_adjacency_edges.csv"))
adj = set(zip(ed.unit_i, ed.unit_j))
uu = units[units.cx.notna() & (units.in_606 == 1)].reset_index(drop=True)
xy = uu[["cx", "cy"]].values
D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000
ii, jj = np.where(np.triu(D <= MAXKM, 1))
P = pd.DataFrame({"unit_i": uu.unit_key.values[ii], "unit_j": uu.unit_key.values[jj], "dist_km": D[ii, jj]})
P["adjacent"] = [int((a, b) in adj) for a, b in zip(P.unit_i, P.unit_j)]
sn = uu.set_index("unit_key").state_now
s11 = uu.set_index("unit_key").state
P["state_i"], P["state_j"] = P.unit_i.map(sn).values, P.unit_j.map(sn).values
P["state11_i"], P["state11_j"] = P.unit_i.map(s11).values, P.unit_j.map(s11).values
P["same_state"] = (P.state_i == P.state_j).astype(int)
P["same_state_2011"] = (P.state11_i == P.state11_j).astype(int)
cov = units.set_index("unit_key")
for c, nm in [("rain_ann_mean", "d_rain"), ("rain_ann_cv", "d_raincv"), ("elevation_mean_m", "d_elev"),
              ("tri_mean", "d_tri"), ("irr_share", "d_irr")]:
    P[nm] = np.abs(P.unit_i.map(cov[c]).values - P.unit_j.map(cov[c]).values)
P["d_rain"] = P.d_rain / 100  # per 100 mm
P["d_elev"] = P.d_elev / 100  # per 100 m
for lab in ["w9802", "w1519"]:
    t = up[up.window == lab].set_index("unit_key")
    A = t.reindex(P.unit_i)[[f"s_{g}" for g in GROUPS]].values
    B = t.reindex(P.unit_j)[[f"s_{g}" for g in GROUPS]].values
    P[f"sim_{lab}"] = 1 - 0.5 * np.abs(A - B).sum(1)
    P[f"dD1_{lab}"] = np.abs(t.reindex(P.unit_i).D1.values - t.reindex(P.unit_j).D1.values)
    for g in ["rice", "wheat", "pulses", "millets", "oilseeds", "maize", "cotton"]:
        P[f"g_{g}_{lab}"] = np.abs(t.reindex(P.unit_i)[f"s_{g}"].values - t.reindex(P.unit_j)[f"s_{g}"].values)
# change in mix: similarity of the change vectors is not a share; use correlation-free L1 of changes
t1 = up[up.window == "w9802"].set_index("unit_key")[[f"s_{g}" for g in GROUPS]]
t2 = up[up.window == "w1519"].set_index("unit_key")[[f"s_{g}" for g in GROUPS]]
dch = (t2 - t1).dropna()
A, B = dch.reindex(P.unit_i).values, dch.reindex(P.unit_j).values
P["l1_change_gap"] = 0.5 * np.abs(A - B).sum(1)
P.to_csv(os.path.join(OUT, "pe_pairs.csv.gz"), index=False, compression="gzip")
w(f"pairs within {MAXKM} km among {len(uu)} sample units with centroids: {len(P):,}; adjacent "
  f"{P.adjacent.sum()} (same state {int(P[P.adjacent == 1].same_state.sum())}, cross-border "
  f"{int((P[P.adjacent == 1].same_state == 0).sum())})")
w(f"adjacent pairs: mean centroid distance {P[P.adjacent == 1].dist_km.mean():.1f} km "
  f"(5-95%: {P[P.adjacent == 1].dist_km.quantile(.05):.0f}-{P[P.adjacent == 1].dist_km.quantile(.95):.0f})")

with open(os.path.join(OUT, "pe_build_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
