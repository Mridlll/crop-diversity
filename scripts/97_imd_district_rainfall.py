"""
97_imd_district_rainfall.py

District-level seasonal rainfall, 1990-2020, from IMD's 0.25-degree gridded daily
rainfall (Pai et al. 2014), aggregated to the district polygons of
docs/data/districts.geojson and keyed by that file's 'k' property.

Source: IMD Pune, https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
        binary files fetched by POST to https://imdpune.gov.in/cmpg/Griddata/rainfall.php
        (form field rain=<year>), the endpoint imdlib 0.1.21 uses.
Raw files: data_raw/imd_rain_025/Rainfall_ind<YYYY>_rfp25.grd (1990-2021, gitignored)
        float32, (days, 129 lat, 135 lon), lat 6.5..38.5, lon 66.5..100.0, -999 = no data.

Method
  1. Daily grids summed to monthly totals per cell (mm).
  2. A cell is used if it is valid on every day of 1990-2021 (fixed land mask).
  3. Each 0.25-degree cell is a polygon; it is intersected with each district polygon in
     an equal-area projection. A district's rainfall is the intersection-area-weighted
     mean of its cells. Polygons that share one 'k' (the site maps several census
     districts onto one crop-record district) are pooled by area.
  4. Seasons: jjas = Jun-Sep of year t; ond = Oct-Dec of year t; jfmam = Jan-May of
     year t; jfmam_next = Jan-May of year t+1 (the rabi season of agricultural year
     t to t+1, matching crop-year labels like '2009-10' -> 2009); annual = Jan-Dec t.
  5. Anomalies against each district's own 1990-2020 mean and SD (31 years):
     pct departure = 100 (R - mean) / mean ; z = (R - mean) / SD.
     Percent departure and its flags are blanked where the district's season mean is
     below 10 mm (arid-season divide-by-near-zero).
  6. Flags: dry20 = pct <= -20, wet20 = pct >= +20; IMD category labels using IMD's
     district thresholds (large excess >= 60, excess 20..59, normal -19..19,
     deficient -59..-20, large deficient -99..-60, no rain -100), applied to the
     own-1990-2020 baseline rather than IMD's LPA.

Outputs (outputs/crop_groups/):
  rainfall_district_year.csv      one row per k x year (1990-2020)
  rainfall_polygon_year.csv       same seasonal totals per map polygon, incl. unmatched
  rainfall_allindia_check.csv     all-India area-weighted JJAS departure by year
"""
import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

warnings.filterwarnings("ignore")

REPO = r"D:/crop-diversity"
RAW = REPO + "/data_raw/imd_rain_025"
GEO = REPO + "/docs/data/districts.geojson"
MONO = REPO + "/docs/data/site_monoculture.json"
OUT = REPO + "/outputs/crop_groups"
os.makedirs(OUT, exist_ok=True)

Y0, Y1 = 1990, 2020          # output years and baseline
YLAST = 2021                 # needed for jfmam_next of 2020
NLAT, NLON = 129, 135
LATS = 6.5 + 0.25 * np.arange(NLAT)
LONS = 66.5 + 0.25 * np.arange(NLON)
EA = "+proj=aea +lat_1=12 +lat_2=32 +lat_0=22 +lon_0=82 +datum=WGS84 +units=m"
MIN_BASE_MM = 10.0

# ---------------------------------------------------------------- 1. monthly grids
years = list(range(Y0, YLAST + 1))
monthly = np.zeros((len(years), 12, NLAT, NLON), dtype=np.float64)
valid = np.ones((NLAT, NLON), dtype=bool)
for i, y in enumerate(years):
    f = "{}/Rainfall_ind{}_rfp25.grd".format(RAW, y)
    nd = 366 if pd.Timestamp(y, 12, 31).dayofyear == 366 else 365
    a = np.fromfile(f, dtype="<f4")
    assert a.size == nd * NLAT * NLON, (y, a.size)
    a = a.reshape(nd, NLAT, NLON)
    miss = a < -900
    valid &= ~miss.any(axis=0)
    a = np.where(miss, 0.0, a)
    mon = pd.date_range("{}-01-01".format(y), periods=nd).month.values
    for m in range(1, 13):
        monthly[i, m - 1] = a[mon == m].sum(axis=0)
    # sanity: missing pattern should be the same every day (fixed land mask)
    partial = miss.any(axis=0) & ~miss.all(axis=0)
    if partial.sum():
        print("  {}: {} cells missing on some but not all days".format(y, int(partial.sum())))
print("valid cells (all days, all years):", int(valid.sum()))

# ---------------------------------------------------------------- 1b. cell QC
# (i) Dead cells: the grid carries 0.0 (not -999) where no station fed a cell, so a
#     cell can report a whole season of zero rain inside a wet region (Saiha, Mizoram,
#     1990-2006). A cell-season block that totals exactly 0 is set to missing when the
#     cell averages more than 50 mm in its non-zero years AND some cell within 3 cells
#     (about 80 km) received more than 200 mm (JJAS) or 100 mm (OND, Jan-May).
# (ii) Bull's-eyes: a cell whose seasonal total is more than 2.5x its own 1990-2020 mean
#      AND more than 2x the median of that ratio over the surrounding 7x7 cells. These
#      are kept in the totals but flagged (share of district area in such cells).
from scipy.ndimage import generic_filter

BLOCKS = {"jjas": [5, 6, 7, 8], "ond": [9, 10, 11], "jfmam": [0, 1, 2, 3, 4]}
from scipy.ndimage import maximum_filter

DEAD_NBR = {"jjas": 200.0, "ond": 100.0, "jfmam": 100.0}
dead = {}
for b, ms in BLOCKS.items():
    tot = monthly[:, ms].sum(1)                                   # (Y, lat, lon)
    nz = np.where(tot > 0, tot, np.nan)
    nzmean = np.nanmean(nz, axis=0)                               # mean over non-zero years
    nbmax = np.stack([maximum_filter(np.where(valid, tot[t], 0), size=7, mode="constant")
                      for t in range(len(years))])
    dz = (tot == 0) & (nzmean > 50) & (nbmax > DEAD_NBR[b]) & valid[None]
    dead[b] = dz
    for t in range(len(years)):
        for m in ms:
            monthly[t, m][dz[t]] = np.nan
    print("  dead cell-seasons masked, {}: {} (cells affected {})".format(
        b, int(dz.sum()), int(dz.any(0).sum())))

YRS = np.arange(Y0, Y1 + 1)
nb = len(YRS)
spike = {}
for b, ms in BLOCKS.items():
    tot = monthly[:nb, ms].sum(1)
    clim = np.nanmean(tot, axis=0)
    r = np.where(valid & (clim > 50), tot / np.where(clim > 0, clim, np.nan), np.nan)
    sp = np.zeros_like(r, dtype=bool)
    for t in range(nb):
        nm = generic_filter(r[t], np.nanmedian, size=7, mode="constant", cval=np.nan)
        sp[t] = (r[t] > 2.5) & (r[t] > 2 * nm)
    spike[b] = sp
    print("  bull's-eye cell-seasons flagged, {}: {}".format(b, int(sp.sum())))

# ---------------------------------------------------------------- 2. cell polygons
ii, jj = np.where(valid)
cells = gpd.GeoDataFrame(
    {"ci": ii, "cj": jj},
    geometry=[box(LONS[j] - 0.125, LATS[i] - 0.125, LONS[j] + 0.125, LATS[i] + 0.125)
              for i, j in zip(ii, jj)], crs=4326).to_crs(EA)

# ---------------------------------------------------------------- 3. districts
g = gpd.read_file(GEO)[["n", "s", "k", "geometry"]]
g["pid"] = np.arange(len(g))
g["geometry"] = g.geometry.make_valid()
g = g.to_crs(EA)
g["parea"] = g.geometry.area
ov = gpd.overlay(g[["pid", "geometry"]], cells, how="intersection", keep_geom_type=True)
ov["w"] = ov.geometry.area
ov = ov[ov["w"] > 0]
cov = ov.groupby("pid")["w"].sum()
g["cov_frac"] = g["pid"].map(cov).fillna(0) / g["parea"]
g["n_cells"] = g["pid"].map(ov.groupby("pid").size()).fillna(0).astype(int)
print("polygons: {}, with grid coverage: {}".format(len(g), int((g["n_cells"] > 0).sum())))

# weighted monthly series per polygon: (polygon, year, month)
flat = monthly[:, :, ov["ci"].values, ov["cj"].values]          # (Y, 12, nov)
W = ov["w"].values
pids = ov["pid"].values
P = np.full((len(g), len(years), 12), np.nan)
PCOV = np.zeros((len(g), len(years), 12))        # share of polygon weight with data
PSPK = {b: np.zeros((len(g), nb)) for b in BLOCKS}  # share of weight in bull's-eye cells
spk_flat = {b: spike[b][:, ov["ci"].values, ov["cj"].values] for b in BLOCKS}
for p in np.unique(pids):
    sel = pids == p
    w = W[sel] / W[sel].sum()
    v = flat[:, :, sel]
    ok = ~np.isnan(v)
    wv = (ok * w).sum(-1)
    PCOV[p] = wv
    P[p] = np.where(wv >= 0.5, np.nansum(np.nan_to_num(v) * w, -1) / np.where(wv > 0, wv, 1),
                    np.nan)
    for b in BLOCKS:
        PSPK[b][p] = spk_flat[b][:, sel] @ w


def seasons(arr):
    """arr (..., Y, 12) -> dict of (..., Y) seasonal totals for years Y0..Y1"""
    t = slice(0, Y1 - Y0 + 1)
    nx = slice(1, Y1 - Y0 + 2)
    return {
        "jjas": arr[..., t, 5:9].sum(-1),
        "ond": arr[..., t, 9:12].sum(-1),
        "jfmam": arr[..., t, 0:5].sum(-1),
        "jfmam_next": arr[..., nx, 0:5].sum(-1),
        "annual": arr[..., t, :].sum(-1),
    }


SEAS = ["jjas", "ond", "jfmam", "jfmam_next", "annual"]

# polygon-level file
S = seasons(P)
rows = []
for p in range(len(g)):
    if g.at[p, "n_cells"] == 0:
        continue
    for t, y in enumerate(YRS):
        rows.append({"pid": p, "polygon_state": g.at[p, "s"], "polygon_district": g.at[p, "n"],
                     "k": g.at[p, "k"], "year": int(y),
                     **{"rain_" + s: round(float(S[s][p, t]), 2) for s in SEAS}})
pol = pd.DataFrame(rows)
pol.to_csv(OUT + "/rainfall_polygon_year.csv", index=False)

# ---------------------------------------------------------------- 4. pool to k
# Polygon -> key membership. Start from the geojson 'k', then
#  (a) undo four fuzzy mis-keys in districts.geojson (a census polygon given a
#      neighbouring district's key), and
#  (b) add crop-record keys that districts.geojson leaves without a polygon, built from
#      the polygons that make up the district (2011 boundaries for parent keys).
# A polygon may serve several keys (e.g. 'uttar pradesh|uttar kashi' and
# 'uttarakhand|uttar kashi' are both crop-record spellings of one district).
UNKEY = {("Punjab", "Firozpur"), ("Karnataka", "Bengaluru Rural"),
         ("Tamil Nadu", "Viluppuram"), ("Tamil Nadu", "Kancheepuram")}
TG = "Telangana"
ALIAS = {
    # 2011-parent keys (Telangana split of 2016, and other post-2011 splits)
    "andhra pradesh|karimnagar": [(TG, "Karimnagar"), (TG, "Jagitial"), (TG, "Peddapalli"),
                                  (TG, "Rajanna Sircilla")],
    "andhra pradesh|warangal": [(TG, "Warangal Rural"), (TG, "Warangal Urban"), (TG, "Jangoan"),
                                (TG, "Mahabubabad"), (TG, "Jayashankar"), (TG, "Mulugu")],
    "andhra pradesh|khammam": [(TG, "Khammam"), (TG, "Bhadradri Kothagudem")],
    "andhra pradesh|nalgonda": [(TG, "Nalgonda"), (TG, "Suryapet"), (TG, "Yadadri Bhuvanagiri")],
    "andhra pradesh|nizamabad": [(TG, "Nizamabad"), (TG, "Kamareddy")],
    "andhra pradesh|medak": [(TG, "Medak"), (TG, "Sangareddy"), (TG, "Siddipet")],
    "andhra pradesh|rangareddi": [(TG, "Ranga Reddy"), (TG, "Vikarabad"), (TG, "Medchal Malkajgiri")],
    "telangana|adilabad": [(TG, "Adilabad"), (TG, "Kumuram Bheem Asifabad"), (TG, "Mancherial"),
                           (TG, "Nirmal")],
    "telangana|mahabubnagar": [(TG, "Mahabubnagar"), (TG, "Jogulamba Gadwal"), (TG, "Wanaparthy"),
                               (TG, "Nagarkurnool"), (TG, "Narayanpet")],
    "andhra pradesh|kadapa": [("Andhra Pradesh", "Y.S.R.")],
    "punjab|firozepur": [("Punjab", "Firozpur"), ("Punjab", "Fazilka")],
    "tamil nadu|kanchipuram": [("Tamil Nadu", "Kancheepuram"), ("Tamil Nadu", "Chengalpattu")],
    "tamil nadu|villupuram": [("Tamil Nadu", "Viluppuram"), ("Tamil Nadu", "Kallakurichi")],
    "west bengal|pashchim medinipur": [("West Bengal", "Medinipur West"), ("West Bengal", "Jhargram")],
    # renames and spellings
    "karnataka|bangalore rural": [("Karnataka", "Bengaluru Rural")],
    "punjab|muktsar": [("Punjab", "Sri Muktsar Sahib")],
    "punjab|s a s nagar": [("Punjab", "Sahibzada Ajit Singh Nag*")],
    "haryana|gurgaon": [("Haryana", "Gurugram")],
    "haryana|mewat": [("Haryana", "Nuh")],
    "uttar pradesh|allahabad": [("Uttar Pradesh", "Prayagraj")],
    "uttar pradesh|sant ravidas nagar": [("Uttar Pradesh", "Bhadohi")],
    "arunachal pradesh|dibang valley": [("Arunachal Pradesh", "Upper Dibang Valley")],
    "ladakh|leh ladakh": [("Ladakh", "Leh")],
    "jammu kashmir|kargil": [("Ladakh", "Kargil")],
    "uttarakhand|rudra prayag": [("Uttarakhand", "Rudraprayag")],
    "uttarakhand|udam singh nagar": [("Uttarakhand", "Udham Singh Nagar")],
    "uttarakhand|uttar kashi": [("Uttarakhand", "Uttarkashi")],
    "west bengal|north 24 parganas": [("West Bengal", "North Twenty Four Pargan*")],
    "west bengal|south 24 parganas": [("West Bengal", "South Twenty Four Pargan*")],
    "gujarat|chhotaudepur": [("Gujarat", "Chota Udaipur")],
    "tripura|unakoti": [("Tripura", "Unokoti")],
    "madhya pradesh|dakshin bastar dantewada": [("Chhattisgarh", "Dakshin Bastar Dantewada")],
    "madhya pradesh|kabirdham": [("Chhattisgarh", "Kabeerdham")],
    "madhya pradesh|korea": [("Chhattisgarh", "Koriya")],
    "telangana|bhadradri": [(TG, "Bhadradri Kothagudem")],
    "telangana|jogulamba": [(TG, "Jogulamba Gadwal")],
    "telangana|rajanna": [(TG, "Rajanna Sircilla")],
    "telangana|rangareddi": [(TG, "Ranga Reddy")],
    "telangana|yadadri": [(TG, "Yadadri Bhuvanagiri")],
    "telangana|hanumakonda": [(TG, "Warangal Urban")],
    "telangana|warangal": [(TG, "Warangal Rural")],
    "dadra nagar haveli|dadra and nagar haveli": [("Dadra & Nagar Have", "Dadra & Nagar Haveli")],
    "dadra nagar haveli|daman": [("Daman & Diu", "Daman")],
    "dadra nagar haveli|diu": [("Daman & Diu", "Diu")],
}
pidx = {(s, n): p for p, s, n in zip(g["pid"], g["s"], g["n"])}
memb = [(p, k, "geojson") for p, k, s, n in zip(g["pid"], g["k"], g["s"], g["n"])
        if isinstance(k, str) and (s, n) not in UNKEY and k not in ALIAS]
for k, lst in ALIAS.items():
    for sn in lst:
        assert sn in pidx, ("alias polygon not found", k, sn)
        memb.append((pidx[sn], k, "alias"))
memb = pd.DataFrame(memb, columns=["pid", "k", "key_source"])
memb = memb.merge(g[["pid", "parea", "cov_frac", "n_cells", "n"]], on="pid")
memb = memb[memb["n_cells"] > 0]
gk = memb
K = sorted(gk["k"].unique())
Kmon = np.full((len(K), len(years), 12), np.nan)
KSPK = {b: np.zeros((len(K), nb)) for b in BLOCKS}
meta = []
for a, k in enumerate(K):
    sub = gk[gk["k"] == k]
    w = (sub["parea"] * sub["cov_frac"]).values
    w = w / w.sum()
    v = P[sub["pid"].values]                                  # (npoly, Y, 12)
    ok = ~np.isnan(v)
    wv = np.tensordot(w, ok.astype(float), axes=1)
    Kmon[a] = np.where(wv >= 0.5, np.tensordot(w, np.nan_to_num(v), axes=1)
                       / np.where(wv > 0, wv, 1), np.nan)
    for b in BLOCKS:
        KSPK[b][a] = w @ PSPK[b][sub["pid"].values]
    meta.append({"k": k, "key_source": sub["key_source"].iloc[0], "n_polygons": len(sub),
                 "n_cells": int(sub["n_cells"].sum()),
                 "area_km2": round(float(sub["parea"].sum()) / 1e6, 1),
                 "grid_cover_frac": round(float((sub["parea"] * sub["cov_frac"]).sum()
                                                / sub["parea"].sum()), 3),
                 "polygons": "; ".join(sub["n"].astype(str))})
meta = pd.DataFrame(meta)
SK = seasons(Kmon)

# ---------------------------------------------------------------- 5. validation
# Against IMD's official subdivision monthly series (station based, 1901-2015; the
# data.gov.in "Sub_Division_IMD" table, mirrored at
# github.com/SAMUR274/rainfall-in-india-1901-2015-1-). Subdivisions that coincide with
# states, plus two built from district lists. A subdivision-year whose JJAS percent
# departure (each series against its own 1990-2015 mean) differs by more than 30
# points flags every key that has a polygon in that subdivision.
SUBF = REPO + "/data_raw/imd_subdivision/rainfall_india_1901_2015.csv"
SUB_STATE = {
    "TAMIL NADU": ["Tamil Nadu", "Puducherry"], "KERALA": ["Kerala"], "PUNJAB": ["Punjab"],
    "BIHAR": ["Bihar"], "ORISSA": ["Odisha"], "JHARKHAND": ["Jharkhand"],
    "CHHATTISGARH": ["Chhattisgarh"], "UTTARAKHAND": ["Uttarakhand"],
    "HIMACHAL PRADESH": ["Himachal Pradesh"], "HARYANA DELHI & CHANDIGARH":
    ["Haryana", "Delhi", "Chandigarh"], "ARUNACHAL PRADESH": ["Arunachal Pradesh"],
    "ASSAM & MEGHALAYA": ["Assam", "Meghalaya"], "TELANGANA": ["Telangana"],
    "NAGA MANI MIZO TRIPURA": ["Nagaland", "Manipur", "Mizoram", "Tripura"],
}
SUB_DIST = {
    "SAURASHTRA & KUTCH": ("Gujarat", ["kachchh", "kutch", "jamnagar", "dwarka", "porbandar",
                           "junagadh", "gir somnath", "amreli", "bhavnagar", "botad",
                           "rajkot", "morbi", "surendranagar"]),
    "MADHYA MAHARASHTRA": ("Maharashtra", ["pune", "satara", "sangli", "kolhapur", "solapur",
                           "ahmednagar", "nashik", "dhule", "nandurbar", "jalgaon"]),
}
pw = (g["parea"] * g["cov_frac"]).values
psub = pd.Series("", index=g["pid"].values)
bad_sy = set()
if os.path.exists(SUBF):
    off = pd.read_csv(SUBF)
    vrows = []
    for sub in list(SUB_STATE) + list(SUB_DIST):
        if sub in SUB_STATE:
            m = g["s"].isin(SUB_STATE[sub]).values
        else:
            st, names = SUB_DIST[sub]
            m = ((g["s"] == st) & g["n"].str.lower().apply(
                lambda v: any(x in v for x in names))).values
        m &= g["n_cells"].values > 0
        psub[g["pid"].values[m]] = sub
        X = S["jjas"][m]
        okx = ~np.isnan(X)
        ours = np.nansum(X * pw[m, None], 0) / (okx * pw[m, None]).sum(0)
        o = off[off["SUBDIVISION"] == sub].set_index("YEAR")["Jun-Sep"]
        for t, y in enumerate(YRS):
            if y in o.index:
                vrows.append({"subdivision": sub, "year": int(y), "n_polygons": int(m.sum()),
                              "grid_jjas": round(float(ours[t]), 1), "official_jjas": float(o[y])})
    val = pd.DataFrame(vrows)
    val["ratio"] = (val["grid_jjas"] / val["official_jjas"]).round(3)
    for c in ["grid_jjas", "official_jjas"]:
        val["pct_" + c] = (100 * (val[c] / val.groupby("subdivision")[c].transform("mean") - 1)).round(1)
    val["gap_pp"] = (val["pct_grid_jjas"] - val["pct_official_jjas"]).round(1)
    val["bad"] = val["gap_pp"].abs() > 30
    val.to_csv(OUT + "/rainfall_subdivision_validation.csv", index=False)
    bad_sy = set(zip(val.loc[val["bad"], "subdivision"], val.loc[val["bad"], "year"]))
    summ = val.groupby("subdivision").apply(lambda t: pd.Series({
        "n": len(t), "corr_level": t["grid_jjas"].corr(t["official_jjas"]),
        "corr_anom": t["pct_grid_jjas"].corr(t["pct_official_jjas"]),
        "median_ratio": t["ratio"].median(),
        "worst_year": int(t.loc[t["gap_pp"].abs().idxmax(), "year"]),
        "worst_gap_pp": t["gap_pp"].abs().max()})).round(3)
    print("\nVALIDATION vs IMD subdivision JJAS, 1990-2015")
    print(summ.to_string())
    print("pooled anomaly correlation: {:.3f}; median per-subdivision {:.3f}".format(
        val["pct_grid_jjas"].corr(val["pct_official_jjas"]), summ["corr_anom"].median()))
    print("subdivision-years with |gap| > 30 pp:")
    print(val[val["bad"]].to_string(index=False))
else:
    print("\nVALIDATION skipped: subdivision file not found")

# key -> subdivisions its polygons fall in
ksubs = gk.assign(sub=gk["pid"].map(psub)).groupby("k")["sub"].apply(
    lambda x: sorted({v for v in x if v}))
SUBFLAG = np.array([[any((sb, int(y)) in bad_sy for sb in ksubs.get(k, []))
                     for y in YRS] for k in K])

# ---------------------------------------------------------------- 6. anomalies and flags
SPK_MIN = 0.10   # flag a key-season if >= 10% of its area sits in bull's-eye cells
QC = {"jjas": (KSPK["jjas"] >= SPK_MIN) | SUBFLAG,
      "ond": KSPK["ond"] >= SPK_MIN,
      "jfmam": KSPK["jfmam"] >= SPK_MIN}
nxt = np.zeros_like(QC["jfmam"])
nxt[:, :-1] = QC["jfmam"][:, 1:]
# Jan-May 2021 (jfmam_next of 2020) has no bull's-eye test: outside the baseline years
QC["jfmam_next"] = nxt
QC["annual"] = QC["jjas"] | QC["ond"] | QC["jfmam"]


def imd_cat(p):
    if np.isnan(p):
        return ""
    if p <= -99.5:
        return "no_rain"
    if p <= -59.5:
        return "large_deficient"
    if p <= -19.5:
        return "deficient"
    if p < 19.5:
        return "normal"
    if p < 59.5:
        return "excess"
    return "large_excess"


out = pd.DataFrame({"k": np.repeat(K, len(YRS)), "year": np.tile(YRS, len(K))})
for s in SEAS:
    R = SK[s]                                   # (K, Y)
    q = QC[s]
    Rc = np.where(q, np.nan, R)                 # baseline excludes flagged years
    mu = np.nanmean(Rc, 1, keepdims=True)
    sd = np.nanstd(Rc, 1, ddof=1, keepdims=True)
    nbase = (~np.isnan(Rc)).sum(1, keepdims=True)
    pct = 100 * (R - mu) / np.where(mu > 0, mu, np.nan)
    pct = np.where(mu >= MIN_BASE_MM, pct, np.nan)
    z = (R - mu) / np.where(sd > 0, sd, np.nan)
    out["rain_" + s] = np.round(R.ravel(), 2)
    out["mean_" + s] = np.round(np.repeat(mu.ravel(), len(YRS)), 2)
    out["sd_" + s] = np.round(np.repeat(sd.ravel(), len(YRS)), 2)
    out["nbase_" + s] = np.repeat(nbase.ravel(), len(YRS))
    out["pct_" + s] = np.round(pct.ravel(), 2)
    out["z_" + s] = np.round(z.ravel(), 3)
    pf = pct.ravel()
    out["dry20_" + s] = np.where(np.isnan(pf), np.nan, (pf <= -20).astype(float))
    out["wet20_" + s] = np.where(np.isnan(pf), np.nan, (pf >= 20).astype(float))
    out["imdcat_" + s] = [imd_cat(v) for v in pf]
    out["qc_" + s] = q.ravel().astype(int)
for b in BLOCKS:
    out["spike_share_" + b] = np.round(KSPK[b].ravel(), 3)
out["qc_subdiv_jjas"] = SUBFLAG.ravel().astype(int)

# attach the analysis key and the 606-sample flag
D = json.load(open(MONO, encoding="utf-8"))["districts"]
out["district_key"] = out["k"].map({k: v["dk"] for k, v in D.items()})
out["in606"] = out["k"].map({k: v["in606"] for k, v in D.items()})
out = out.merge(meta, on="k", how="left")
lead = ["k", "district_key", "in606", "year"]
out = out[lead + [c for c in out.columns if c not in lead]]
out.to_csv(OUT + "/rainfall_district_year.csv", index=False)
print("\nwrote", OUT + "/rainfall_district_year.csv", out.shape)
print("QC-flagged key-years:", {s: int(out["qc_" + s].sum()) for s in SEAS})
print("missing rain values:", {s: int(out["rain_" + s].isna().sum()) for s in SEAS})
print("flagged jjas, by year:", out.groupby("year")["qc_jjas"].sum()[lambda x: x > 0].to_dict())

# ---------------------------------------------------------------- 7. checks
print("\nCOVERAGE")
print("  geojson polygons {}, with a k {}, unique k {}".format(
    len(g), int(g["k"].notna().sum()), g["k"].nunique()))
print("  polygons with no grid cell:", g.loc[g["n_cells"] == 0, "n"].tolist())
print("  k with rainfall:", len(K))
print("  grid cover fraction of k-areas: min {:.3f}, p5 {:.3f}, median {:.3f}".format(
    meta["grid_cover_frac"].min(), meta["grid_cover_frac"].quantile(.05),
    meta["grid_cover_frac"].median()))
print("  low cover (<0.8):", meta.loc[meta["grid_cover_frac"] < 0.8, ["k", "grid_cover_frac"]]
      .values.tolist())
s606 = {k for k, v in D.items() if v["in606"]}
got = set(K)
print("  geojson k values without rainfall:", sorted(set(g["k"].dropna()) - got))
print("  keys from alias table:", int((meta["key_source"] == "alias").sum()))
print("  606 sample: {} of {} have rainfall".format(len(s606 & got), len(s606)))
print("  606 missing:", sorted(s606 - got))
print("  725 crop districts: {} of {} have rainfall".format(len(set(D) & got), len(D)))
print("  725 missing:", sorted(set(D) - got))

# all-India JJAS: each polygon once (keys can share polygons), by gridded area
okp = ~np.isnan(S["jjas"])
ai = np.nansum(S["jjas"] * pw[:, None], 0) / (okp * pw[:, None]).sum(0)
cell_area = np.cos(np.deg2rad(LATS))[:, None] * np.ones(NLON)[None, :]
cw = np.where(valid, cell_area, 0)
gj = monthly[:nb, 5:9].sum(1)
grid_jjas = np.nansum(gj * cw, (1, 2)) / ((~np.isnan(gj)) * cw).sum((1, 2))
chk = pd.DataFrame({"year": YRS, "allindia_jjas_mm_polygons": ai.round(1),
                    "allindia_jjas_mm_grid": grid_jjas.round(1)})
chk["pct_dep_polygons"] = (100 * (ai / ai.mean() - 1)).round(1)
chk["pct_dep_grid"] = (100 * (grid_jjas / grid_jjas.mean() - 1)).round(1)
jj = out.pivot(index="k", columns="year", values="pct_jjas")
qq = out.pivot(index="k", columns="year", values="qc_jjas").astype(bool)
jc = jj.mask(qq)
chk["share_districts_dry20"] = ((jc <= -20).sum(0) / jc.notna().sum(0)).round(3).values
chk["share_districts_wet20"] = ((jc >= 20).sum(0) / jc.notna().sum(0)).round(3).values
chk["share_districts_imd_deficient"] = ((jc <= -19.5).sum(0) / jc.notna().sum(0)).round(3).values
chk.to_csv(OUT + "/rainfall_allindia_check.csv", index=False)
print("\nALL-INDIA JJAS (district shares exclude QC-flagged key-years)")
print(chk.to_string(index=False))
nanpct = {s: int(out.groupby("k")["pct_" + s].apply(lambda x: x.isna().all()).sum()) for s in SEAS}
print("keys with pct blanked (season mean < {} mm): {}".format(MIN_BASE_MM, nanpct))
