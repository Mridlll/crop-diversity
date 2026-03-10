"""
Generate Timeline GeoJSON Data
==============================
Loads the district-year diversity panel and shapefile, merges them,
and exports a single GeoJSON with per-year properties for each district.

Usage:
    python scripts/64_generate_timeline_data.py

Output:
    docs/data/district_timeline.geojson
"""

import json
import re
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from thefuzz import fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
PANEL_PATH = BASE / "outputs" / "crop_diversity_analysis" / "district_year_diversity_panel.csv"
SHP_PATH = BASE / "Package_Maps_Share_20251120_FINAL" / "shapefiles" / "in_district.shp"
OUT_DIR = BASE / "docs" / "data"
OUT_PATH = OUT_DIR / "district_timeline.geojson"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading panel data...")
panel = pd.read_csv(PANEL_PATH)
print(f"  Panel rows: {len(panel)}, years: {sorted(panel['year_start'].unique())}")

print("Loading shapefile...")
gdf = gpd.read_file(SHP_PATH)
print(f"  Shapefile districts: {len(gdf)}")

# ---------------------------------------------------------------------------
# 2. District matching (same MANUAL_MAP as other scripts)
# ---------------------------------------------------------------------------
def normalize(s):
    s = str(s).upper().strip()
    s = s.replace("&", "AND")
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

MANUAL_MAP = {
    ("ANDAMAN AND NICOBAR", "NICOBARS"): ("ANDAMAN AND NICOBAR ISLANDS", "NICOBARS"),
    ("ANDAMAN AND NICOBAR", "NORTH AND MIDDLE ANDAMAN"): ("ANDAMAN AND NICOBAR ISLANDS", "NORTH AND MIDDLE ANDAMAN"),
    ("ANDAMAN AND NICOBAR", "SOUTH ANDAMAN"): ("ANDAMAN AND NICOBAR ISLANDS", "SOUTH ANDAMANS"),
    ("DADRA AND NAGAR HAVE", "DADRA AND NAGAR HAVELI"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DADRA AND NAGAR HAVELI"),
    ("DAMAN AND DIU", "DAMAN"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DAMAN"),
    ("DAMAN AND DIU", "DIU"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DIU"),
    ("ANDHRA PRADESH", "SRI POTTI SRIRAMULU NELL"): ("ANDHRA PRADESH", "SRI POTTI SRIRAMULU NELLORE"),
    ("ANDHRA PRADESH", "VISAKHAPATNAM"): ("ANDHRA PRADESH", "VISAKHAPATANAM"),
    ("ANDHRA PRADESH", "YSR"): ("ANDHRA PRADESH", "KADAPA"),
    ("ARUNACHAL PRADESH", "LEPA RADA"): ("ARUNACHAL PRADESH", "LEPARADA"),
    ("ASSAM", "KAMRUP METROPOLITAN"): ("ASSAM", "KAMRUP METRO"),
    ("ASSAM", "MORIGAON"): ("ASSAM", "MARIGAON"),
    ("BIHAR", "PURBA CHAMPARAN"): ("BIHAR", "PURBI CHAMPARAN"),
    ("CHHATTISGARH", "BAMETARA"): ("CHHATTISGARH", "BEMETARA"),
    ("CHHATTISGARH", "GARIABAND"): ("CHHATTISGARH", "GARIYABAND"),
    ("CHHATTISGARH", "JANJGIR CHAMPA"): ("CHHATTISGARH", "JANJGIRCHAMPA"),
    ("CHHATTISGARH", "KABEERDHAM"): ("CHHATTISGARH", "KABIRDHAM"),
    ("CHHATTISGARH", "KORIYA"): ("CHHATTISGARH", "KOREA"),
    ("CHHATTISGARH", "UTTAR BASTAR KANKER"): ("CHHATTISGARH", "KANKER"),
    ("GUJARAT", "CHOTA UDAIPUR"): ("GUJARAT", "CHHOTAUDEPUR"),
    ("GUJARAT", "THE DANGS"): ("GUJARAT", "DANG"),
    ("HARYANA", "GURUGRAM"): ("HARYANA", "GURGAON"),
    ("HARYANA", "NUH"): ("HARYANA", "MEWAT"),
    ("JAMMU AND KASHMIR", "BANDIPORE"): ("JAMMU AND KASHMIR", "BANDIPORA"),
    ("JAMMU AND KASHMIR", "BARAMULA"): ("JAMMU AND KASHMIR", "BARAMULLA"),
    ("JAMMU AND KASHMIR", "PUNCH"): ("JAMMU AND KASHMIR", "POONCH"),
    ("JAMMU AND KASHMIR", "RAJOURI"): ("JAMMU AND KASHMIR", "RAJAURI"),
    ("JAMMU AND KASHMIR", "SHUPIYAN"): ("JAMMU AND KASHMIR", "SHOPIAN"),
    ("JHARKHAND", "KODARMA"): ("JHARKHAND", "KODERMA"),
    ("JHARKHAND", "PASHCHIMI SINGHBHUM"): ("JHARKHAND", "WEST SINGHBHUM"),
    ("JHARKHAND", "PURBI SINGHBHUM"): ("JHARKHAND", "EAST SINGHBUM"),
    ("JHARKHAND", "SAHIBGANJ"): ("JHARKHAND", "SAHEBGANJ"),
    ("JHARKHAND", "SARAIKELAKHARSAWAN"): ("JHARKHAND", "SARAIKELA KHARSAWAN"),
    ("KARNATAKA", "BANGALORE"): ("KARNATAKA", "BANGALORE RURAL"),
    ("KARNATAKA", "BENGALURU RURAL"): ("KARNATAKA", "BANGALORE RURAL"),
    ("KARNATAKA", "DAVANAGERE"): ("KARNATAKA", "DAVANGERE"),
    ("KARNATAKA", "YADGIR"): ("KARNATAKA", "YADAGIRI"),
    ("LADAKH", "LEH"): ("LADAKH", "LEH LADAKH"),
    ("MADHYA PRADESH", "EAST NIMAR"): ("MADHYA PRADESH", "KHANDWA"),
    ("MADHYA PRADESH", "WEST NIMAR"): ("MADHYA PRADESH", "KHARGONE"),
    ("MADHYA PRADESH", "NARSIMHAPUR"): ("MADHYA PRADESH", "NARSINGHPUR"),
    ("MAHARASHTRA", "AHMADNAGAR"): ("MAHARASHTRA", "AHMEDNAGAR"),
    ("MAHARASHTRA", "BID"): ("MAHARASHTRA", "BEED"),
    ("MAHARASHTRA", "BULDANA"): ("MAHARASHTRA", "BULDHANA"),
    ("MAHARASHTRA", "GONDIYA"): ("MAHARASHTRA", "GONDIA"),
    ("MAHARASHTRA", "RAIGARH"): ("MAHARASHTRA", "RAIGAD"),
    ("MEGHALAYA", "RIBHOI"): ("MEGHALAYA", "RI BHOI"),
    ("ODISHA", "BAUDH"): ("ODISHA", "BOUDH"),
    ("ODISHA", "DEBAGARH"): ("ODISHA", "DEOGARH"),
    ("ODISHA", "NABARANGAPUR"): ("ODISHA", "NABARANGPUR"),
    ("ODISHA", "SUBARNAPUR"): ("ODISHA", "SONEPUR"),
    ("PUDUCHERRY", "PUDUCHERRY"): ("PUDUCHERRY", "PONDICHERRY"),
    ("PUNJAB", "FIROZPUR"): ("PUNJAB", "FIROZEPUR"),
    ("PUNJAB", "SAHIBZADA AJIT SINGH NAG"): ("PUNJAB", "SAS NAGAR"),
    ("PUNJAB", "SRI MUKTSAR SAHIB"): ("PUNJAB", "MUKTSAR"),
    ("RAJASTHAN", "CHITTAURGARH"): ("RAJASTHAN", "CHITTORGARH"),
    ("RAJASTHAN", "DHAULPUR"): ("RAJASTHAN", "DHOLPUR"),
    ("RAJASTHAN", "JALOR"): ("RAJASTHAN", "JALORE"),
    ("RAJASTHAN", "JHUNJHUNUN"): ("RAJASTHAN", "JHUNJHUNU"),
    ("SIKKIM", "EAST DISTRICT"): ("SIKKIM", "GANGTOK"),
    ("SIKKIM", "NORTH DISTRICT"): ("SIKKIM", "MANGAN"),
    ("SIKKIM", "SOUTH DISTRICT"): ("SIKKIM", "NAMCHI"),
    ("SIKKIM", "WEST DISTRICT"): ("SIKKIM", "GYALSHING"),
    ("TAMIL NADU", "KANCHEEPURAM"): ("TAMIL NADU", "KANCHIPURAM"),
    ("TAMIL NADU", "THOOTHUKKUDI"): ("TAMIL NADU", "TUTICORIN"),
    ("TAMIL NADU", "VILUPPURAM"): ("TAMIL NADU", "VILLUPURAM"),
    ("TELANGANA", "BHADRADRI KOTHAGUDEM"): ("TELANGANA", "BHADRADRI"),
    ("TELANGANA", "JOGULAMBA GADWAL"): ("TELANGANA", "JOGULAMBA"),
    ("TELANGANA", "KUMURAM BHEEM ASIFABAD"): ("TELANGANA", "KOMARAM BHEEM ASIFABAD"),
    ("TELANGANA", "NARAYANPET"): ("TELANGANA", "NARAYANAPET"),
    ("TELANGANA", "RAJANNA SIRCILLA"): ("TELANGANA", "RAJANNA"),
    ("TELANGANA", "RANGA REDDY"): ("TELANGANA", "RANGAREDDI"),
    ("TELANGANA", "WARANGAL RURAL"): ("TELANGANA", "WARANGAL"),
    ("TELANGANA", "WARANGAL URBAN"): ("TELANGANA", "HANUMAKONDA"),
    ("TELANGANA", "YADADRI BHUVANAGIRI"): ("TELANGANA", "YADADRI"),
    ("TRIPURA", "SIPAHIJALA"): ("TRIPURA", "SEPAHIJALA"),
    ("TRIPURA", "UNOKOTI"): ("TRIPURA", "UNAKOTI"),
    ("UTTAR PRADESH", "BARA BANKI"): ("UTTAR PRADESH", "BARABANKI"),
    ("UTTAR PRADESH", "BHADOHI"): ("UTTAR PRADESH", "SANT RAVIDAS NAGAR"),
    ("UTTAR PRADESH", "KUSHINAGAR"): ("UTTAR PRADESH", "KUSHI NAGAR"),
    ("UTTAR PRADESH", "MAHRAJGANJ"): ("UTTAR PRADESH", "MAHARAJGANJ"),
    ("UTTAR PRADESH", "PRAYAGRAJ"): ("UTTAR PRADESH", "ALLAHABAD"),
    ("UTTAR PRADESH", "SANT KABIR NAGAR"): ("UTTAR PRADESH", "SANT KABEER NAGAR"),
    ("UTTAR PRADESH", "SHRAWASTI"): ("UTTAR PRADESH", "SHRAVASTI"),
    ("UTTAR PRADESH", "SIDDHARTHNAGAR"): ("UTTAR PRADESH", "SIDDHARTH NAGAR"),
    ("UTTARAKHAND", "GARHWAL"): ("UTTARAKHAND", "PAURI GARHWAL"),
    ("UTTARAKHAND", "HARDWAR"): ("UTTARAKHAND", "HARIDWAR"),
    ("UTTARAKHAND", "RUDRAPRAYAG"): ("UTTARAKHAND", "RUDRA PRAYAG"),
    ("UTTARAKHAND", "UDHAM SINGH NAGAR"): ("UTTARAKHAND", "UDAM SINGH NAGAR"),
    ("UTTARAKHAND", "UTTARKASHI"): ("UTTARAKHAND", "UTTAR KASHI"),
    ("WEST BENGAL", "COOCH BEHAR"): ("WEST BENGAL", "COOCHBEHAR"),
    ("WEST BENGAL", "DARJILING"): ("WEST BENGAL", "DARJEELING"),
    ("WEST BENGAL", "MEDINIPUR WEST"): ("WEST BENGAL", "PASHCHIM MEDINIPUR"),
    ("WEST BENGAL", "NORTH TWENTY FOUR PARGAN"): ("WEST BENGAL", "NORTH 24 PARGANAS"),
    ("WEST BENGAL", "PURULIYA"): ("WEST BENGAL", "PURULIA"),
    ("WEST BENGAL", "SOUTH TWENTY FOUR PARGAN"): ("WEST BENGAL", "SOUTH 24 PARGANAS"),
}

# Build panel lookup: (state_norm, district_norm) -> district_key
panel["state_norm"] = panel["state_name"].apply(normalize)
panel["district_norm"] = panel["district_name"].apply(normalize)

# Build CSV lookup keyed by normalized (state, district)
csv_keys = {}
for key, grp in panel.groupby(["state_norm", "district_norm"]):
    csv_keys[key] = grp["district_key"].iloc[0]

# ---------------------------------------------------------------------------
# 3. Match shapefile districts to panel
# ---------------------------------------------------------------------------
print("Matching shapefile districts...")
match_col = [None] * len(gdf)
exact = manual = fuzzy_cnt = unmatched_cnt = 0

for idx, row in gdf.iterrows():
    shp_st = normalize(row["stname"])
    shp_dt = normalize(row["dtname"])

    if (shp_st, shp_dt) in csv_keys:
        match_col[idx] = csv_keys[(shp_st, shp_dt)]
        exact += 1
        continue

    mapped = MANUAL_MAP.get((shp_st, shp_dt))
    if mapped and mapped in csv_keys:
        match_col[idx] = csv_keys[mapped]
        manual += 1
        continue

    district_only = [(k, v) for k, v in csv_keys.items() if k[1] == shp_dt]
    if len(district_only) == 1:
        match_col[idx] = district_only[0][1]
        exact += 1
        continue

    best_score = 0
    best_key = None
    same_state = [(k, v) for k, v in csv_keys.items() if fuzz.ratio(k[0], shp_st) > 80]
    for (csv_st, csv_dt), dk in same_state:
        score = fuzz.ratio(shp_dt, csv_dt)
        if score > best_score:
            best_score = score
            best_key = dk
    if best_score < 85:
        for (csv_st, csv_dt), dk in csv_keys.items():
            score = fuzz.ratio(shp_dt, csv_dt)
            if score > best_score:
                best_score = score
                best_key = dk
    if best_score >= 85:
        match_col[idx] = best_key
        fuzzy_cnt += 1
    else:
        unmatched_cnt += 1

gdf["district_key"] = match_col
total_matched = exact + manual + fuzzy_cnt
print(f"  Exact: {exact}, Manual: {manual}, Fuzzy: {fuzzy_cnt}, Unmatched: {unmatched_cnt}")
print(f"  Coverage: {total_matched}/{len(gdf)} ({total_matched/len(gdf)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Simplify geometries aggressively
# ---------------------------------------------------------------------------
print("Simplifying geometries (tolerance=0.02)...")
gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.02, preserve_topology=True)
gdf = gdf.to_crs(epsg=4326)

# ---------------------------------------------------------------------------
# 5. Pivot panel data: one row per district, columns per year
# ---------------------------------------------------------------------------
years = sorted(panel["year_start"].unique())
print(f"  Years: {years[0]} to {years[-1]} ({len(years)} years)")

# Create pivot tables for shannon and crop_richness
panel_pivot_shannon = panel.pivot_table(
    index="district_key", columns="year_start", values="shannon_index"
)
panel_pivot_richness = panel.pivot_table(
    index="district_key", columns="year_start", values="crop_richness"
)

# Compute global min/max for consistent color scales
shannon_vals = panel["shannon_index"].dropna()
richness_vals = panel["crop_richness"].dropna()

stats = {
    "shannon_min": float(shannon_vals.min()),
    "shannon_max": float(shannon_vals.max()),
    "richness_min": float(richness_vals.min()),
    "richness_max": float(richness_vals.max()),
    "years": [int(y) for y in years],
}
print(f"  Shannon range: {stats['shannon_min']:.2f} - {stats['shannon_max']:.2f}")
print(f"  Richness range: {stats['richness_min']:.0f} - {stats['richness_max']:.0f}")

# ---------------------------------------------------------------------------
# 6. Build GeoJSON features
# ---------------------------------------------------------------------------
print("Building GeoJSON features...")
features = []

for idx, row in gdf.iterrows():
    geom = row["geometry"]
    if geom is None or geom.is_empty:
        continue

    dk = row.get("district_key")
    props = {
        "name": str(row.get("dtname") or ""),
        "state": str(row.get("stname") or ""),
    }

    # Add per-year Shannon and Richness values
    if dk and dk in panel_pivot_shannon.index:
        for yr in years:
            val = panel_pivot_shannon.at[dk, yr] if yr in panel_pivot_shannon.columns else np.nan
            props[f"s_{yr}"] = round(float(val), 3) if not pd.isna(val) else None
            val2 = panel_pivot_richness.at[dk, yr] if yr in panel_pivot_richness.columns else np.nan
            props[f"r_{yr}"] = int(val2) if not pd.isna(val2) else None
    else:
        for yr in years:
            props[f"s_{yr}"] = None
            props[f"r_{yr}"] = None

    feat = {
        "type": "Feature",
        "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
        "properties": props,
    }
    features.append(feat)

geojson_data = {
    "type": "FeatureCollection",
    "metadata": stats,
    "features": features,
}

print(f"  {len(features)} features built")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(geojson_data, f)

file_size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
print(f"\nGeoJSON saved to: {OUT_PATH}")
print(f"File size: {file_size_mb:.1f} MB")
