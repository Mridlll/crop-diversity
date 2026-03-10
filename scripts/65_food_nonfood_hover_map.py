"""
Food vs Non-Food (Cash Crop) Interactive Hover Map
===================================================
Generates a full-screen Folium map with dropdown to switch between:
  Food Crop Area Share, Cereal Share, Pulse Share, Oilseed Share,
  Cash Crop Share, Sugar Share, Fruit & Vegetable Share.
Hover shows compact food vs non-food breakdown per district.

Usage:
    python scripts/65_food_nonfood_hover_map.py

Requires:
    outputs/crop_diversity_analysis/district_diversity_calorie_merged.csv
    Package_Maps_Share_20251120_FINAL/shapefiles/in_district.shp

Output:
    outputs/crop_diversity_analysis/food_nonfood_hover_map.html
    docs/food-nonfood.html
"""

import json
import re
import shutil
import webbrowser
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from thefuzz import fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "outputs" / "crop_diversity_analysis" / "district_diversity_calorie_merged.csv"
SHP_PATH = BASE / "Package_Maps_Share_20251120_FINAL" / "shapefiles" / "in_district.shp"
OUT_PATH = BASE / "outputs" / "crop_diversity_analysis" / "food_nonfood_hover_map.html"
DOCS_PATH = BASE / "docs" / "food-nonfood.html"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH)
gdf = gpd.read_file(SHP_PATH)

# Compute derived columns
df["cash_crop_share"] = df["share_fiber_crops"].fillna(0) + df["share_drugs_and_narcotics"].fillna(0)
df["fruit_veg_share"] = df["share_fruits"].fillna(0) + df["share_vegetable"].fillna(0)
# Non-food share = 1 - food_crop_area_share (already in data)
df["nonfood_area_share"] = 1.0 - df["food_crop_area_share"].fillna(0)

print(f"  CSV districts: {len(df)}")
print(f"  Shapefile districts: {len(gdf)}")

# ---------------------------------------------------------------------------
# 2. District matching (reused from script 63)
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

# Build CSV lookup
csv_keys = {}
for i, row in df.iterrows():
    st = normalize(row["state_name"])
    dt = normalize(row["district_name"])
    csv_keys[(st, dt)] = i

# Match shapefile districts
print("Matching shapefile districts...")
match_col = [None] * len(gdf)
exact = manual = fuzzy_cnt = unmatched = 0

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
    best_idx_val = None
    same_state = [(k, v) for k, v in csv_keys.items() if fuzz.ratio(k[0], shp_st) > 80]
    for (csv_st, csv_dt), csv_idx_val in same_state:
        score = fuzz.ratio(shp_dt, csv_dt)
        if score > best_score:
            best_score = score
            best_idx_val = csv_idx_val
    if best_score < 85:
        for (csv_st, csv_dt), csv_idx_val in csv_keys.items():
            score = fuzz.ratio(shp_dt, csv_dt)
            if score > best_score:
                best_score = score
                best_idx_val = csv_idx_val
    if best_score >= 85:
        match_col[idx] = best_idx_val
        fuzzy_cnt += 1
    else:
        unmatched += 1

gdf["csv_idx"] = match_col
total_matched = exact + manual + fuzzy_cnt
print(f"  Exact: {exact}, Manual: {manual}, Fuzzy: {fuzzy_cnt}, Unmatched: {unmatched}")
print(f"  Coverage: {total_matched}/{len(gdf)} ({total_matched/len(gdf)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Merge data into GeoDataFrame
# ---------------------------------------------------------------------------
merge_cols = [
    "state_name", "district_name",
    "food_crop_area_share", "nonfood_area_share",
    "share_cereals", "share_pulses", "share_oilseeds",
    "share_sugar", "share_fruits", "share_vegetable", "share_spices",
    "share_fiber_crops", "share_drugs_and_narcotics", "share_fodder",
    "cash_crop_share", "fruit_veg_share",
    "dominant_crop", "dominant_crop_share",
    "coconut_dominant", "coconut_kcal_share",
    "food_crop_kcal_share",
    "cereal_kcal_share", "pulse_kcal_share", "oilseed_kcal_share",
    "sugar_kcal_share", "vegetable_kcal_share", "fruit_kcal_share",
    "spice_kcal_share",
]
merge_cols = [c for c in merge_cols if c in df.columns]

for col in merge_cols:
    gdf[col] = None

for idx, row in gdf.iterrows():
    csv_idx_val = row["csv_idx"]
    if csv_idx_val is not None and not pd.isna(csv_idx_val):
        csv_idx_val = int(csv_idx_val)
        for col in merge_cols:
            gdf.at[idx, col] = df.at[csv_idx_val, col]

gdf["coconut_dominant"] = gdf["coconut_dominant"].fillna(False)
gdf["coconut_kcal_share"] = gdf["coconut_kcal_share"].fillna(0)

# Simplify geometries
print("Simplifying geometries...")
gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
gdf = gdf.to_crs(epsg=4326)

# ---------------------------------------------------------------------------
# 4. Build Folium map
# ---------------------------------------------------------------------------
print("Building map...")

def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{float(val) * 100:.1f}%"

def fmt_num(val, fmt=".2f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{float(val):{fmt}}"

# Layer configs for the dropdown
INDEX_CONFIG = {
    "food_crop_area_share": {"label": "Food Crop Area Share", "cmap": "RdYlGn", "type": "continuous"},
    "share_cereals": {"label": "Cereal Share", "cmap": "YlGn", "type": "continuous"},
    "share_pulses": {"label": "Pulse Share", "cmap": "YlGn", "type": "continuous"},
    "share_oilseeds": {"label": "Oilseed Share", "cmap": "YlOrBr", "type": "continuous"},
    "cash_crop_share": {"label": "Cash Crop Share (Fiber + Drugs)", "cmap": "Oranges", "type": "continuous"},
    "share_sugar": {"label": "Sugar Share", "cmap": "PuRd", "type": "continuous"},
    "fruit_veg_share": {"label": "Fruit & Vegetable Share", "cmap": "BuGn", "type": "continuous"},
}

def get_color_scale(values, cmap_name, n_bins=8):
    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return [], []
    vmin, vmax = min(valid), max(valid)
    if vmin == vmax:
        vmax = vmin + 1
    edges = np.linspace(vmin, vmax, n_bins + 1)
    cmap = plt.get_cmap(cmap_name, n_bins)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_bins)]
    return edges, colors

# Build a mini-bar for percentages (inline CSS bar)
def pct_bar(val, color="#4caf50"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:#aaa;">N/A</span>'
    pct = float(val) * 100
    width = min(pct, 100)
    return (
        f'<div style="display:flex;align-items:center;gap:4px;">'
        f'<div style="background:#eee;border-radius:2px;width:60px;height:8px;flex-shrink:0;">'
        f'<div style="background:{color};border-radius:2px;width:{width:.0f}%;height:100%;"></div></div>'
        f'<span style="font-weight:600;font-size:11px;">{pct:.1f}%</span></div>'
    )

def build_tooltip_html(row):
    name = row.get("district_name") or row.get("dtname") or "Unknown"
    state = row.get("state_name") or row.get("stname") or "Unknown"
    dom_crop = row.get("dominant_crop") or "N/A"
    dom_share = fmt_pct(row.get("dominant_crop_share"))

    food_area = row.get("food_crop_area_share")
    nonfood_area = row.get("nonfood_area_share")

    # Coconut badge
    coconut_dom = row.get("coconut_dominant")
    coconut_share = row.get("coconut_kcal_share")
    coconut_badge = ""
    if coconut_dom and str(coconut_dom).lower() not in ("false", "0", "nan", "none", ""):
        pct = fmt_pct(coconut_share)
        coconut_badge = (
            '<div style="display:inline-block;padding:2px 8px;border-radius:3px;'
            'background:#8B4513;color:white;font-size:11px;font-weight:600;margin-bottom:4px;">'
            f'&#x1F965; Coconut-dominant ({pct} kcal)</div>'
        )

    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;font-size:13px;line-height:1.5;min-width:280px;max-width:360px;">
      <div style="font-size:15px;font-weight:700;color:#1a1a2e;border-bottom:2px solid #0f3460;padding-bottom:4px;margin-bottom:4px;">
        {name}
      </div>
      <div style="color:#555;font-size:12px;margin-bottom:4px;">{state}</div>
      {coconut_badge}

      <table style="width:100%;font-size:12px;border-collapse:collapse;margin-top:4px;">
        <tr style="background:#e8f5e9;">
          <td colspan="2" style="padding:3px 6px;font-weight:700;color:#2e7d32;">Food Crops</td>
          <td style="padding:3px 6px;text-align:right;font-weight:700;color:#2e7d32;">{fmt_pct(food_area)}</td>
        </tr>
        <tr><td style="padding:2px 6px;color:#666;">Cereals</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_cereals"), "#66bb6a")}</td></tr>
        <tr style="background:#fafafa;"><td style="padding:2px 6px;color:#666;">Pulses</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_pulses"), "#81c784")}</td></tr>
        <tr><td style="padding:2px 6px;color:#666;">Oilseeds</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_oilseeds"), "#aed581")}</td></tr>
        <tr style="background:#fafafa;"><td style="padding:2px 6px;color:#666;">Sugar</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_sugar"), "#ce93d8")}</td></tr>
        <tr><td style="padding:2px 6px;color:#666;">Fruits</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_fruits"), "#4db6ac")}</td></tr>
        <tr style="background:#fafafa;"><td style="padding:2px 6px;color:#666;">Vegetables</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_vegetable"), "#4dd0e1")}</td></tr>
        <tr><td style="padding:2px 6px;color:#666;">Spices</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_spices"), "#ffb74d")}</td></tr>
      </table>

      <table style="width:100%;font-size:12px;border-collapse:collapse;margin-top:6px;">
        <tr style="background:#ffebee;">
          <td colspan="2" style="padding:3px 6px;font-weight:700;color:#c62828;">Non-Food Crops</td>
          <td style="padding:3px 6px;text-align:right;font-weight:700;color:#c62828;">{fmt_pct(nonfood_area)}</td>
        </tr>
        <tr><td style="padding:2px 6px;color:#666;">Fiber (Cotton/Jute)</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_fiber_crops"), "#ef5350")}</td></tr>
        <tr style="background:#fafafa;"><td style="padding:2px 6px;color:#666;">Drugs/Narcotics</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_drugs_and_narcotics"), "#ff7043")}</td></tr>
        <tr><td style="padding:2px 6px;color:#666;">Fodder</td><td colspan="2" style="padding:2px 6px;">{pct_bar(row.get("share_fodder"), "#ffa726")}</td></tr>
      </table>

      <div style="margin-top:8px;padding-top:6px;border-top:1px solid #ddd;font-size:11px;">
        <div style="font-weight:600;color:#1a1a2e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          Dominant: {dom_crop} ({dom_share})
        </div>
      </div>
    </div>
    """

# Build GeoJSON features
print("Preparing GeoJSON features...")
features = []

# Columns to include in properties for coloring
layer_keys = list(INDEX_CONFIG.keys())

for idx, row in gdf.iterrows():
    geom = row["geometry"]
    if geom is None or geom.is_empty:
        continue
    props = {
        "tooltip": build_tooltip_html(row),
        "district_name": str(row.get("district_name") or row.get("dtname") or ""),
        "state_name": str(row.get("state_name") or row.get("stname") or ""),
    }

    for key in layer_keys:
        val = row.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                props[key] = float(val)
            except (ValueError, TypeError):
                props[key] = None
        else:
            props[key] = None

    feat = {
        "type": "Feature",
        "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
        "properties": props,
    }
    features.append(feat)

geojson_data = {"type": "FeatureCollection", "features": features}
print(f"  {len(features)} features prepared")

# Compute color scales
color_scales = {}
for idx_key in layer_keys:
    vals = [f["properties"][idx_key] for f in features]
    cfg = INDEX_CONFIG[idx_key]
    edges, colors = get_color_scale(vals, cfg["cmap"])
    color_scales[idx_key] = (edges, colors)

# ---------------------------------------------------------------------------
# 5. Create Folium map
# ---------------------------------------------------------------------------
m = folium.Map(
    location=[22.5, 82.0],
    zoom_start=5,
    tiles=None,
    prefer_canvas=True,
)

folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
    attr="CartoDB",
    name="Light",
    control=False,
).add_to(m)

geojson_js = json.dumps(geojson_data)

color_scales_dict = {}
for k, v in color_scales.items():
    color_scales_dict[k] = {"edges": [float(x) for x in v[0]], "colors": v[1]}
color_scales_js = json.dumps(color_scales_dict)

custom_html = f"""
<style>
  .control-panel {{
    position: fixed;
    top: 12px;
    left: 60px;
    z-index: 1000;
    background: rgba(255,255,255,0.95);
    padding: 12px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
  }}
  .control-panel label {{
    font-weight: 600;
    color: #1a1a2e;
    display: block;
    margin-bottom: 4px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .control-panel select {{
    width: 260px;
    padding: 4px 8px;
    margin-bottom: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 13px;
    background: white;
  }}
  .map-title {{
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    background: rgba(255,255,255,0.92);
    padding: 8px 24px;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #1a1a2e;
    white-space: nowrap;
  }}
  .legend-panel {{
    position: fixed;
    bottom: 30px;
    right: 20px;
    z-index: 1000;
    background: rgba(255,255,255,0.95);
    padding: 10px 14px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11px;
    max-height: 300px;
    overflow-y: auto;
  }}
  .legend-panel .legend-title {{
    font-weight: 700;
    margin-bottom: 6px;
    color: #1a1a2e;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    margin: 2px 0;
  }}
  .legend-swatch {{
    width: 20px;
    height: 14px;
    margin-right: 6px;
    border: 1px solid #ccc;
    flex-shrink: 0;
  }}
  .district-tooltip {{
    background: white;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 0;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
  }}
  .district-tooltip .leaflet-tooltip-content {{
    margin: 0;
    padding: 8px;
  }}
</style>

<div class="map-title" id="mapTitle">Food vs Non-Food: Food Crop Area Share</div>

<div class="control-panel">
  <label>Display Layer</label>
  <select id="indexSelect" onchange="updateMap()">
    <option value="food_crop_area_share">Food Crop Area Share</option>
    <option value="share_cereals">Cereal Share</option>
    <option value="share_pulses">Pulse Share</option>
    <option value="share_oilseeds">Oilseed Share</option>
    <option value="cash_crop_share">Cash Crop Share (Fiber + Drugs)</option>
    <option value="share_sugar">Sugar Share</option>
    <option value="fruit_veg_share">Fruit &amp; Vegetable Share</option>
  </select>
</div>

<div class="legend-panel" id="legendPanel"></div>

<script>
var geojsonData = {geojson_js};
var colorScales = {color_scales_js};

var indexLabels = {{
  "food_crop_area_share": "Food Crop Area Share",
  "share_cereals": "Cereal Share",
  "share_pulses": "Pulse Share",
  "share_oilseeds": "Oilseed Share",
  "cash_crop_share": "Cash Crop Share (Fiber + Drugs)",
  "share_sugar": "Sugar Share",
  "fruit_veg_share": "Fruit & Vegetable Share"
}};

var currentLayer = null;

function valToColor(val, edges, colors) {{
  if (val === null || val === undefined) return "#d3d3d3";
  for (var i = 0; i < edges.length - 1; i++) {{
    if (val <= edges[i+1]) return colors[i];
  }}
  return colors[colors.length - 1];
}}

function updateLegend(indexKey) {{
  var html = '<div class="legend-title">' + indexLabels[indexKey] + '</div>';
  var cs = colorScales[indexKey];
  if (!cs || !cs.edges || cs.edges.length === 0) return;
  for (var i = 0; i < cs.colors.length; i++) {{
    var lo = (cs.edges[i] * 100).toFixed(1) + "%";
    var hi = (cs.edges[i+1] * 100).toFixed(1) + "%";
    html += '<div class="legend-item"><div class="legend-swatch" style="background:' + cs.colors[i] + '"></div>' + lo + ' - ' + hi + '</div>';
  }}
  html += '<div class="legend-item"><div class="legend-swatch" style="background:#d3d3d3"></div>No data</div>';
  document.getElementById("legendPanel").innerHTML = html;
}}

function updateMap() {{
  var indexKey = document.getElementById("indexSelect").value;
  document.getElementById("mapTitle").innerHTML = "Food vs Non-Food: " + indexLabels[indexKey];

  var cs = colorScales[indexKey];

  if (currentLayer) {{
    window._map.removeLayer(currentLayer);
  }}

  currentLayer = L.geoJson(geojsonData, {{
    style: function(feature) {{
      var val = feature.properties[indexKey];
      var fillColor = valToColor(val, cs.edges, cs.colors);
      return {{
        fillColor: fillColor,
        weight: 0.5,
        color: "#888",
        fillOpacity: 0.8
      }};
    }},
    onEachFeature: function(feature, layer) {{
      layer.bindTooltip(feature.properties.tooltip, {{
        sticky: true,
        direction: "auto",
        className: "district-tooltip"
      }});
      layer.on({{
        mouseover: function(e) {{
          e.target.setStyle({{weight: 2, color: "#333"}});
          e.target.bringToFront();
        }},
        mouseout: function(e) {{
          currentLayer.resetStyle(e.target);
        }}
      }});
    }}
  }}).addTo(window._map);

  updateLegend(indexKey);
}}

setTimeout(function() {{
  document.querySelectorAll('[class*="folium-map"]').forEach(function(el) {{
    var mapId = el.id;
    if (mapId && window[mapId]) {{
      window._map = window[mapId];
    }}
  }});
  if (!window._map) {{
    for (var key in window) {{
      try {{
        if (window[key] instanceof L.Map) {{
          window._map = window[key];
          break;
        }}
      }} catch(e) {{}}
    }}
  }}
  if (window._map) {{
    updateMap();
  }} else {{
    console.error("Could not find Leaflet map instance");
  }}
}}, 500);
</script>
"""

m.get_root().html.add_child(folium.Element(custom_html))

# ---------------------------------------------------------------------------
# 6. Save and copy
# ---------------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUT_PATH))
print(f"\nMap saved to: {OUT_PATH}")

DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(OUT_PATH), str(DOCS_PATH))
print(f"Copied to: {DOCS_PATH}")

print("Opening in browser...")
webbrowser.open(str(OUT_PATH))
