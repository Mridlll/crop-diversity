"""
Crop Diversity Interactive Hover Map
=====================================
Generates a full-screen Folium map of India colored by crop diversity indices.
Hover over any district to see detailed crop diversity information.

Usage:
    python scripts/60_crop_diversity_hover_map.py

Output:
    outputs/crop_diversity_analysis/crop_diversity_hover_map.html
"""

import json
import re
import webbrowser
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import folium
from thefuzz import fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "outputs" / "crop_diversity_analysis" / "district_diversity_indices.csv"
SHP_PATH = BASE / "Package_Maps_Share_20251120_FINAL" / "shapefiles" / "in_district.shp"
OUT_PATH = BASE / "outputs" / "crop_diversity_analysis" / "crop_diversity_hover_map.html"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH)
gdf = gpd.read_file(SHP_PATH)

print(f"  CSV districts: {len(df)}")
print(f"  Shapefile districts: {len(gdf)}")

# ---------------------------------------------------------------------------
# 2. Manual mapping + Fuzzy matching
# ---------------------------------------------------------------------------
def normalize(s):
    """Normalize district/state names for matching."""
    s = str(s).upper().strip()
    s = s.replace("&", "AND")
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---- Manual mapping: (shp_state, shp_district) -> (csv_state, csv_district) ----
# Maps shapefile names to the CSV district_key components they should match.
# Covers: state name changes, spelling differences, renamed districts,
#         state reorganization (Telangana/AP, Uttarakhand/UP, Chhattisgarh/MP,
#         Jharkhand/Bihar), and UT mergers.
MANUAL_MAP = {
    # --- State name differences (ANDAMAN AND NICOBAR vs ANDAMAN AND NICOBAR ISLANDS) ---
    ("ANDAMAN AND NICOBAR", "NICOBARS"): ("ANDAMAN AND NICOBAR ISLANDS", "NICOBARS"),
    ("ANDAMAN AND NICOBAR", "NORTH AND MIDDLE ANDAMAN"): ("ANDAMAN AND NICOBAR ISLANDS", "NORTH AND MIDDLE ANDAMAN"),
    ("ANDAMAN AND NICOBAR", "SOUTH ANDAMAN"): ("ANDAMAN AND NICOBAR ISLANDS", "SOUTH ANDAMANS"),
    # --- DADRA AND NAGAR HAVELI / DAMAN AND DIU (UT merger) ---
    ("DADRA AND NAGAR HAVE", "DADRA AND NAGAR HAVELI"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DADRA AND NAGAR HAVELI"),
    ("DAMAN AND DIU", "DAMAN"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DAMAN"),
    ("DAMAN AND DIU", "DIU"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DIU"),
    # --- Andhra Pradesh spelling ---
    ("ANDHRA PRADESH", "SRI POTTI SRIRAMULU NELL"): ("ANDHRA PRADESH", "SRI POTTI SRIRAMULU NELLORE"),
    ("ANDHRA PRADESH", "VISAKHAPATNAM"): ("ANDHRA PRADESH", "VISAKHAPATANAM"),
    ("ANDHRA PRADESH", "YSR"): ("ANDHRA PRADESH", "KADAPA"),
    # --- Arunachal Pradesh ---
    ("ARUNACHAL PRADESH", "LEPA RADA"): ("ARUNACHAL PRADESH", "LEPARADA"),
    # --- Assam ---
    ("ASSAM", "KAMRUP METROPOLITAN"): ("ASSAM", "KAMRUP METRO"),
    ("ASSAM", "MORIGAON"): ("ASSAM", "MARIGAON"),
    # --- Bihar ---
    ("BIHAR", "PURBA CHAMPARAN"): ("BIHAR", "PURBI CHAMPARAN"),
    # --- Chhattisgarh spelling ---
    ("CHHATTISGARH", "BAMETARA"): ("CHHATTISGARH", "BEMETARA"),
    ("CHHATTISGARH", "GARIABAND"): ("CHHATTISGARH", "GARIYABAND"),
    ("CHHATTISGARH", "JANJGIR CHAMPA"): ("CHHATTISGARH", "JANJGIRCHAMPA"),
    ("CHHATTISGARH", "KABEERDHAM"): ("CHHATTISGARH", "KABIRDHAM"),
    ("CHHATTISGARH", "KORIYA"): ("CHHATTISGARH", "KOREA"),
    ("CHHATTISGARH", "UTTAR BASTAR KANKER"): ("CHHATTISGARH", "KANKER"),
    # --- Gujarat ---
    ("GUJARAT", "CHOTA UDAIPUR"): ("GUJARAT", "CHHOTAUDEPUR"),
    ("GUJARAT", "THE DANGS"): ("GUJARAT", "DANG"),
    # --- Haryana ---
    ("HARYANA", "GURUGRAM"): ("HARYANA", "GURGAON"),
    ("HARYANA", "NUH"): ("HARYANA", "MEWAT"),
    # --- Jammu and Kashmir ---
    ("JAMMU AND KASHMIR", "BANDIPORE"): ("JAMMU AND KASHMIR", "BANDIPORA"),
    ("JAMMU AND KASHMIR", "BARAMULA"): ("JAMMU AND KASHMIR", "BARAMULLA"),
    ("JAMMU AND KASHMIR", "PUNCH"): ("JAMMU AND KASHMIR", "POONCH"),
    ("JAMMU AND KASHMIR", "RAJOURI"): ("JAMMU AND KASHMIR", "RAJAURI"),
    ("JAMMU AND KASHMIR", "SHUPIYAN"): ("JAMMU AND KASHMIR", "SHOPIAN"),
    # --- Jharkhand ---
    ("JHARKHAND", "KODARMA"): ("JHARKHAND", "KODERMA"),
    ("JHARKHAND", "PASHCHIMI SINGHBHUM"): ("JHARKHAND", "WEST SINGHBHUM"),
    ("JHARKHAND", "PURBI SINGHBHUM"): ("JHARKHAND", "EAST SINGHBUM"),
    ("JHARKHAND", "SAHIBGANJ"): ("JHARKHAND", "SAHEBGANJ"),
    ("JHARKHAND", "SARAIKELAKHARSAWAN"): ("JHARKHAND", "SARAIKELA KHARSAWAN"),
    # --- Karnataka ---
    ("KARNATAKA", "BANGALORE"): ("KARNATAKA", "BANGALORE RURAL"),
    ("KARNATAKA", "BENGALURU RURAL"): ("KARNATAKA", "BANGALORE RURAL"),
    ("KARNATAKA", "DAVANAGERE"): ("KARNATAKA", "DAVANGERE"),
    ("KARNATAKA", "YADGIR"): ("KARNATAKA", "YADAGIRI"),
    # --- Ladakh ---
    ("LADAKH", "LEH"): ("LADAKH", "LEH LADAKH"),
    # --- Madhya Pradesh ---
    ("MADHYA PRADESH", "EAST NIMAR"): ("MADHYA PRADESH", "KHANDWA"),
    ("MADHYA PRADESH", "WEST NIMAR"): ("MADHYA PRADESH", "KHARGONE"),
    ("MADHYA PRADESH", "NARSIMHAPUR"): ("MADHYA PRADESH", "NARSINGHPUR"),
    # --- Maharashtra ---
    ("MAHARASHTRA", "AHMADNAGAR"): ("MAHARASHTRA", "AHMEDNAGAR"),
    ("MAHARASHTRA", "BID"): ("MAHARASHTRA", "BEED"),
    ("MAHARASHTRA", "BULDANA"): ("MAHARASHTRA", "BULDHANA"),
    ("MAHARASHTRA", "GONDIYA"): ("MAHARASHTRA", "GONDIA"),
    ("MAHARASHTRA", "RAIGARH"): ("MAHARASHTRA", "RAIGAD"),
    # --- Meghalaya ---
    ("MEGHALAYA", "RIBHOI"): ("MEGHALAYA", "RI BHOI"),
    # --- Odisha ---
    ("ODISHA", "BAUDH"): ("ODISHA", "BOUDH"),
    ("ODISHA", "DEBAGARH"): ("ODISHA", "DEOGARH"),
    ("ODISHA", "NABARANGAPUR"): ("ODISHA", "NABARANGPUR"),
    ("ODISHA", "SUBARNAPUR"): ("ODISHA", "SONEPUR"),
    # --- Puducherry ---
    ("PUDUCHERRY", "PUDUCHERRY"): ("PUDUCHERRY", "PONDICHERRY"),
    # --- Punjab ---
    ("PUNJAB", "FIROZPUR"): ("PUNJAB", "FIROZEPUR"),
    ("PUNJAB", "SAHIBZADA AJIT SINGH NAG"): ("PUNJAB", "SAS NAGAR"),
    ("PUNJAB", "SRI MUKTSAR SAHIB"): ("PUNJAB", "MUKTSAR"),
    # --- Rajasthan ---
    ("RAJASTHAN", "CHITTAURGARH"): ("RAJASTHAN", "CHITTORGARH"),
    ("RAJASTHAN", "DHAULPUR"): ("RAJASTHAN", "DHOLPUR"),
    ("RAJASTHAN", "JALOR"): ("RAJASTHAN", "JALORE"),
    ("RAJASTHAN", "JHUNJHUNUN"): ("RAJASTHAN", "JHUNJHUNU"),
    # --- Sikkim (renamed districts) ---
    ("SIKKIM", "EAST DISTRICT"): ("SIKKIM", "GANGTOK"),
    ("SIKKIM", "NORTH DISTRICT"): ("SIKKIM", "MANGAN"),
    ("SIKKIM", "SOUTH DISTRICT"): ("SIKKIM", "NAMCHI"),
    ("SIKKIM", "WEST DISTRICT"): ("SIKKIM", "GYALSHING"),
    # --- Tamil Nadu ---
    ("TAMIL NADU", "KANCHEEPURAM"): ("TAMIL NADU", "KANCHIPURAM"),
    ("TAMIL NADU", "THOOTHUKKUDI"): ("TAMIL NADU", "TUTICORIN"),
    ("TAMIL NADU", "VILUPPURAM"): ("TAMIL NADU", "VILLUPURAM"),
    # --- Telangana ---
    ("TELANGANA", "BHADRADRI KOTHAGUDEM"): ("TELANGANA", "BHADRADRI"),
    ("TELANGANA", "JOGULAMBA GADWAL"): ("TELANGANA", "JOGULAMBA"),
    ("TELANGANA", "KUMURAM BHEEM ASIFABAD"): ("TELANGANA", "KOMARAM BHEEM ASIFABAD"),
    ("TELANGANA", "NARAYANPET"): ("TELANGANA", "NARAYANAPET"),
    ("TELANGANA", "RAJANNA SIRCILLA"): ("TELANGANA", "RAJANNA"),
    ("TELANGANA", "RANGA REDDY"): ("TELANGANA", "RANGAREDDI"),
    ("TELANGANA", "WARANGAL RURAL"): ("TELANGANA", "WARANGAL"),
    ("TELANGANA", "WARANGAL URBAN"): ("TELANGANA", "HANUMAKONDA"),
    ("TELANGANA", "YADADRI BHUVANAGIRI"): ("TELANGANA", "YADADRI"),
    # --- Tripura ---
    ("TRIPURA", "SIPAHIJALA"): ("TRIPURA", "SEPAHIJALA"),
    ("TRIPURA", "UNOKOTI"): ("TRIPURA", "UNAKOTI"),
    # --- Uttar Pradesh ---
    ("UTTAR PRADESH", "BARA BANKI"): ("UTTAR PRADESH", "BARABANKI"),
    ("UTTAR PRADESH", "BHADOHI"): ("UTTAR PRADESH", "SANT RAVIDAS NAGAR"),
    ("UTTAR PRADESH", "KUSHINAGAR"): ("UTTAR PRADESH", "KUSHI NAGAR"),
    ("UTTAR PRADESH", "MAHRAJGANJ"): ("UTTAR PRADESH", "MAHARAJGANJ"),
    ("UTTAR PRADESH", "PRAYAGRAJ"): ("UTTAR PRADESH", "ALLAHABAD"),
    ("UTTAR PRADESH", "SANT KABIR NAGAR"): ("UTTAR PRADESH", "SANT KABEER NAGAR"),
    ("UTTAR PRADESH", "SHRAWASTI"): ("UTTAR PRADESH", "SHRAVASTI"),
    ("UTTAR PRADESH", "SIDDHARTHNAGAR"): ("UTTAR PRADESH", "SIDDHARTH NAGAR"),
    # --- Uttarakhand ---
    ("UTTARAKHAND", "GARHWAL"): ("UTTARAKHAND", "PAURI GARHWAL"),
    ("UTTARAKHAND", "HARDWAR"): ("UTTARAKHAND", "HARIDWAR"),
    ("UTTARAKHAND", "RUDRAPRAYAG"): ("UTTARAKHAND", "RUDRA PRAYAG"),
    ("UTTARAKHAND", "UDHAM SINGH NAGAR"): ("UTTARAKHAND", "UDAM SINGH NAGAR"),
    ("UTTARAKHAND", "UTTARKASHI"): ("UTTARAKHAND", "UTTAR KASHI"),
    # --- West Bengal ---
    ("WEST BENGAL", "COOCH BEHAR"): ("WEST BENGAL", "COOCHBEHAR"),
    ("WEST BENGAL", "DARJILING"): ("WEST BENGAL", "DARJEELING"),
    ("WEST BENGAL", "MEDINIPUR WEST"): ("WEST BENGAL", "PASHCHIM MEDINIPUR"),
    ("WEST BENGAL", "NORTH TWENTY FOUR PARGAN"): ("WEST BENGAL", "NORTH 24 PARGANAS"),
    ("WEST BENGAL", "PURULIYA"): ("WEST BENGAL", "PURULIA"),
    ("WEST BENGAL", "SOUTH TWENTY FOUR PARGAN"): ("WEST BENGAL", "SOUTH 24 PARGANAS"),
}

# Build lookup from CSV: (state_norm, district_norm) -> row index
csv_keys = {}
for i, row in df.iterrows():
    st = normalize(row["state_name"])
    dt = normalize(row["district_name"])
    csv_keys[(st, dt)] = i

# Try matching each shapefile district
match_col = [None] * len(gdf)
exact_matches = 0
manual_matches = 0
fuzzy_matches = 0
unmatched = 0

for idx, row in gdf.iterrows():
    shp_st = normalize(row["stname"])
    shp_dt = normalize(row["dtname"])

    # 1) Exact match
    if (shp_st, shp_dt) in csv_keys:
        match_col[idx] = csv_keys[(shp_st, shp_dt)]
        exact_matches += 1
        continue

    # 2) Manual mapping
    mapped = MANUAL_MAP.get((shp_st, shp_dt))
    if mapped and mapped in csv_keys:
        match_col[idx] = csv_keys[mapped]
        manual_matches += 1
        continue

    # 3) District-only exact match (state names may differ)
    district_only = [(k, v) for k, v in csv_keys.items() if k[1] == shp_dt]
    if len(district_only) == 1:
        match_col[idx] = district_only[0][1]
        exact_matches += 1
        continue

    # 4) Fuzzy match within same state first, then across all
    best_score = 0
    best_idx = None

    # Same-state fuzzy
    same_state = [(k, v) for k, v in csv_keys.items()
                  if fuzz.ratio(k[0], shp_st) > 80]
    for (csv_st, csv_dt), csv_idx in same_state:
        score = fuzz.ratio(shp_dt, csv_dt)
        if score > best_score:
            best_score = score
            best_idx = csv_idx

    # If no good same-state match, try cross-state
    if best_score < 85:
        for (csv_st, csv_dt), csv_idx in csv_keys.items():
            score = fuzz.ratio(shp_dt, csv_dt)
            if score > best_score:
                best_score = score
                best_idx = csv_idx

    if best_score >= 85:
        match_col[idx] = best_idx
        fuzzy_matches += 1
    else:
        unmatched += 1

gdf["csv_idx"] = match_col
total_matched = exact_matches + manual_matches + fuzzy_matches
coverage = total_matched / len(gdf) * 100
print(f"\nMatch statistics:")
print(f"  Exact matches:  {exact_matches}")
print(f"  Manual matches: {manual_matches}")
print(f"  Fuzzy matches:  {fuzzy_matches}")
print(f"  Unmatched:      {unmatched}")
print(f"  Coverage:       {total_matched}/{len(gdf)} ({coverage:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Merge data into GeoDataFrame
# ---------------------------------------------------------------------------
# Create columns for all needed data
merge_cols = [
    "state_name", "district_name", "shannon_index", "simpson_index",
    "crop_richness", "agro_biodiversity_index", "irrigation_regime",
    "dominant_crop", "dominant_crop_share",
    "share_cereals", "share_pulses", "share_oilseeds",
    "share_fruits", "share_spices",
]
# Cash crops = fiber + sugar + drugs_and_narcotics
df["share_cash_crops"] = (
    df["share_fiber_crops"].fillna(0)
    + df["share_sugar"].fillna(0)
    + df["share_drugs_and_narcotics"].fillna(0)
)
merge_cols.append("share_cash_crops")

for col in merge_cols:
    gdf[col] = None

for idx, row in gdf.iterrows():
    csv_idx = row["csv_idx"]
    if csv_idx is not None and not pd.isna(csv_idx):
        csv_idx = int(csv_idx)
        for col in merge_cols:
            gdf.at[idx, col] = df.at[csv_idx, col]

# Fill missing irrigation regime
gdf["irrigation_regime"] = gdf["irrigation_regime"].fillna("Unknown")

# Simplify geometries for speed
print("\nSimplifying geometries...")
gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)

# Ensure WGS84
gdf = gdf.to_crs(epsg=4326)

# ---------------------------------------------------------------------------
# 4. Build Folium map
# ---------------------------------------------------------------------------
print("Building map...")

# Index configs
INDEX_CONFIG = {
    "shannon_index":            {"label": "Shannon Index",      "cmap": "YlGnBu", "fmt": ".2f"},
    "simpson_index":            {"label": "Simpson Index",      "cmap": "YlGnBu", "fmt": ".2f"},
    "crop_richness":            {"label": "Crop Richness",      "cmap": "YlGnBu", "fmt": ".0f"},
    "agro_biodiversity_index":  {"label": "Agro-Biodiversity Index", "cmap": "YlGnBu", "fmt": ".2f"},
}

def get_color_scale(values, n_bins=8):
    """Return bin edges and colors for a YlGnBu-style palette."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return [], []
    vmin, vmax = min(valid), max(valid)
    if vmin == vmax:
        vmax = vmin + 1
    edges = np.linspace(vmin, vmax, n_bins + 1)
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("YlGnBu", n_bins)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_bins)]
    return edges, colors

def value_to_color(val, edges, colors):
    """Map a value to a hex color."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#d3d3d3"  # light gray for missing
    for i in range(len(edges) - 1):
        if val <= edges[i + 1]:
            return colors[i]
    return colors[-1]

def fmt_pct(val):
    """Format a share value as percentage."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{float(val) * 100:.1f}%"

def fmt_idx(val, fmt=".2f"):
    """Format an index value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{float(val):{fmt}}"

# Pre-compute tooltip HTML for each feature
def build_tooltip_html(row):
    """Build rich HTML tooltip for a district."""
    name = row.get("district_name") or row.get("dtname") or "Unknown"
    state = row.get("state_name") or row.get("stname") or "Unknown"
    irr = row.get("irrigation_regime") or "Unknown"
    dom_crop = row.get("dominant_crop") or "N/A"
    dom_share = fmt_pct(row.get("dominant_crop_share"))

    shannon = fmt_idx(row.get("shannon_index"))
    simpson = fmt_idx(row.get("simpson_index"))
    richness = fmt_idx(row.get("crop_richness"), ".0f")
    abi = fmt_idx(row.get("agro_biodiversity_index"))

    cereals = fmt_pct(row.get("share_cereals"))
    pulses = fmt_pct(row.get("share_pulses"))
    oilseeds = fmt_pct(row.get("share_oilseeds"))
    cash = fmt_pct(row.get("share_cash_crops"))
    fruits = fmt_pct(row.get("share_fruits"))
    spices = fmt_pct(row.get("share_spices"))

    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif; font-size:13px; line-height:1.5; min-width:240px;">
      <div style="font-size:15px; font-weight:700; color:#1a1a2e; border-bottom:2px solid #0f3460; padding-bottom:4px; margin-bottom:6px;">
        {name}
      </div>
      <div style="color:#555; font-size:12px; margin-bottom:8px;">{state} &bull; {irr}</div>
      <table style="width:100%; font-size:12px; border-collapse:collapse;">
        <tr style="background:#f0f4f8;"><td style="padding:2px 6px;">Shannon</td><td style="padding:2px 6px; text-align:right; font-weight:600;">{shannon}</td></tr>
        <tr><td style="padding:2px 6px;">Simpson</td><td style="padding:2px 6px; text-align:right; font-weight:600;">{simpson}</td></tr>
        <tr style="background:#f0f4f8;"><td style="padding:2px 6px;">Richness</td><td style="padding:2px 6px; text-align:right; font-weight:600;">{richness}</td></tr>
        <tr><td style="padding:2px 6px;">ABI</td><td style="padding:2px 6px; text-align:right; font-weight:600;">{abi}</td></tr>
      </table>
      <div style="margin-top:8px; padding-top:6px; border-top:1px solid #ddd; font-size:12px;">
        <div style="font-weight:600; color:#1a1a2e;">Dominant: {dom_crop} ({dom_share})</div>
      </div>
      <table style="width:100%; font-size:11px; margin-top:4px; border-collapse:collapse;">
        <tr style="background:#f0f4f8;"><td style="padding:1px 6px;">Cereals</td><td style="text-align:right; padding:1px 6px;">{cereals}</td>
            <td style="padding:1px 6px;">Pulses</td><td style="text-align:right; padding:1px 6px;">{pulses}</td></tr>
        <tr><td style="padding:1px 6px;">Oilseeds</td><td style="text-align:right; padding:1px 6px;">{oilseeds}</td>
            <td style="padding:1px 6px;">Cash Crops</td><td style="text-align:right; padding:1px 6px;">{cash}</td></tr>
        <tr style="background:#f0f4f8;"><td style="padding:1px 6px;">Fruits</td><td style="text-align:right; padding:1px 6px;">{fruits}</td>
            <td style="padding:1px 6px;">Spices</td><td style="text-align:right; padding:1px 6px;">{spices}</td></tr>
      </table>
    </div>
    """

# Convert GeoDataFrame to GeoJSON with properties
print("Preparing GeoJSON features...")
features = []
for idx, row in gdf.iterrows():
    geom = row["geometry"]
    if geom is None or geom.is_empty:
        continue
    props = {
        "tooltip": build_tooltip_html(row),
        "district_name": str(row.get("district_name") or row.get("dtname") or ""),
        "state_name": str(row.get("state_name") or row.get("stname") or ""),
        "irrigation_regime": str(row.get("irrigation_regime") or "Unknown"),
    }
    # Add index values (as float or None)
    for idx_key in INDEX_CONFIG:
        val = row.get(idx_key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            props[idx_key] = float(val)
        else:
            props[idx_key] = None

    feat = {
        "type": "Feature",
        "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
        "properties": props,
    }
    features.append(feat)

geojson_data = {"type": "FeatureCollection", "features": features}
print(f"  {len(features)} features prepared")

# Compute color scales for each index
color_scales = {}
for idx_key in INDEX_CONFIG:
    vals = [f["properties"][idx_key] for f in features]
    edges, colors = get_color_scale(vals)
    color_scales[idx_key] = (edges, colors)
    # Add color to each feature
    for f in features:
        val = f["properties"][idx_key]
        f["properties"][f"color_{idx_key}"] = value_to_color(val, edges, colors)

# ---------------------------------------------------------------------------
# 5. Create Folium map with custom JS for index/filter switching
# ---------------------------------------------------------------------------
m = folium.Map(
    location=[22.5, 82.0],
    zoom_start=5,
    tiles=None,
    prefer_canvas=True,
)

# Minimal light tile layer (no labels clutter)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
    attr="CartoDB",
    name="Light",
    control=False,
).add_to(m)

# Write GeoJSON data as a JS variable
geojson_js = json.dumps(geojson_data)

# Pre-compute color scales JSON (outside f-string to avoid brace conflicts)
color_scales_dict = {}
for k, v in color_scales.items():
    color_scales_dict[k] = {"edges": [float(x) for x in v[0]], "colors": v[1]}
color_scales_js = json.dumps(color_scales_dict)

# Build the full custom HTML/JS
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
    width: 200px;
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
  }}
</style>

<div class="map-title">Crop Diversity Across Indian Districts (1997&ndash;2021)</div>

<div class="control-panel">
  <label>Index</label>
  <select id="indexSelect" onchange="updateMap()">
    <option value="agro_biodiversity_index">Agro-Biodiversity Index</option>
    <option value="shannon_index">Shannon Index</option>
    <option value="simpson_index">Simpson Index</option>
    <option value="crop_richness">Crop Richness</option>
  </select>
  <label>Irrigation</label>
  <select id="irrSelect" onchange="updateMap()">
    <option value="All">All Districts</option>
    <option value="Rainfed">Rainfed</option>
    <option value="Semi-Irrigated">Semi-Irrigated</option>
    <option value="Irrigated">Irrigated</option>
  </select>
</div>

<div class="legend-panel" id="legendPanel"></div>

<script>
var geojsonData = {geojson_js};

var colorScales = {color_scales_js};

var indexLabels = {{
  "shannon_index": "Shannon Index",
  "simpson_index": "Simpson Index",
  "crop_richness": "Crop Richness",
  "agro_biodiversity_index": "Agro-Biodiversity Index"
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
  var cs = colorScales[indexKey];
  var html = '<div class="legend-title">' + indexLabels[indexKey] + '</div>';
  for (var i = 0; i < cs.colors.length; i++) {{
    var lo = cs.edges[i].toFixed(2);
    var hi = cs.edges[i+1].toFixed(2);
    if (indexKey === "crop_richness") {{
      lo = Math.round(cs.edges[i]);
      hi = Math.round(cs.edges[i+1]);
    }}
    html += '<div class="legend-item"><div class="legend-swatch" style="background:' + cs.colors[i] + '"></div>' + lo + ' &ndash; ' + hi + '</div>';
  }}
  html += '<div class="legend-item"><div class="legend-swatch" style="background:#d3d3d3"></div>No data</div>';
  document.getElementById("legendPanel").innerHTML = html;
}}

function updateMap() {{
  var indexKey = document.getElementById("indexSelect").value;
  var irrFilter = document.getElementById("irrSelect").value;
  var cs = colorScales[indexKey];

  // Filter features
  var filtered = {{type: "FeatureCollection", features: []}};
  for (var i = 0; i < geojsonData.features.length; i++) {{
    var f = geojsonData.features[i];
    var irr = f.properties.irrigation_regime || "Unknown";
    if (irrFilter === "All" || irr.indexOf(irrFilter) !== -1) {{
      filtered.features.push(f);
    }}
  }}

  if (currentLayer) {{
    window._map.removeLayer(currentLayer);
  }}

  currentLayer = L.geoJson(filtered, {{
    style: function(feature) {{
      var val = feature.properties[indexKey];
      return {{
        fillColor: valToColor(val, cs.edges, cs.colors),
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

// Initialize after map loads
setTimeout(function() {{
  // Get map instance - Folium stores it
  var maps = Object.keys(window).filter(function(k) {{ return window[k] instanceof L.Map; }});
  if (maps.length === 0) {{
    // Fallback: find map container
    var containers = document.querySelectorAll('.folium-map');
    containers.forEach(function(c) {{
      if (c._leaflet_id) {{
        window._map = c._leaflet_map || L.map(c);
      }}
    }});
  }}
  // Try another approach - iterate over leaflet map instances
  document.querySelectorAll('[class*="folium-map"]').forEach(function(el) {{
    var mapId = el.id;
    if (mapId && window[mapId]) {{
      window._map = window[mapId];
    }}
  }});
  if (!window._map) {{
    // Last resort: find any L.Map in window
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

<style>
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
"""

# Add the custom HTML to the map
m.get_root().html.add_child(folium.Element(custom_html))

# ---------------------------------------------------------------------------
# 6. Save and open
# ---------------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUT_PATH))
print(f"\nMap saved to: {OUT_PATH}")
print(f"Opening in browser...")
webbrowser.open(str(OUT_PATH))
