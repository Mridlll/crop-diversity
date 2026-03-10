"""
District-Level Calorie Production & Diversity-Calorie Overlay Analysis
======================================================================
Computes district-level kcal production from crop production data,
merges with diversity indices, classifies districts into quadrants,
and generates static maps and scatter plots.

Usage:
    python scripts/62_district_calorie_production.py

Outputs:
    outputs/crop_diversity_analysis/district_diversity_calorie_merged.csv
    outputs/crop_diversity_analysis/kcal_per_hectare_choropleth.png
    outputs/crop_diversity_analysis/quadrant_map.png
    outputs/crop_diversity_analysis/bivariate_abi_kcal_map.png
    outputs/crop_diversity_analysis/nutritionally_hollow_map.png
    outputs/crop_diversity_analysis/abi_vs_kcal_scatter.png
"""

import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from thefuzz import fuzz

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
APY_PATH = BASE / "outputs" / "all_crops_apy_1997_2021_india_data_portal.csv"
KCAL_XLSX = BASE / "crop_summary_with_kcal.xlsx"
DIV_PATH = BASE / "outputs" / "crop_diversity_analysis" / "district_diversity_indices.csv"
SHP_PATH = BASE / "Package_Maps_Share_20251120_FINAL" / "shapefiles" / "in_district.shp"
OUT_DIR = BASE / "outputs" / "crop_diversity_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Build comprehensive kcal conversion table
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Building kcal conversion table")
print("=" * 70)

# Kcal per 100g values from IFCT 2017 / USDA / standard references
# For oilseeds: using kcal of the seed (not extracted oil)
KCAL_PER_100G = {
    # Cereals
    "Rice": 345,
    "Wheat": 341,
    "Jowar": 349,
    "Bajra": 361,
    "Maize": 342,
    "Ragi": 328,
    "Barley": 336,
    "Small Millets": 341,       # avg of minor millets (kodo, foxtail, etc.)
    "Other Cereals": 340,       # approximate average of cereals

    # Pulses
    "Arhar/Tur": 343,
    "Gram": 360,
    "Moong(Green Gram)": 347,
    "Urad": 341,
    "Masoor": 343,
    "Khesari": 345,
    "Moth": 330,
    "Horse-Gram": 321,
    "Cowpea(Lobia)": 323,
    "Peas & Beans (Pulses)": 315,
    "Other Kharif Pulses": 335,  # approximate
    "Other  Rabi Pulses": 335,   # approximate
    "Other Summer Pulses": 335,  # approximate

    # Oilseeds (whole seed kcal)
    "Groundnut": 567,
    "Soyabean": 432,
    "Rapeseed &Mustard": 481,
    "Sunflower": 584,
    "Sesamum": 573,
    "Linseed": 534,
    "Castor Seed": 0,            # toxic, not food
    "Safflower": 517,
    "Niger Seed": 470,
    "Coconut": 354,              # fresh coconut meat
    "Other Oilseeds": 450,       # approximate

    # Sugar
    "Sugarcane": 40,             # kcal per 100g of raw cane (juice ~10-15% sugar)

    # Vegetables
    "Potato": 77,
    "Onion": 40,
    "Sweet Potato": 86,
    "Tapioca": 160,              # cassava

    # Fruits
    "Banana": 89,
    "Cashewnut": 553,

    # Spices (kcal per 100g dry weight)
    "Dry Chillies": 246,
    "Turmeric": 312,
    "Ginger": 80,
    "Garlic": 149,
    "Coriander": 298,
    "Black Pepper": 251,
    "Cardamom": 311,
    "Arecanut": 230,             # betel nut

    # Drugs & Narcotics
    "Tobacco": 0,                # not a food crop

    # Fiber Crops (non-food)
    "Cotton(Lint)": 0,
    "Jute": 0,
    "Mesta": 0,
    "Sannhamp": 0,

    # Fodder
    "Guar Seed": 0,              # primarily fodder/industrial gum, not direct food
}

# Define food crop types (for food_crop_kcal calculation)
FOOD_CROP_TYPES = {"Cereals", "Pulses", "Oilseeds", "Fruits", "Vegetable", "Sugar", "Spices"}
NON_FOOD_CROP_TYPES = {"Fiber Crops", "Drugs And Narcotics", "Fodder"}

# Nutritional categories for share computation
NUTRITION_CATEGORIES = {
    "cereal": "Cereals",
    "pulse": "Pulses",
    "oilseed": "Oilseeds",
    "sugar": "Sugar",
    "vegetable": "Vegetable",
    "fruit": "Fruits",
    "spice": "Spices",
}

print(f"  Kcal values defined for {len(KCAL_PER_100G)} crops")
print(f"  Food crop types: {FOOD_CROP_TYPES}")

# ---------------------------------------------------------------------------
# Step 2: Load and process APY data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Computing district-level kcal from APY data")
print("=" * 70)

print("  Loading APY data...")
apy = pd.read_csv(APY_PATH)
print(f"  Raw rows: {len(apy):,}")

# Clean up
apy["state_name"] = apy["state_name"].str.strip()
apy["district_name"] = apy["district_name"].str.strip()
apy["crop_name"] = apy["crop_name"].str.strip()
apy["crop_type"] = apy["crop_type"].str.strip()

# Drop rows with missing production
apy = apy.dropna(subset=["production", "area"])
apy = apy[apy["production"] > 0]
print(f"  Rows with valid production: {len(apy):,}")

# Map kcal values
apy["kcal_per_100g"] = apy["crop_name"].map(KCAL_PER_100G)
unmapped = apy[apy["kcal_per_100g"].isna()]["crop_name"].unique()
if len(unmapped) > 0:
    print(f"  WARNING: No kcal mapping for: {unmapped}")
apy["kcal_per_100g"] = apy["kcal_per_100g"].fillna(0)

# Compute kcal for each row: production(tonnes) * 1000(kg/tonne) * kcal_per_100g / 100
# = production * 1000 * kcal_per_100g / 100 = production * 10 * kcal_per_100g
apy["row_kcal"] = apy["production"] * 10 * apy["kcal_per_100g"]

# Flag food crops
apy["is_food_crop"] = apy["crop_type"].isin(FOOD_CROP_TYPES)

# Compute MEAN ANNUAL values per district-crop
# First, aggregate across seasons within a year
print("  Aggregating by district-crop-year...")
annual = (
    apy.groupby(["state_name", "district_name", "crop_name", "crop_type", "year"])
    .agg(
        annual_production=("production", "sum"),
        annual_area=("area", "sum"),
        annual_kcal=("row_kcal", "sum"),
        is_food=("is_food_crop", "first"),
    )
    .reset_index()
)

# Mean annual per district-crop
print("  Computing mean annual production per district-crop...")
mean_annual = (
    annual.groupby(["state_name", "district_name", "crop_name", "crop_type"])
    .agg(
        mean_production=("annual_production", "mean"),
        mean_area=("annual_area", "mean"),
        mean_kcal=("annual_kcal", "mean"),
        is_food=("is_food", "first"),
    )
    .reset_index()
)

# Add category info
mean_annual["nutrition_cat"] = mean_annual["crop_type"]

# Aggregate to district level
print("  Aggregating to district level...")

# Total kcal per district
district_total = (
    mean_annual.groupby(["state_name", "district_name"])
    .agg(
        total_kcal_annual=("mean_kcal", "sum"),
        total_cropped_area=("mean_area", "sum"),
    )
    .reset_index()
)

# Food crop kcal
food_only = mean_annual[mean_annual["is_food"] == True]
district_food = (
    food_only.groupby(["state_name", "district_name"])
    .agg(food_crop_kcal_annual=("mean_kcal", "sum"))
    .reset_index()
)

# Kcal by nutritional category
cat_kcal = (
    mean_annual.groupby(["state_name", "district_name", "crop_type"])
    .agg(cat_kcal=("mean_kcal", "sum"))
    .reset_index()
)
cat_pivot = cat_kcal.pivot_table(
    index=["state_name", "district_name"],
    columns="crop_type",
    values="cat_kcal",
    fill_value=0,
).reset_index()

# Top crops by kcal per district
print("  Finding top crops by kcal contribution per district...")
top_crops = (
    mean_annual.sort_values("mean_kcal", ascending=False)
    .groupby(["state_name", "district_name"])
    .head(5)
    .groupby(["state_name", "district_name"])
    .apply(lambda g: ", ".join(f"{r['crop_name']}({r['mean_kcal']/g['mean_kcal'].sum()*100:.0f}%)" for _, r in g.iterrows()), include_groups=False)
    .reset_index()
    .rename(columns={0: "top_crops_by_kcal"})
)

# Merge everything
print("  Merging district-level kcal data...")
district_kcal = district_total.merge(district_food, on=["state_name", "district_name"], how="left")
district_kcal = district_kcal.merge(top_crops, on=["state_name", "district_name"], how="left")

# Compute kcal per hectare
district_kcal["kcal_per_hectare"] = (
    district_kcal["total_kcal_annual"] / district_kcal["total_cropped_area"]
)

# Add category shares
for short_name, crop_type in NUTRITION_CATEGORIES.items():
    col_name = f"{short_name}_kcal_share"
    if crop_type in cat_pivot.columns:
        merged = cat_pivot[["state_name", "district_name", crop_type]].copy()
        merged = merged.rename(columns={crop_type: col_name})
        district_kcal = district_kcal.merge(merged, on=["state_name", "district_name"], how="left")
        district_kcal[col_name] = district_kcal[col_name].fillna(0) / district_kcal["total_kcal_annual"]
    else:
        district_kcal[col_name] = 0.0

# Food crop kcal share (NOTE: always ~1.0 since non-food crops have 0 kcal -- not useful)
# Instead compute food crop AREA share from diversity data (after merge)
district_kcal["food_crop_kcal_share"] = (
    district_kcal["food_crop_kcal_annual"].fillna(0) / district_kcal["total_kcal_annual"]
)

print(f"  Districts with kcal data: {len(district_kcal)}")
print(f"  Mean kcal/ha: {district_kcal['kcal_per_hectare'].mean():,.0f}")

# ---------------------------------------------------------------------------
# Step 3: Merge with diversity indices
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Merging with diversity indices")
print("=" * 70)

div = pd.read_csv(DIV_PATH)
print(f"  Diversity index districts: {len(div)}")

# Normalize names for matching
def normalize(s):
    s = str(s).upper().strip()
    s = s.replace("&", "AND")
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

div["state_norm"] = div["state_name"].apply(normalize)
div["district_norm"] = div["district_name"].apply(normalize)
district_kcal["state_norm"] = district_kcal["state_name"].apply(normalize)
district_kcal["district_norm"] = district_kcal["district_name"].apply(normalize)

# Direct merge on normalized names
merged = div.merge(
    district_kcal.drop(columns=["state_name", "district_name"]),
    on=["state_norm", "district_norm"],
    how="left",
)

# For unmatched, try fuzzy matching
unmatched_mask = merged["total_kcal_annual"].isna()
print(f"  Direct match: {(~unmatched_mask).sum()} / {len(merged)}")

if unmatched_mask.sum() > 0:
    print(f"  Attempting fuzzy match for {unmatched_mask.sum()} remaining...")
    kcal_lookup = {}
    for _, row in district_kcal.iterrows():
        kcal_lookup[(row["state_norm"], row["district_norm"])] = row

    for idx in merged[unmatched_mask].index:
        div_st = merged.at[idx, "state_norm"]
        div_dt = merged.at[idx, "district_norm"]

        best_score = 0
        best_key = None

        # Same-state fuzzy
        for (kst, kdt), krow in kcal_lookup.items():
            if fuzz.ratio(kst, div_st) > 80:
                score = fuzz.ratio(div_dt, kdt)
                if score > best_score:
                    best_score = score
                    best_key = (kst, kdt)

        if best_score >= 80 and best_key is not None:
            krow = kcal_lookup[best_key]
            for col in district_kcal.columns:
                if col not in ["state_name", "district_name", "state_norm", "district_norm"]:
                    if col in merged.columns:
                        merged.at[idx, col] = krow[col]

    matched_after = (~merged["total_kcal_annual"].isna()).sum()
    print(f"  After fuzzy match: {matched_after} / {len(merged)}")

# Drop helper columns
merged = merged.drop(columns=["state_norm", "district_norm"], errors="ignore")

# Compute food crop AREA share (meaningful, unlike kcal share which is always ~1.0)
# Food crops = cereals + pulses + oilseeds + fruits + vegetables + sugar + spices
# Non-food = fiber_crops + drugs_and_narcotics + fodder
non_food_cols = ["share_fiber_crops", "share_drugs_and_narcotics", "share_fodder"]
for c in non_food_cols:
    if c not in merged.columns:
        merged[c] = 0.0
merged["food_crop_area_share"] = 1.0 - merged[non_food_cols].fillna(0).sum(axis=1)
merged["food_crop_area_share"] = merged["food_crop_area_share"].clip(0, 1)

# Replace the useless kcal share with the meaningful area share
merged["food_crop_kcal_share"] = merged["food_crop_area_share"]

print(f"  Food crop area share: mean={merged['food_crop_area_share'].mean():.1%}, "
      f"min={merged['food_crop_area_share'].min():.1%}, max={merged['food_crop_area_share'].max():.1%}")

# ---------------------------------------------------------------------------
# Step 4: Classify into quadrants
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Classifying districts into diversity-calorie quadrants")
print("=" * 70)

valid = merged.dropna(subset=["agro_biodiversity_index", "kcal_per_hectare"])
abi_median = valid["agro_biodiversity_index"].median()
kcal_ha_median = valid["kcal_per_hectare"].median()

print(f"  ABI median: {abi_median:.4f}")
print(f"  Kcal/ha median: {kcal_ha_median:,.0f}")

def classify_quadrant(row):
    abi = row.get("agro_biodiversity_index")
    kph = row.get("kcal_per_hectare")
    if pd.isna(abi) or pd.isna(kph):
        return "No Data"
    if abi > abi_median and kph > kcal_ha_median:
        return "Diverse & Calorie-Rich"
    elif abi <= abi_median and kph > kcal_ha_median:
        return "Monoculture Breadbasket"
    elif abi > abi_median and kph <= kcal_ha_median:
        return "Diverse & Calorie-Poor"
    else:
        return "Vulnerable"

merged["kcal_diversity_quadrant"] = merged.apply(classify_quadrant, axis=1)

quad_counts = merged["kcal_diversity_quadrant"].value_counts()
print("\n  Quadrant distribution:")
for q, c in quad_counts.items():
    print(f"    {q}: {c}")

# Save merged CSV
OUT_CSV = OUT_DIR / "district_diversity_calorie_merged.csv"
merged.to_csv(OUT_CSV, index=False)
print(f"\n  Saved: {OUT_CSV}")

# ---------------------------------------------------------------------------
# Step 5: Shapefile matching (reuse MANUAL_MAP from script 60)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Loading shapefile and matching districts")
print("=" * 70)

gdf = gpd.read_file(SHP_PATH)
print(f"  Shapefile districts: {len(gdf)}")

# Same MANUAL_MAP as script 60
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

# Build lookup from merged CSV
csv_keys = {}
for i, row in merged.iterrows():
    st = normalize(row["state_name"])
    dt = normalize(row["district_name"])
    csv_keys[(st, dt)] = i

# Match shapefile districts
match_col = [None] * len(gdf)
for idx, row in gdf.iterrows():
    shp_st = normalize(row["stname"])
    shp_dt = normalize(row["dtname"])

    if (shp_st, shp_dt) in csv_keys:
        match_col[idx] = csv_keys[(shp_st, shp_dt)]
        continue

    mapped = MANUAL_MAP.get((shp_st, shp_dt))
    if mapped and mapped in csv_keys:
        match_col[idx] = csv_keys[mapped]
        continue

    district_only = [(k, v) for k, v in csv_keys.items() if k[1] == shp_dt]
    if len(district_only) == 1:
        match_col[idx] = district_only[0][1]
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

gdf["csv_idx"] = match_col
matched_count = sum(1 for x in match_col if x is not None)
print(f"  Shapefile match: {matched_count}/{len(gdf)} ({matched_count/len(gdf)*100:.1f}%)")

# Transfer data to geodataframe
transfer_cols = [
    "kcal_per_hectare", "total_kcal_annual", "food_crop_kcal_annual",
    "kcal_diversity_quadrant", "agro_biodiversity_index",
    "food_crop_kcal_share", "irrigation_regime",
    "cereal_kcal_share", "pulse_kcal_share", "oilseed_kcal_share",
    "sugar_kcal_share", "vegetable_kcal_share", "fruit_kcal_share",
    "spice_kcal_share",
]

for col in transfer_cols:
    gdf[col] = None

for idx, row in gdf.iterrows():
    csv_idx_val = row["csv_idx"]
    if csv_idx_val is not None and not pd.isna(csv_idx_val):
        csv_idx_val = int(csv_idx_val)
        for col in transfer_cols:
            if col in merged.columns:
                gdf.at[idx, col] = merged.at[csv_idx_val, col]

gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
gdf = gdf.to_crs(epsg=4326)

# ---------------------------------------------------------------------------
# Step 6: Generate static maps
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Generating static maps")
print("=" * 70)


def add_map_furniture(ax, title, source_note=True):
    """Add title, north arrow, source note to academic-style map."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16, fontfamily="serif")
    ax.set_axis_off()
    # North arrow
    x, y = 0.95, 0.95
    ax.annotate(
        "N", xy=(x, y), xycoords="axes fraction",
        ha="center", va="center", fontsize=11, fontweight="bold",
    )
    ax.annotate(
        "", xy=(x, y - 0.01), xycoords="axes fraction",
        xytext=(x, y - 0.06), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    if source_note:
        ax.annotate(
            "Source: India Data Portal (1997-2021) | CEEW Analysis",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=7, color="#666", fontstyle="italic",
        )


# --- Map 1: Kcal per hectare choropleth ---
print("  [1/5] Kcal per hectare choropleth...")
fig, ax = plt.subplots(1, 1, figsize=(12, 14))
gdf_valid = gdf[gdf["kcal_per_hectare"].notna()].copy()
gdf_valid["kcal_per_hectare"] = pd.to_numeric(gdf_valid["kcal_per_hectare"], errors="coerce")

# Plot background (no data)
gdf.plot(ax=ax, color="#f0f0f0", edgecolor="#ccc", linewidth=0.3)
# Plot data
gdf_valid.plot(
    ax=ax, column="kcal_per_hectare", cmap="YlOrRd", scheme="quantiles", k=7,
    edgecolor="#888", linewidth=0.2, legend=True,
    legend_kwds={"title": "Kcal/ha (quantiles)", "fontsize": 8, "title_fontsize": 9, "loc": "lower left"},
)
add_map_furniture(ax, "District-Level Caloric Productivity (Mean Annual Kcal per Hectare)")
fig.tight_layout()
fig.savefig(str(OUT_DIR / "kcal_per_hectare_choropleth.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("    Saved: kcal_per_hectare_choropleth.png")

# --- Map 2: Quadrant map ---
print("  [2/5] Quadrant map...")
QUAD_COLORS = {
    "Diverse & Calorie-Rich": "#2166ac",
    "Monoculture Breadbasket": "#ef8a62",
    "Diverse & Calorie-Poor": "#67a9cf",
    "Vulnerable": "#d6604d",
    "No Data": "#f0f0f0",
}

fig, ax = plt.subplots(1, 1, figsize=(12, 14))
gdf["quad_color"] = gdf["kcal_diversity_quadrant"].map(QUAD_COLORS).fillna("#f0f0f0")
gdf.plot(ax=ax, color=gdf["quad_color"], edgecolor="#888", linewidth=0.2)
legend_patches = [mpatches.Patch(color=c, label=q) for q, c in QUAD_COLORS.items()]
ax.legend(handles=legend_patches, loc="lower left", fontsize=8, title="Quadrant", title_fontsize=9)
add_map_furniture(ax, "Diversity-Calorie Quadrant Classification of Indian Districts")
fig.tight_layout()
fig.savefig(str(OUT_DIR / "quadrant_map.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("    Saved: quadrant_map.png")

# --- Map 3: Bivariate map (ABI x Kcal/ha) ---
print("  [3/5] Bivariate map...")

def bivariate_color(abi_val, kcal_val, abi_terciles, kcal_terciles):
    """Return color from a 3x3 bivariate palette."""
    # Blue-yellow bivariate: rows=kcal(low->high), cols=ABI(low->high)
    palette = [
        ["#e8e8e8", "#ace4e4", "#5ac8c8"],  # low kcal
        ["#dfb0d6", "#a5add3", "#5698b9"],   # mid kcal
        ["#be64ac", "#8c62aa", "#3b4994"],   # high kcal
    ]
    if pd.isna(abi_val) or pd.isna(kcal_val):
        return "#f0f0f0"
    abi_bin = 0 if abi_val <= abi_terciles[0] else (1 if abi_val <= abi_terciles[1] else 2)
    kcal_bin = 0 if kcal_val <= kcal_terciles[0] else (1 if kcal_val <= kcal_terciles[1] else 2)
    return palette[kcal_bin][abi_bin]

gdf_biv = gdf.copy()
gdf_biv["abi_f"] = pd.to_numeric(gdf_biv["agro_biodiversity_index"], errors="coerce")
gdf_biv["kcal_f"] = pd.to_numeric(gdf_biv["kcal_per_hectare"], errors="coerce")

abi_t = gdf_biv["abi_f"].quantile([1/3, 2/3]).values
kcal_t = gdf_biv["kcal_f"].quantile([1/3, 2/3]).values

gdf_biv["biv_color"] = gdf_biv.apply(
    lambda r: bivariate_color(r["abi_f"], r["kcal_f"], abi_t, kcal_t), axis=1
)

fig, ax = plt.subplots(1, 1, figsize=(12, 14))
gdf_biv.plot(ax=ax, color=gdf_biv["biv_color"], edgecolor="#888", linewidth=0.2)
add_map_furniture(ax, "Bivariate Map: Agro-Biodiversity Index vs Caloric Productivity")

# Bivariate legend
palette = [
    ["#e8e8e8", "#ace4e4", "#5ac8c8"],
    ["#dfb0d6", "#a5add3", "#5698b9"],
    ["#be64ac", "#8c62aa", "#3b4994"],
]
legend_ax = fig.add_axes([0.12, 0.08, 0.12, 0.12])
for i in range(3):
    for j in range(3):
        legend_ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=palette[i][j], edgecolor="white", lw=0.5))
legend_ax.set_xlim(0, 3)
legend_ax.set_ylim(0, 3)
legend_ax.set_xlabel("ABI  \u2192", fontsize=8)
legend_ax.set_ylabel("Kcal/ha  \u2192", fontsize=8)
legend_ax.set_xticks([0.5, 1.5, 2.5])
legend_ax.set_xticklabels(["Low", "Mid", "High"], fontsize=7)
legend_ax.set_yticks([0.5, 1.5, 2.5])
legend_ax.set_yticklabels(["Low", "Mid", "High"], fontsize=7)
legend_ax.tick_params(length=0)

fig.savefig(str(OUT_DIR / "bivariate_abi_kcal_map.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("    Saved: bivariate_abi_kcal_map.png")

# --- Map 4: Nutritionally hollow diversity map ---
print("  [4/5] Nutritionally hollow diversity map...")
fig, ax = plt.subplots(1, 1, figsize=(12, 14))

gdf["abi_f"] = pd.to_numeric(gdf["agro_biodiversity_index"], errors="coerce")
gdf["food_share_f"] = pd.to_numeric(gdf["food_crop_kcal_share"], errors="coerce")

abi_med = gdf["abi_f"].median()

def hollow_color(row):
    abi = row["abi_f"]
    fs = row["food_share_f"]
    if pd.isna(abi) or pd.isna(fs):
        return "#f0f0f0"
    if abi > abi_med and fs < 0.5:
        return "#d73027"   # Nutritionally hollow: high diversity, low food share
    elif abi > abi_med and fs >= 0.5:
        return "#1a9850"   # Diverse & nutritious
    elif abi <= abi_med and fs < 0.5:
        return "#fdae61"   # Low diversity, low food
    else:
        return "#a6d96a"   # Low diversity but food-focused

gdf["hollow_color"] = gdf.apply(hollow_color, axis=1)
gdf.plot(ax=ax, color=gdf["hollow_color"], edgecolor="#888", linewidth=0.2)

legend_patches = [
    mpatches.Patch(color="#d73027", label="Nutritionally Hollow (High ABI, <50% food kcal)"),
    mpatches.Patch(color="#1a9850", label="Diverse & Nutritious (High ABI, >50% food kcal)"),
    mpatches.Patch(color="#fdae61", label="Non-Food Focus (Low ABI, <50% food kcal)"),
    mpatches.Patch(color="#a6d96a", label="Food-Focused (Low ABI, >50% food kcal)"),
    mpatches.Patch(color="#f0f0f0", label="No Data"),
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=7.5, title="Category", title_fontsize=9)
add_map_furniture(ax, "Nutritionally Hollow Diversity: Districts with High ABI but Low Food Calorie Share")
fig.tight_layout()
fig.savefig(str(OUT_DIR / "nutritionally_hollow_map.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("    Saved: nutritionally_hollow_map.png")

# --- Map 5 (Scatter): ABI vs Kcal/ha ---
print("  [5/5] ABI vs Kcal/ha scatter plot...")
fig, ax = plt.subplots(figsize=(12, 8))

scatter_df = merged.dropna(subset=["agro_biodiversity_index", "kcal_per_hectare"]).copy()
scatter_df["kcal_per_hectare"] = pd.to_numeric(scatter_df["kcal_per_hectare"])
scatter_df["agro_biodiversity_index"] = pd.to_numeric(scatter_df["agro_biodiversity_index"])

irr_colors = {
    "Rainfed (<40%)": "#e41a1c",
    "Semi-Irrigated (40-60%)": "#ff7f00",
    "Irrigated (>60%)": "#377eb8",
}

for irr, color in irr_colors.items():
    mask = scatter_df["irrigation_regime"] == irr
    sub = scatter_df[mask]
    ax.scatter(sub["agro_biodiversity_index"], sub["kcal_per_hectare"],
               c=color, alpha=0.5, s=20, label=irr, edgecolors="none")

# Districts without irrigation data
mask_other = ~scatter_df["irrigation_regime"].isin(irr_colors.keys())
sub_other = scatter_df[mask_other]
if len(sub_other) > 0:
    ax.scatter(sub_other["agro_biodiversity_index"], sub_other["kcal_per_hectare"],
               c="#999", alpha=0.3, s=15, label="Unknown", edgecolors="none")

# Quadrant lines
ax.axvline(x=abi_median, color="#333", linestyle="--", linewidth=1, alpha=0.6)
ax.axhline(y=kcal_ha_median, color="#333", linestyle="--", linewidth=1, alpha=0.6)

# Quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
label_props = dict(fontsize=8, alpha=0.5, ha="center", va="center", fontstyle="italic")
ax.text((xlim[0] + abi_median) / 2, (ylim[1] + kcal_ha_median) / 2,
        "Monoculture\nBreadbasket", **label_props)
ax.text((xlim[1] + abi_median) / 2, (ylim[1] + kcal_ha_median) / 2,
        "Diverse &\nCalorie-Rich", **label_props)
ax.text((xlim[0] + abi_median) / 2, (ylim[0] + kcal_ha_median) / 2,
        "Vulnerable", **label_props)
ax.text((xlim[1] + abi_median) / 2, (ylim[0] + kcal_ha_median) / 2,
        "Diverse &\nCalorie-Poor", **label_props)

ax.set_xlabel("Agro-Biodiversity Index (ABI)", fontsize=11)
ax.set_ylabel("Caloric Productivity (Kcal per Hectare)", fontsize=11)
ax.set_title("Crop Diversity vs Caloric Productivity Across Indian Districts", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.2)
ax.annotate(
    "Source: India Data Portal (1997-2021) | CEEW Analysis",
    xy=(0.01, 0.01), xycoords="axes fraction",
    fontsize=7, color="#666", fontstyle="italic",
)

fig.tight_layout()
fig.savefig(str(OUT_DIR / "abi_vs_kcal_scatter.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("    Saved: abi_vs_kcal_scatter.png")

# ---------------------------------------------------------------------------
# Step 7: Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Summary Statistics")
print("=" * 70)

valid_merged = merged.dropna(subset=["kcal_per_hectare"])
print(f"\n  Total districts with kcal data: {len(valid_merged)}")
print(f"  Total districts in diversity dataset: {len(merged)}")

# Mean kcal/ha by irrigation regime
print("\n  Mean Kcal/ha by irrigation regime:")
irr_stats = valid_merged.groupby("irrigation_regime")["kcal_per_hectare"].agg(["mean", "median", "count"])
for irr, row in irr_stats.iterrows():
    print(f"    {irr:30s}  mean={row['mean']:>12,.0f}  median={row['median']:>12,.0f}  n={int(row['count'])}")

# Quadrant distribution
print("\n  Quadrant distribution:")
for q, c in merged["kcal_diversity_quadrant"].value_counts().items():
    pct = c / len(merged) * 100
    print(f"    {q:30s}  {c:4d} ({pct:.1f}%)")

# Top 10 / Bottom 10
print("\n  Top 10 districts by Kcal/ha:")
top10 = valid_merged.nlargest(10, "kcal_per_hectare")[["state_name", "district_name", "kcal_per_hectare", "agro_biodiversity_index"]]
for _, r in top10.iterrows():
    print(f"    {r['state_name']:20s} {r['district_name']:25s}  {r['kcal_per_hectare']:>12,.0f} kcal/ha  ABI={r['agro_biodiversity_index']:.3f}")

print("\n  Bottom 10 districts by Kcal/ha:")
bot10 = valid_merged.nsmallest(10, "kcal_per_hectare")[["state_name", "district_name", "kcal_per_hectare", "agro_biodiversity_index"]]
for _, r in bot10.iterrows():
    print(f"    {r['state_name']:20s} {r['district_name']:25s}  {r['kcal_per_hectare']:>12,.0f} kcal/ha  ABI={r['agro_biodiversity_index']:.3f}")

# Correlation
corr = valid_merged[["agro_biodiversity_index", "kcal_per_hectare"]].corr().iloc[0, 1]
print(f"\n  Pearson correlation (ABI vs Kcal/ha): {corr:.4f}")

# Spearman
from scipy.stats import spearmanr
sp_corr, sp_pval = spearmanr(
    valid_merged["agro_biodiversity_index"],
    valid_merged["kcal_per_hectare"],
)
print(f"  Spearman correlation (ABI vs Kcal/ha): {sp_corr:.4f} (p={sp_pval:.2e})")

print("\n" + "=" * 70)
print("DONE. All outputs saved to:", OUT_DIR)
print("=" * 70)
