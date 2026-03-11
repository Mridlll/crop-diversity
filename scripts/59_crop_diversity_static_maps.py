"""
59_crop_diversity_static_maps.py
Generate publication-quality static maps and animated GIFs for crop diversity analysis.

Outputs saved to: outputs/crop_diversity_analysis/maps/
Requires: matplotlib, geopandas, pandas, numpy, imageio, pillow

Usage:
    python scripts/59_crop_diversity_static_maps.py
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio.v2 as imageio
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs" / "crop_diversity_analysis"
SHP_PATH = BASE_DIR / "Package_Maps_Share_20251120_FINAL" / "shapefiles" / "in_district.shp"
OUT_DIR = DATA_DIR / "maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_NOTE = "Data: India Data Portal, 1997\u20132021"

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

# Try to use a nice serif font, fall back gracefully
try:
    import matplotlib.font_manager as fm
    serif_fonts = [f.name for f in fm.fontManager.ttflist if 'Times' in f.name or 'Garamond' in f.name or 'Georgia' in f.name]
    if serif_fonts:
        plt.rcParams['font.family'] = serif_fonts[0]
except Exception:
    pass


def normalize_name(s: str) -> str:
    """Uppercase, & -> AND, strip non-alphanumeric (except |), collapse whitespace."""
    import re as _re
    s = str(s).upper().strip()
    s = s.replace("&", "AND")
    s = _re.sub(r"[^A-Z0-9 |]", "", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


# ---- Manual mapping: (shp_state, shp_district) -> (csv_state, csv_district) ----
# Maps shapefile names to the CSV district_key components they should match.
# Covers: state name changes, spelling differences, renamed districts,
#         state reorganization, and UT mergers.
MANUAL_DISTRICT_MAP = {
    # --- State name differences ---
    ("ANDAMAN AND NICOBAR", "NICOBARS"): ("ANDAMAN AND NICOBAR ISLANDS", "NICOBARS"),
    ("ANDAMAN AND NICOBAR", "NORTH AND MIDDLE ANDAMAN"): ("ANDAMAN AND NICOBAR ISLANDS", "NORTH AND MIDDLE ANDAMAN"),
    ("ANDAMAN AND NICOBAR", "SOUTH ANDAMAN"): ("ANDAMAN AND NICOBAR ISLANDS", "SOUTH ANDAMANS"),
    # --- DADRA AND NAGAR HAVELI / DAMAN AND DIU (UT merger) ---
    ("DADRA AND NAGAR HAVE", "DADRA AND NAGAR HAVELI"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DADRA AND NAGAR HAVELI"),
    ("DAMAN AND DIU", "DAMAN"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DAMAN"),
    ("DAMAN AND DIU", "DIU"): ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DIU"),
    # --- Andhra Pradesh ---
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
    # --- Chhattisgarh ---
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


def add_source_note(fig, note=SOURCE_NOTE):
    fig.text(0.5, 0.01, note, ha='center', va='bottom', fontsize=7, style='italic', color='gray')


def add_north_arrow(ax, x=0.95, y=0.95, arrow_length=0.06):
    ax.annotate('N', xy=(x, y), xycoords='axes fraction',
                ha='center', va='center', fontsize=10, fontweight='bold',
                arrowprops=dict(facecolor='black', width=3, headwidth=8, headlength=6),
                xytext=(x, y - arrow_length))


def save_fig(fig, name):
    """Save as both PNG and PDF."""
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches='tight')
    try:
        fig.savefig(str(pdf_path), bbox_inches='tight')
        print(f"  Saved: {png_path.name} + {pdf_path.name}")
    except PermissionError:
        print(f"  Saved: {png_path.name} (PDF skipped — file is open elsewhere)")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load & merge data
# ---------------------------------------------------------------------------
def _apply_manual_mapping(gdf):
    """Apply MANUAL_DISTRICT_MAP to remap shapefile join_keys to CSV join_keys."""
    for idx, row in gdf.iterrows():
        shp_st = normalize_name(row['stname'])
        shp_dt = normalize_name(row['dtname'])
        mapped = MANUAL_DISTRICT_MAP.get((shp_st, shp_dt))
        if mapped:
            gdf.at[idx, 'join_key'] = mapped[0] + "|" + mapped[1]
    return gdf


def load_data():
    print("Loading data...")
    indices = pd.read_csv(DATA_DIR / "district_diversity_indices.csv")
    panel = pd.read_csv(DATA_DIR / "district_year_diversity_panel.csv")
    change = pd.read_csv(DATA_DIR / "district_diversity_change.csv")
    gdf = gpd.read_file(str(SHP_PATH))

    # Create join keys
    gdf['_state_norm'] = gdf['stname'].apply(normalize_name)
    gdf['_dist_norm'] = gdf['dtname'].apply(normalize_name)
    gdf['join_key'] = gdf['_state_norm'] + "|" + gdf['_dist_norm']

    # Apply manual mapping to fix known mismatches
    gdf = _apply_manual_mapping(gdf)

    indices['join_key'] = indices['district_key'].apply(normalize_name)
    change['join_key'] = change['district_key'].apply(normalize_name)

    # Panel: build join key from state_name + district_name
    panel['join_key'] = (panel['state_name'].apply(normalize_name)
                         + "|"
                         + panel['district_name'].apply(normalize_name))

    # Merge indices onto shapefile
    gdf_idx = gdf.merge(indices, on='join_key', how='left', suffixes=('', '_csv'))
    matched = gdf_idx['shannon_index'].notna().sum()
    print(f"  Matched {matched}/{len(gdf)} districts (indices)")

    # Merge change onto shapefile
    gdf_chg = gdf.merge(change, on='join_key', how='left', suffixes=('', '_chg'))
    matched_chg = gdf_chg['shannon_change'].notna().sum()
    print(f"  Matched {matched_chg}/{len(gdf)} districts (change)")

    # State boundaries (dissolve)
    states = gdf.dissolve(by='stname').reset_index()

    return gdf_idx, gdf_chg, panel, gdf, states


# ---------------------------------------------------------------------------
# Map 1-4: Single index choropleth
# ---------------------------------------------------------------------------
def plot_single_choropleth(gdf_idx, states, col, title, cmap, label, fname, vmin=None, vmax=None):
    print(f"Generating: {title}")
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Background: all districts in light gray
    gdf_idx.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.15)

    # Data layer
    data = gdf_idx[gdf_idx[col].notna()]
    n = len(data)
    data.plot(ax=ax, column=col, cmap=cmap, edgecolor='#cccccc', linewidth=0.15,
              legend=True, vmin=vmin, vmax=vmax,
              legend_kwds={'label': label, 'shrink': 0.6, 'pad': 0.02})

    # State boundaries
    states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    ax.set_title(f"{title}\n(n = {n} districts)", fontsize=14, fontweight='bold', pad=12)
    ax.set_axis_off()
    add_north_arrow(ax)

    try:
        ax.add_artist(ScaleBar(1, units='m', location='lower left', length_fraction=0.2,
                                box_alpha=0.5, font_properties={'size': 7}))
    except Exception:
        pass

    add_source_note(fig)
    save_fig(fig, fname)


# ---------------------------------------------------------------------------
# Map 5: Irrigation regime (categorical)
# ---------------------------------------------------------------------------
def plot_irrigation_map(gdf_idx, states):
    print("Generating: Irrigation Regime Map")
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Background
    gdf_idx.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.15)

    color_map = {
        'Rainfed (<40%)': '#8B6914',       # earth tone
        'Semi-Irrigated (40-60%)': '#DAA520', # amber/goldenrod
        'Irrigated (>60%)': '#4682B4',      # steel blue
    }

    for regime, color in color_map.items():
        subset = gdf_idx[gdf_idx['irrigation_regime'] == regime]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='#cccccc', linewidth=0.15)

    states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    patches = [mpatches.Patch(facecolor=c, edgecolor='gray', label=f"{k} ({len(gdf_idx[gdf_idx['irrigation_regime']==k])})")
               for k, c in color_map.items()]
    patches.append(mpatches.Patch(facecolor='#f0f0f0', edgecolor='gray', label='No data'))
    ax.legend(handles=patches, loc='lower left', fontsize=9, framealpha=0.9, title='Irrigation Regime')

    n = gdf_idx['irrigation_regime'].notna().sum()
    ax.set_title(f"Irrigation Regime Classification\n(n = {n} classified districts)", fontsize=14, fontweight='bold', pad=12)
    ax.set_axis_off()
    add_north_arrow(ax)
    add_source_note(fig)
    save_fig(fig, 'map_irrigation_regime')


# ---------------------------------------------------------------------------
# Map 6: Diversity change (diverging)
# ---------------------------------------------------------------------------
def plot_diversity_change(gdf_chg, states):
    print("Generating: Diversity Change Map")
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    gdf_chg.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.15)

    data = gdf_chg[gdf_chg['shannon_change'].notna()]
    n = len(data)
    vabs = max(abs(data['shannon_change'].quantile(0.02)), abs(data['shannon_change'].quantile(0.98)))
    data.plot(ax=ax, column='shannon_change', cmap='RdBu', edgecolor='#cccccc', linewidth=0.15,
              legend=True, vmin=-vabs, vmax=vabs,
              legend_kwds={'label': 'Shannon Index Change (Early\u2192Late)', 'shrink': 0.6, 'pad': 0.02})

    states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    ax.set_title(f"Change in Crop Diversity (Early \u22642005 vs Late \u22652015)\n(n = {n} districts)",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_axis_off()
    add_north_arrow(ax)
    add_source_note(fig)
    save_fig(fig, 'map_diversity_change')


# ---------------------------------------------------------------------------
# Map 7: Combined 2x2 panel
# ---------------------------------------------------------------------------
def plot_combined_panel(gdf_idx, states):
    print("Generating: Combined 2x2 Panel")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    configs = [
        ('shannon_index', 'Shannon Index (H\')', 'viridis', axes[0, 0]),
        ('simpson_index', 'Simpson Index', 'viridis', axes[0, 1]),
        ('crop_richness', 'Crop Richness (count)', 'YlGn', axes[1, 0]),
        ('agro_biodiversity_index', 'Agro-Biodiversity Index', 'RdYlGn', axes[1, 1]),
    ]

    for col, label, cmap, ax in configs:
        gdf_idx.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.1)
        data = gdf_idx[gdf_idx[col].notna()]
        data.plot(ax=ax, column=col, cmap=cmap, edgecolor='#cccccc', linewidth=0.1,
                  legend=True,
                  legend_kwds={'label': label, 'shrink': 0.5, 'pad': 0.02, 'aspect': 30})
        states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_axis_off()

    fig.suptitle("Crop Diversity Indices Across Indian Districts", fontsize=16, fontweight='bold', y=0.98)
    add_source_note(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, 'map_combined_panel_2x2')


# ---------------------------------------------------------------------------
# Map 8: Irrigation x Diversity panel (1x3)
# ---------------------------------------------------------------------------
def plot_irrigation_diversity_panel(gdf_idx, states):
    print("Generating: Irrigation x Diversity Panel (1x3)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 12))

    regimes = ['Rainfed (<40%)', 'Semi-Irrigated (40-60%)', 'Irrigated (>60%)']
    # Consistent color scale
    abi_data = gdf_idx[gdf_idx['agro_biodiversity_index'].notna()]
    vmin = abi_data['agro_biodiversity_index'].quantile(0.02)
    vmax = abi_data['agro_biodiversity_index'].quantile(0.98)

    for ax, regime in zip(axes, regimes):
        gdf_idx.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.1)
        sub = gdf_idx[(gdf_idx['irrigation_regime'] == regime) & gdf_idx['agro_biodiversity_index'].notna()]
        n = len(sub)
        if n > 0:
            sub.plot(ax=ax, column='agro_biodiversity_index', cmap='RdYlGn',
                     edgecolor='#cccccc', linewidth=0.1,
                     vmin=vmin, vmax=vmax,
                     legend=True,
                     legend_kwds={'label': 'ABI', 'shrink': 0.4, 'pad': 0.02, 'aspect': 25})
        states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3)
        ax.set_title(f"{regime}\n(n = {n})", fontsize=12, fontweight='bold')
        ax.set_axis_off()

    fig.suptitle("Agro-Biodiversity Index by Irrigation Regime", fontsize=16, fontweight='bold', y=0.98)
    add_source_note(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, 'map_irrigation_diversity_panel')


# ---------------------------------------------------------------------------
# Map 9: Top/Bottom 20 districts by ABI
# ---------------------------------------------------------------------------
def plot_top_bottom(gdf_idx, states):
    print("Generating: Top/Bottom 20 Districts by ABI")
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    data = gdf_idx[gdf_idx['agro_biodiversity_index'].notna()].copy()
    top20_keys = data.nlargest(20, 'agro_biodiversity_index')['join_key'].values
    bot20_keys = data.nsmallest(20, 'agro_biodiversity_index')['join_key'].values

    gdf_idx.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.15)

    top = gdf_idx[gdf_idx['join_key'].isin(top20_keys)]
    bot = gdf_idx[gdf_idx['join_key'].isin(bot20_keys)]

    top.plot(ax=ax, color='#2ca02c', edgecolor='#1a7a1a', linewidth=0.4, alpha=0.85)
    bot.plot(ax=ax, color='#d62728', edgecolor='#a01010', linewidth=0.4, alpha=0.85)

    states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    legend_elements = [
        mpatches.Patch(facecolor='#2ca02c', edgecolor='#1a7a1a', label='Top 20 (highest ABI)'),
        mpatches.Patch(facecolor='#d62728', edgecolor='#a01010', label='Bottom 20 (lowest ABI)'),
        mpatches.Patch(facecolor='#f0f0f0', edgecolor='gray', label='Other districts'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

    ax.set_title("Top 20 & Bottom 20 Districts by Agro-Biodiversity Index", fontsize=14, fontweight='bold', pad=12)
    ax.set_axis_off()
    add_north_arrow(ax)
    add_source_note(fig)
    save_fig(fig, 'map_top_bottom_20_abi')


# ---------------------------------------------------------------------------
# GIF 1: Shannon Index timeline
# ---------------------------------------------------------------------------
def make_shannon_timeline_gif(panel, gdf, states):
    print("Generating: Shannon Index Timeline GIF...")
    years = sorted(panel['year_start'].unique())

    # Global min/max for consistent scale
    vmin = panel['shannon_index'].quantile(0.01)
    vmax = panel['shannon_index'].quantile(0.99)

    frames = []
    tmp_dir = OUT_DIR / "_tmp_frames"
    tmp_dir.mkdir(exist_ok=True)

    for i, yr in enumerate(years):
        yr_data = panel[panel['year_start'] == yr][['join_key', 'shannon_index']].copy()
        yr_data['join_key'] = yr_data['join_key'].apply(normalize_name)
        merged = gdf.merge(yr_data, on='join_key', how='left')

        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        merged.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.1)

        data = merged[merged['shannon_index'].notna()]
        if len(data) > 0:
            data.plot(ax=ax, column='shannon_index', cmap='viridis',
                      edgecolor='#cccccc', linewidth=0.1,
                      vmin=vmin, vmax=vmax,
                      legend=True,
                      legend_kwds={'label': "Shannon Index (H')", 'shrink': 0.5, 'pad': 0.02})

        states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.4)

        yr_label = f"{yr}\u2013{str(yr+1)[-2:]}"
        ax.set_title(f"Crop Diversity: Shannon Index", fontsize=14, fontweight='bold')
        ax.text(0.5, 0.94, yr_label, transform=ax.transAxes, fontsize=28,
                fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.set_axis_off()
        add_source_note(fig)

        frame_path = tmp_dir / f"frame_{i:03d}.png"
        fig.savefig(str(frame_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        frames.append(str(frame_path))
        print(f"  Frame {i+1}/{len(years)}: {yr_label}")

    # Assemble GIF
    gif_path = OUT_DIR / "gif_shannon_timeline.gif"
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(str(gif_path), images, duration=0.5, loop=0)
    print(f"  Saved: {gif_path.name} ({len(frames)} frames)")

    # Cleanup
    for f in frames:
        os.remove(f)
    tmp_dir.rmdir()


# ---------------------------------------------------------------------------
# GIF 2: Diversity by Irrigation Timeline (3 side-by-side)
# ---------------------------------------------------------------------------
def make_irrigation_diversity_gif(panel, gdf, gdf_idx, states):
    print("Generating: Diversity by Irrigation Timeline GIF...")
    years = sorted(panel['year_start'].unique())

    # Get irrigation regime mapping
    regime_map = gdf_idx[['join_key', 'irrigation_regime']].dropna(subset=['irrigation_regime'])
    regime_dict = dict(zip(regime_map['join_key'], regime_map['irrigation_regime']))

    vmin = panel['shannon_index'].quantile(0.01)
    vmax = panel['shannon_index'].quantile(0.99)

    regimes = ['Rainfed (<40%)', 'Semi-Irrigated (40-60%)', 'Irrigated (>60%)']
    frames = []
    tmp_dir = OUT_DIR / "_tmp_frames2"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, yr in enumerate(years):
        yr_data = panel[panel['year_start'] == yr][['join_key', 'shannon_index']].copy()
        yr_data['join_key'] = yr_data['join_key'].apply(normalize_name)
        merged = gdf.merge(yr_data, on='join_key', how='left')
        merged['irrigation_regime'] = merged['join_key'].map(regime_dict)

        fig, axes = plt.subplots(1, 3, figsize=(18, 10))

        for ax, regime in zip(axes, regimes):
            merged.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.08)
            sub = merged[(merged['irrigation_regime'] == regime) & merged['shannon_index'].notna()]
            if len(sub) > 0:
                sub.plot(ax=ax, column='shannon_index', cmap='viridis',
                         edgecolor='#cccccc', linewidth=0.08,
                         vmin=vmin, vmax=vmax,
                         legend=True,
                         legend_kwds={'label': "H'", 'shrink': 0.35, 'pad': 0.02, 'aspect': 20})
            states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3)
            ax.set_title(regime, fontsize=13, fontweight='bold')
            ax.set_axis_off()

        yr_label = f"{yr}\u2013{str(yr+1)[-2:]}"
        fig.suptitle(f"Shannon Index by Irrigation Regime", fontsize=15, fontweight='bold', y=0.97)
        fig.text(0.5, 0.92, yr_label, ha='center', fontsize=24, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
        add_source_note(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.90])

        frame_path = tmp_dir / f"frame_{i:03d}.png"
        fig.savefig(str(frame_path), dpi=120, bbox_inches='tight')
        plt.close(fig)
        frames.append(str(frame_path))
        print(f"  Frame {i+1}/{len(years)}: {yr_label}")

    gif_path = OUT_DIR / "gif_irrigation_diversity_timeline.gif"
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(str(gif_path), images, duration=0.5, loop=0)
    print(f"  Saved: {gif_path.name} ({len(frames)} frames)")

    for f in frames:
        os.remove(f)
    tmp_dir.rmdir()


# ---------------------------------------------------------------------------
# GIF 3: Crop Richness progression (yield not in panel)
# ---------------------------------------------------------------------------
def make_richness_timeline_gif(panel, gdf, states):
    print("Generating: Crop Richness Timeline GIF (yield not available, using richness)...")
    years = sorted(panel['year_start'].unique())

    vmin = panel['crop_richness'].quantile(0.01)
    vmax = panel['crop_richness'].quantile(0.99)

    frames = []
    tmp_dir = OUT_DIR / "_tmp_frames3"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, yr in enumerate(years):
        yr_data = panel[panel['year_start'] == yr][['join_key', 'crop_richness']].copy()
        yr_data['join_key'] = yr_data['join_key'].apply(normalize_name)
        merged = gdf.merge(yr_data, on='join_key', how='left')

        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        merged.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.1)

        data = merged[merged['crop_richness'].notna()]
        if len(data) > 0:
            data.plot(ax=ax, column='crop_richness', cmap='YlGn',
                      edgecolor='#cccccc', linewidth=0.1,
                      vmin=vmin, vmax=vmax,
                      legend=True,
                      legend_kwds={'label': 'Number of Crops', 'shrink': 0.5, 'pad': 0.02})

        states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.4)

        yr_label = f"{yr}\u2013{str(yr+1)[-2:]}"
        ax.set_title("Crop Richness (Number of Distinct Crops)", fontsize=14, fontweight='bold')
        ax.text(0.5, 0.94, yr_label, transform=ax.transAxes, fontsize=28,
                fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.set_axis_off()
        add_source_note(fig)

        frame_path = tmp_dir / f"frame_{i:03d}.png"
        fig.savefig(str(frame_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        frames.append(str(frame_path))
        print(f"  Frame {i+1}/{len(years)}: {yr_label}")

    gif_path = OUT_DIR / "gif_crop_richness_timeline.gif"
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(str(gif_path), images, duration=0.5, loop=0)
    print(f"  Saved: {gif_path.name} ({len(frames)} frames)")

    for f in frames:
        os.remove(f)
    tmp_dir.rmdir()


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate crop diversity static maps and GIFs")
    parser.add_argument('--only', nargs='+', default=None,
                        help='Generate only specific outputs. Choices: '
                             'shannon, simpson, richness, abi, irrigation, '
                             'change, panel_2x2, irrigation_panel, top_bottom, '
                             'gif_shannon, gif_irrigation, gif_richness')
    args = parser.parse_args()

    only = set(args.only) if args.only else None

    def should_run(name):
        return only is None or name in only

    print("=" * 60)
    print("Crop Diversity — Static Maps & Animated GIFs")
    if only:
        print(f"  (--only: {', '.join(sorted(only))})")
    print("=" * 60)

    gdf_idx, gdf_chg, panel, gdf, states = load_data()

    # --- Part 1: Static Maps ---
    print("\n--- Part 1: Static Maps ---")

    # 1. Shannon Index
    if should_run('shannon'):
        plot_single_choropleth(gdf_idx, states, 'shannon_index',
                               "Shannon Diversity Index (H')",
                               'viridis', "Shannon Index (H')", 'map_shannon_index')

    # 2. Simpson Index
    if should_run('simpson'):
        plot_single_choropleth(gdf_idx, states, 'simpson_index',
                               "Simpson Diversity Index",
                               'viridis', "Simpson Index", 'map_simpson_index')

    # 3. Crop Richness
    if should_run('richness'):
        plot_single_choropleth(gdf_idx, states, 'crop_richness',
                               "Crop Richness (Number of Distinct Crops)",
                               'YlGn', "Crop Richness", 'map_crop_richness')

    # 4. ABI
    if should_run('abi'):
        plot_single_choropleth(gdf_idx, states, 'agro_biodiversity_index',
                               "Agro-Biodiversity Index (ABI)",
                               'RdYlGn', "ABI", 'map_abi')

    # 5. Irrigation regime
    if should_run('irrigation'):
        plot_irrigation_map(gdf_idx, states)

    # 6. Diversity change
    if should_run('change'):
        plot_diversity_change(gdf_chg, states)

    # 7. Combined 2x2 panel
    if should_run('panel_2x2'):
        plot_combined_panel(gdf_idx, states)

    # 8. Irrigation x Diversity panel
    if should_run('irrigation_panel'):
        plot_irrigation_diversity_panel(gdf_idx, states)

    # 9. Top/Bottom 20
    if should_run('top_bottom'):
        plot_top_bottom(gdf_idx, states)

    # --- Part 2: Animated GIFs ---
    print("\n--- Part 2: Animated GIFs ---")

    gif1 = OUT_DIR / "gif_shannon_timeline.gif"
    gif2 = OUT_DIR / "gif_irrigation_diversity_timeline.gif"
    gif3 = OUT_DIR / "gif_crop_richness_timeline.gif"

    if should_run('gif_shannon'):
        if not gif1.exists() or only:
            make_shannon_timeline_gif(panel, gdf, states)
        else:
            print(f"Skipping (exists): {gif1.name}")

    if should_run('gif_irrigation'):
        if not gif2.exists() or only:
            make_irrigation_diversity_gif(panel, gdf, gdf_idx, states)
        else:
            print(f"Skipping (exists): {gif2.name}")

    if should_run('gif_richness'):
        if not gif3.exists() or only:
            make_richness_timeline_gif(panel, gdf, states)
        else:
            print(f"Skipping (exists): {gif3.name}")

    print("\n" + "=" * 60)
    print("All maps and GIFs generated successfully!")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
