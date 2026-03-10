"""
57_crop_diversity_agro_biodiversity.py

Compute district-level crop diversity indices and agro-biodiversity analysis
cross-cut with irrigation classification.

Indices computed:
  - Shannon Index (H'): -Σ(pi * ln(pi))
  - Simpson Index (1-D): 1 - Σ(pi²)
  - Crop Richness: number of distinct crops grown
  - Agro-Biodiversity Index (ABI): composite of above three (normalized + averaged)

Cross-cut:
  - Irrigation regime (rainfed <40%, semi-irrigated 40-60%, irrigated >60%)
  - Temporal trends (early period vs late period)
  - Crop type concentration (cereals vs pulses vs oilseeds share)

Outputs saved to: outputs/crop_diversity_analysis/
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "E:/CEEW Project"
INPUT_CROP = os.path.join(BASE_DIR, "outputs/all_crops_apy_1997_2021_india_data_portal.csv")
INPUT_IRR_IRRIGATED = os.path.join(BASE_DIR, "outputs/rainfed_irrigated_separate_maps_20251118_100436/irrigated_districts_list.csv")
INPUT_IRR_RAINFED = os.path.join(BASE_DIR, "outputs/rainfed_irrigated_separate_maps_20251118_100436/rainfed_districts_list.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/crop_diversity_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD AND CLEAN CROP DATA
# ============================================================
print("Loading crop data...")
df = pd.read_csv(INPUT_CROP)
df['year'] = df['year'].str.strip()
df['season'] = df['season'].str.strip()
df['state_name'] = df['state_name'].str.strip()
df['district_name'] = df['district_name'].str.strip()
df['crop_name'] = df['crop_name'].str.strip()
df['crop_type'] = df['crop_type'].str.strip()

# Drop rows with missing area
df = df.dropna(subset=['area'])
df = df[df['area'] > 0]

print(f"  Loaded {len(df):,} rows | {df.district_name.nunique()} districts | {df.crop_name.nunique()} crops | {df.year.nunique()} years")

# Extract start year for grouping
df['year_start'] = df['year'].apply(lambda x: int(x.split('-')[0]))

# Create a district key for consistent matching
df['district_key'] = df['state_name'].str.upper() + '|' + df['district_name'].str.upper()

# ============================================================
# 2. LOAD AND PROCESS IRRIGATION DATA
# ============================================================
print("Loading irrigation classification...")
irr_high = pd.read_csv(INPUT_IRR_IRRIGATED)
irr_low = pd.read_csv(INPUT_IRR_RAINFED)

# Combine and compute mean irrigation % per district
irr_all = pd.concat([irr_high, irr_low], ignore_index=True)
irr_all['State'] = irr_all['State'].str.strip().str.upper()
irr_all['District'] = irr_all['District'].str.strip().str.upper()
irr_avg = irr_all.groupby(['State', 'District'])['Gross_Irrigated_Area_%'].mean().reset_index()
irr_avg.rename(columns={'Gross_Irrigated_Area_%': 'irrigation_pct'}, inplace=True)
irr_avg['district_key'] = irr_avg['State'] + '|' + irr_avg['District']

# Classify irrigation regime
def classify_irrigation(pct):
    if pct < 40:
        return 'Rainfed (<40%)'
    elif pct < 60:
        return 'Semi-Irrigated (40-60%)'
    else:
        return 'Irrigated (>60%)'

irr_avg['irrigation_regime'] = irr_avg['irrigation_pct'].apply(classify_irrigation)
print(f"  Irrigation data: {len(irr_avg)} districts")
print(f"  Regime distribution:\n{irr_avg.irrigation_regime.value_counts().to_string()}")

# ============================================================
# 3. COMPUTE DIVERSITY INDICES
# ============================================================
print("\nComputing diversity indices...")

def compute_diversity(group):
    """Compute Shannon, Simpson, and richness for a group of crops."""
    total_area = group['area'].sum()
    if total_area <= 0:
        return pd.Series({
            'shannon_index': np.nan,
            'simpson_index': np.nan,
            'crop_richness': 0,
            'total_cropped_area': 0,
            'dominant_crop': '',
            'dominant_crop_share': np.nan,
            'top3_crops_share': np.nan,
        })

    # Proportions
    props = group.groupby('crop_name')['area'].sum() / total_area
    props = props[props > 0]

    # Shannon: H' = -Σ(pi * ln(pi))
    shannon = -np.sum(props * np.log(props))

    # Simpson: 1 - Σ(pi²)
    simpson = 1 - np.sum(props ** 2)

    # Richness
    richness = len(props)

    # Dominant crop info
    sorted_props = props.sort_values(ascending=False)
    dominant_crop = sorted_props.index[0]
    dominant_share = sorted_props.iloc[0]
    top3_share = sorted_props.iloc[:3].sum()

    return pd.Series({
        'shannon_index': round(shannon, 4),
        'simpson_index': round(simpson, 4),
        'crop_richness': richness,
        'total_cropped_area': round(total_area, 1),
        'dominant_crop': dominant_crop,
        'dominant_crop_share': round(dominant_share, 4),
        'top3_crops_share': round(top3_share, 4),
    })


# --- 3a. District-Year level indices (aggregate across seasons) ---
print("  Computing district-year level indices...")
district_year = df.groupby(['state_name', 'district_name', 'district_key', 'year', 'year_start']).apply(
    compute_diversity, include_groups=False
).reset_index()

print(f"  District-year observations: {len(district_year):,}")

# --- 3b. District-level averages (all years pooled) ---
print("  Computing district-level averages...")
district_avg = df.groupby(['state_name', 'district_name', 'district_key']).apply(
    compute_diversity, include_groups=False
).reset_index()

print(f"  Districts with indices: {len(district_avg)}")

# --- 3c. Period comparison (early vs late) ---
early_years = df[df['year_start'] <= 2005]
late_years = df[df['year_start'] >= 2015]

district_early = early_years.groupby(['state_name', 'district_name', 'district_key']).apply(
    compute_diversity, include_groups=False
).reset_index()
district_early.columns = [c if c in ['state_name', 'district_name', 'district_key'] else f'{c}_early' for c in district_early.columns]

district_late = late_years.groupby(['state_name', 'district_name', 'district_key']).apply(
    compute_diversity, include_groups=False
).reset_index()
district_late.columns = [c if c in ['state_name', 'district_name', 'district_key'] else f'{c}_late' for c in district_late.columns]

district_change = district_early.merge(district_late, on=['state_name', 'district_name', 'district_key'], how='inner')
district_change['shannon_change'] = district_change['shannon_index_late'] - district_change['shannon_index_early']
district_change['simpson_change'] = district_change['simpson_index_late'] - district_change['simpson_index_early']
district_change['richness_change'] = district_change['crop_richness_late'] - district_change['crop_richness_early']

print(f"  Districts with both periods: {len(district_change)}")

# ============================================================
# 4. COMPUTE AGRO-BIODIVERSITY INDEX (ABI)
# ============================================================
print("\nComputing Agro-Biodiversity Index...")

# Normalize each component to [0, 1] using min-max
for col in ['shannon_index', 'simpson_index', 'crop_richness']:
    min_val = district_avg[col].min()
    max_val = district_avg[col].max()
    district_avg[f'{col}_norm'] = (district_avg[col] - min_val) / (max_val - min_val)

# ABI = equal-weighted average of normalized components
district_avg['agro_biodiversity_index'] = round(
    (district_avg['shannon_index_norm'] + district_avg['simpson_index_norm'] + district_avg['crop_richness_norm']) / 3,
    4
)

# Classify ABI
district_avg['abi_category'] = pd.cut(
    district_avg['agro_biodiversity_index'],
    bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['Very Low', 'Low', 'Moderate', 'High'],
    include_lowest=True
)

# ============================================================
# 5. COMPUTE CROP TYPE CONCENTRATION
# ============================================================
print("Computing crop type concentration...")

crop_type_shares = df.groupby(['state_name', 'district_name', 'district_key', 'crop_type'])['area'].sum().reset_index()
total_by_district = crop_type_shares.groupby('district_key')['area'].sum().reset_index()
total_by_district.rename(columns={'area': 'total_area'}, inplace=True)
crop_type_shares = crop_type_shares.merge(total_by_district, on='district_key')
crop_type_shares['share'] = crop_type_shares['area'] / crop_type_shares['total_area']

# Pivot crop types
crop_type_pivot = crop_type_shares.pivot_table(
    index='district_key', columns='crop_type', values='share', fill_value=0
).reset_index()
crop_type_pivot.columns = ['district_key'] + [f'share_{c.lower().replace(" ", "_")}' for c in crop_type_pivot.columns[1:]]

# ============================================================
# 6. MERGE WITH IRRIGATION DATA
# ============================================================
print("Merging with irrigation classification...")

district_full = district_avg.merge(
    irr_avg[['district_key', 'irrigation_pct', 'irrigation_regime']],
    on='district_key', how='left'
)
district_full = district_full.merge(crop_type_pivot, on='district_key', how='left')

matched = district_full['irrigation_regime'].notna().sum()
print(f"  Matched with irrigation: {matched}/{len(district_full)} districts")

# ============================================================
# 7. ANALYSIS SUMMARIES
# ============================================================
print("\n" + "="*70)
print("CROP DIVERSITY ANALYSIS RESULTS")
print("="*70)

# Overall summary
print("\n--- Overall Diversity Statistics ---")
for col in ['shannon_index', 'simpson_index', 'crop_richness', 'agro_biodiversity_index']:
    print(f"  {col}: mean={district_avg[col].mean():.3f}, median={district_avg[col].median():.3f}, "
          f"std={district_avg[col].std():.3f}, min={district_avg[col].min():.3f}, max={district_avg[col].max():.3f}")

# ABI distribution
print(f"\n--- Agro-Biodiversity Index Distribution ---")
print(district_avg['abi_category'].value_counts().sort_index().to_string())

# By irrigation regime
print(f"\n--- Diversity by Irrigation Regime ---")
irr_matched = district_full[district_full['irrigation_regime'].notna()]
irr_summary = irr_matched.groupby('irrigation_regime').agg({
    'shannon_index': ['mean', 'median', 'std'],
    'simpson_index': ['mean', 'median'],
    'crop_richness': ['mean', 'median'],
    'agro_biodiversity_index': ['mean', 'median'],
    'dominant_crop_share': ['mean'],
    'district_key': 'count'
}).round(3)
print(irr_summary.to_string())

# Temporal change summary
print(f"\n--- Temporal Change (pre-2006 vs 2015+) ---")
print(f"  Shannon change: mean={district_change['shannon_change'].mean():.4f}, "
      f"median={district_change['shannon_change'].median():.4f}")
print(f"  Simpson change: mean={district_change['simpson_change'].mean():.4f}, "
      f"median={district_change['simpson_change'].median():.4f}")
print(f"  Richness change: mean={district_change['richness_change'].mean():.2f}, "
      f"median={district_change['richness_change'].median():.1f}")

# Districts gaining vs losing diversity
gaining = (district_change['shannon_change'] > 0).sum()
losing = (district_change['shannon_change'] < 0).sum()
stable = (district_change['shannon_change'] == 0).sum()
print(f"  Gaining diversity: {gaining} | Losing diversity: {losing} | Stable: {stable}")

# Top 10 most diverse
print(f"\n--- Top 15 Most Diverse Districts (by ABI) ---")
top = district_full.nlargest(15, 'agro_biodiversity_index')[
    ['state_name', 'district_name', 'shannon_index', 'simpson_index',
     'crop_richness', 'agro_biodiversity_index', 'irrigation_regime', 'dominant_crop']
]
print(top.to_string(index=False))

# Bottom 10 least diverse (monoculture)
print(f"\n--- Bottom 15 Least Diverse Districts (by ABI) ---")
bottom = district_full.nsmallest(15, 'agro_biodiversity_index')[
    ['state_name', 'district_name', 'shannon_index', 'simpson_index',
     'crop_richness', 'agro_biodiversity_index', 'irrigation_regime', 'dominant_crop']
]
print(bottom.to_string(index=False))

# Dominant crops by irrigation
print(f"\n--- Most Common Dominant Crop by Irrigation Regime ---")
for regime in ['Rainfed (<40%)', 'Semi-Irrigated (40-60%)', 'Irrigated (>60%)']:
    subset = irr_matched[irr_matched['irrigation_regime'] == regime]
    if len(subset) > 0:
        dom = subset['dominant_crop'].value_counts().head(5)
        print(f"\n  {regime} (n={len(subset)}):")
        for crop, count in dom.items():
            print(f"    {crop}: {count} districts ({count/len(subset)*100:.1f}%)")

# Crop type concentration by irrigation
print(f"\n--- Crop Type Shares by Irrigation Regime ---")
share_cols = [c for c in district_full.columns if c.startswith('share_')]
if share_cols:
    type_by_irr = irr_matched.groupby('irrigation_regime')[share_cols].mean().round(3)
    print(type_by_irr.T.to_string())

# State-level diversity ranking
print(f"\n--- State-Level Average Diversity (ranked by ABI) ---")
state_diversity = district_avg.groupby('state_name').agg({
    'shannon_index': 'mean',
    'simpson_index': 'mean',
    'crop_richness': 'mean',
    'agro_biodiversity_index': 'mean',
    'district_key': 'count'
}).round(3)
state_diversity.rename(columns={'district_key': 'n_districts'}, inplace=True)
state_diversity = state_diversity.sort_values('agro_biodiversity_index', ascending=False)
print(state_diversity.to_string())

# ============================================================
# 8. SAVE OUTPUTS
# ============================================================
print(f"\n{'='*70}")
print("Saving outputs...")

# Main district-level file
district_full.to_csv(os.path.join(OUTPUT_DIR, 'district_diversity_indices.csv'), index=False)
district_full.to_excel(os.path.join(OUTPUT_DIR, 'district_diversity_indices.xlsx'), index=False)
print(f"  Saved: district_diversity_indices.csv/xlsx ({len(district_full)} districts)")

# District-year panel
district_year.to_csv(os.path.join(OUTPUT_DIR, 'district_year_diversity_panel.csv'), index=False)
print(f"  Saved: district_year_diversity_panel.csv ({len(district_year)} obs)")

# Temporal change
district_change.to_csv(os.path.join(OUTPUT_DIR, 'district_diversity_change.csv'), index=False)
print(f"  Saved: district_diversity_change.csv ({len(district_change)} districts)")

# State summary
state_diversity.to_csv(os.path.join(OUTPUT_DIR, 'state_diversity_summary.csv'))
print(f"  Saved: state_diversity_summary.csv")

# Irrigation regime summary
if len(irr_matched) > 0:
    irr_summary_flat = irr_matched.groupby('irrigation_regime').agg({
        'shannon_index': ['mean', 'median', 'std', 'min', 'max'],
        'simpson_index': ['mean', 'median', 'std'],
        'crop_richness': ['mean', 'median', 'min', 'max'],
        'agro_biodiversity_index': ['mean', 'median', 'std'],
        'dominant_crop_share': ['mean', 'median'],
        'district_key': 'count'
    }).round(4)
    irr_summary_flat.to_csv(os.path.join(OUTPUT_DIR, 'irrigation_regime_diversity_summary.csv'))
    print(f"  Saved: irrigation_regime_diversity_summary.csv")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Done!")
