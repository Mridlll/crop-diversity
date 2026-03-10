# Handoff: Crop Diversity & Agro-Biodiversity Analysis

**Date:** 2026-03-09
**Status:** Analysis complete, interactive vizzes pending

## What Was Done

Computed district-level crop diversity indices for 755 districts across 54 crops over 24 years (1997-2021), cross-cut with irrigation classification.

### Indices Computed
- **Shannon Index** (H' = -Σpi·ln(pi)) — evenness + richness
- **Simpson Index** (1 - Σpi²) — probability two random hectares differ
- **Crop Richness** — count of distinct crops
- **Agro-Biodiversity Index (ABI)** — normalized composite of above three

### Script
`scripts/57_crop_diversity_agro_biodiversity.py`

### Outputs (all in `outputs/crop_diversity_analysis/`)
| File | Description | Rows |
|------|-------------|------|
| `district_diversity_indices.csv/xlsx` | Main file: all indices + irrigation regime + crop type shares | 755 districts |
| `district_year_diversity_panel.csv` | Yearly panel for trend analysis/viz | 14,136 obs |
| `district_diversity_change.csv` | Early (≤2005) vs Late (≥2015) period comparison | 590 districts |
| `state_diversity_summary.csv` | State-level rankings | 35 states |
| `irrigation_regime_diversity_summary.csv` | Summary stats by irrigation regime | 3 regimes |

### Key Input Files
- `outputs/all_crops_apy_1997_2021_india_data_portal.csv` — 345K rows, district-crop-year-season area/production/yield
- `outputs/rainfed_irrigated_separate_maps_20251118_100436/irrigated_districts_list.csv` — irrigated districts
- `outputs/rainfed_irrigated_separate_maps_20251118_100436/rainfed_districts_list.csv` — rainfed districts

### Irrigation Classification
- **Rainfed:** <40% gross irrigated area (277 districts)
- **Semi-Irrigated:** 40-60% (113 districts)
- **Irrigated:** >60% (193 districts)
- 503/755 districts matched with irrigation data

## Key Findings

1. **Semi-irrigated districts are MOST diverse** (ABI 0.69) — more than both rainfed (0.62) and irrigated (0.67)
2. **Diversity is declining:** 357 districts losing vs 233 gaining; median -0.05 Shannon, -2 crops per district over 20 years
3. **Irrigation drives monoculture:** Irrigated → 53% wheat-dominant; Rainfed → 60% rice-dominant; Semi-irrigated → mixed
4. **Irrigated = 67% cereals, 8.5% pulses; Semi-irrigated = 53% cereals, 23% oilseeds, 13% pulses**
5. **Karnataka most diverse** (ABI 0.85), **Punjab least** among major states (0.44)

## Data Quality Notes
- Bottom-ranked districts include UT mismatches (Delhi, Chandigarh showing in wrong state codes) — data artifacts
- Irrigation match rate is 503/755 (67%) — remaining 252 districts lack irrigation classification
- Some district names don't match between crop data (Title Case) and irrigation data (UPPER CASE) — fuzzy matching could improve coverage

## What's Next

1. **Interactive visualizations** — choropleth maps of diversity indices, time-series animations, irrigation scatter plots
2. Possible improvements:
   - Fuzzy district name matching to improve irrigation coverage from 67% to ~90%
   - Separate Kharif vs Rabi diversity analysis
   - Herfindahl-Hirschman Index as additional concentration metric
   - Margalef richness index (accounts for sample size)
3. Paper structure was discussed but deferred — focus on vizzes first
