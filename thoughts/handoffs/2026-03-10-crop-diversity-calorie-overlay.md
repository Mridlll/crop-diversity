# Handoff: Crop Diversity & Calorie-Diversity Overlay Analysis

**Date:** 2026-03-10
**Status:** Core analysis complete, hosted on GitHub Pages

## What Was Done

### 1. Crop Diversity Indices (from previous session, refined this session)
- Computed Shannon, Simpson, Richness, ABI for **725 districts** (cleaned from 755 — removed 7 bogus + 23 state-split duplicates)
- 54 crops, 24 agricultural years (1997-2021)
- District matching: **96.6% coverage** (710/735 shapefile districts) via exact + 96 manual + fuzzy matching
- Irrigation classification: Rainfed (<40%), Semi-Irrigated (40-60%), Irrigated (>60%) — 503 districts

### 2. District-Level Calorie Production (NEW this session)
- Computed kcal/ha for all 725 districts using crop production × IFCT 2017 conversion factors
- Non-food crops (cotton, jute, tobacco) assigned 0 kcal
- Quadrant classification: Diverse & Calorie-Rich (23.6%), Monoculture Breadbasket (26.3%), Diverse & Calorie-Poor (26.2%), Vulnerable (23.9%)
- **Known issue:** kcal/ha is bimodal — P75=13k but P90=837k due to coconut effect (Kerala/TN). 40 districts with <10k ha flagged as unreliable.

### 3. GitHub Pages Site
Live at: **https://mridlll.github.io/crop-diversity/**
- Landing page with 4 tools + index explainer
- Crop diversity hover map (Folium)
- Calorie-diversity hover map (7 dropdown indices)
- Static maps gallery (14 maps with lightbox)
- ABI timeline (1997-2020 year slider with animation)

### 4. GitHub Repo
**https://github.com/Mridlll/crop-diversity** (public)

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/57_crop_diversity_agro_biodiversity.py` | Core index computation (cleaned source data) |
| `scripts/58_crop_diversity_dashboard.py` | Plotly Dash dashboard (5 tabs, laggy with mapbox) |
| `scripts/59_crop_diversity_static_maps.py` | 9 static maps + 3 GIFs, `--only` flag for selective regen |
| `scripts/60_crop_diversity_hover_map.py` | Folium diversity hover map |
| `scripts/61_generate_notebook.py` | Generates Jupyter notebook (91 cells, 11 sections) |
| `scripts/62_district_calorie_production.py` | Kcal/ha computation + quadrant maps + scatter |
| `scripts/63_calorie_diversity_hover_map.py` | Folium calorie-diversity hover map |
| `scripts/64_generate_timeline_data.py` | Generates GeoJSON for timeline page |

## Key Outputs
| File | Location |
|------|----------|
| Main diversity CSV | `outputs/crop_diversity_analysis/district_diversity_indices.csv` (725 districts) |
| Calorie-merged CSV | `outputs/crop_diversity_analysis/district_diversity_calorie_merged.csv` |
| Yearly panel | `outputs/crop_diversity_analysis/district_year_diversity_panel.csv` (14,136 obs) |
| Change analysis | `outputs/crop_diversity_analysis/district_diversity_change.csv` |
| Static maps (PNG) | `outputs/crop_diversity_analysis/maps/` (9 maps) |
| Calorie maps (PNG) | `outputs/crop_diversity_analysis/` (quadrant, kcal choropleth, bivariate, scatter, hollow) |
| GIFs | `outputs/crop_diversity_analysis/maps/` (Shannon, irrigation×diversity, richness timelines) |
| Hover maps (HTML) | `outputs/crop_diversity_analysis/` (diversity + calorie-diversity) |
| Jupyter notebook | `notebooks/crop_diversity_analysis.ipynb` |
| GitHub Pages | `docs/` (index, diversity, calorie-diversity, gallery, timeline) |

## Key Findings
1. Semi-irrigated districts most diverse (ABI 0.69) > irrigated (0.67) > rainfed (0.62)
2. Diversity declining: 357 losing vs 233 gaining districts
3. ABI vs kcal/ha correlation: Pearson -0.18, Spearman -0.06 (weak) — tradeoff is overstated
4. Quadrants nearly even (~25% each) — no dominant pattern
5. Coconut effect dominates kcal/ha top tail (Kerala/TN)
6. Karnataka most diverse (0.85), Punjab least (0.44)

## Data Quality Notes
- Source data had 7 bogus district-state pairs (Chandigarh in Delhi/Goa/DNH) — removed
- 23 state-split duplicates (Uttarakhand/Chhattisgarh/Jharkhand under old state names) — remapped
- 25 unmatched shapefile districts are genuinely absent (Delhi sub-districts, new NE districts, PoK)
- Calorie data: IFCT 2017 factors cover all 54 crops; non-food = 0 kcal
- 40 tiny districts (<10k ha) flagged as unreliable for kcal/ha

## What's Next
1. **Recompute correlations excluding coconut districts** — test if the weak tradeoff holds without the coconut outlier effect
2. **Kharif vs Rabi diversity** — separate seasonal analysis (data has season column)
3. **Herfindahl-Hirschman Index** — additional concentration metric
4. **Per-capita calorie analysis** — need district population data to go from kcal/ha to kcal/person/day
5. **Paper draft** — structure was discussed but deferred
6. **Dashboard performance** — current Dash dashboard is laggy with mapbox; could switch to local shapefiles like the static maps
7. **Fuzzy matching for static maps** — script 59 still at 613/735; needs the MANUAL_MAP from script 60
