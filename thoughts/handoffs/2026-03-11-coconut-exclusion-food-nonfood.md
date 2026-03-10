# Handoff: Coconut Exclusion, Food/Non-Food, Timeline Filters

**Date:** 2026-03-11
**Status:** All pushed to GitHub, Pages live
**Repo:** https://github.com/Mridlll/crop-diversity
**Pages:** https://mridlll.github.io/crop-diversity/

## What Was Done (9 commits: 1a8410a → 9d2bf35)

### 1. Coconut-Excluded Analysis
- Flagged **154 coconut-dominant districts** (>50% of kcal from coconut) in script 62
- Added `coconut_dominant`, `coconut_kcal_share`, `kcal_diversity_quadrant_ex_coconut` columns to merged CSV
- Recomputed quadrants using ex-coconut medians (ABI=0.66, kcal/ha=5,799 vs full: 6,944)
- Generated 4 ex-coconut static maps: quadrant (hatched), bivariate (hatched), scatter (hollow diamonds), choropleth (P2-P98 winsorized)
- **Key finding:** Pearson correlation shifts from -0.18 to +0.02 — the negative correlation was entirely a coconut artifact. Spearman goes from -0.06 to -0.14 (p=0.001) — a real weak negative emerges.

### 2. Kcal/ha Scale Fix
- Log-scale choropleth for full dataset (range 752 → 75.5M was killing all granularity)
- Log-spaced color bins in the Folium hover map's kcal/ha layer
- K/M unit formatting everywhere: axes, legends, tooltips (e.g., "75.5M" not "75,515,179")
- Ex-coconut choropleth with P2-P98 winsorization (2K-16K range) for clean linear gradients

### 3. Food vs Non-Food Hover Map (Script 65)
- 7 display layers: Food Crop Area Share, Cereal, Pulse, Oilseed, Cash Crop, Sugar, Fruit & Veg
- Inline bar charts in tooltips showing full food/non-food breakdown
- Coconut-dominant badge in tooltip
- Added as 5th card on GitHub Pages index

### 4. Timeline Gainer/Loser Filter
- Computed Shannon change (last year minus first, requiring ≥5 years of data)
- Classified: 101 Top Gainers (>+0.61), 101 Top Losers (<-0.27), 468 Stable, 55 Insufficient Data
- Filter dropdown on timeline page — non-matching districts grey out
- Tooltip shows change value and classification

### 5. Calorie-Diversity Hover Map Updates
- Coconut mode dropdown: "Including Coconut" / "Excluding Coconut" for quadrant layer
- Coconut badge in tooltips (brown badge with share percentage)

### 6. Documentation Updates
- README: fixed 755→725 district count, added scripts 64-65, new outputs, coconut analysis notes
- Notebook (script 61): added sections 9.7 (coconut-excluded sensitivity) and 9.8 (food/non-food breakdown), now 100 cells
- Gallery: added 5 new ex-coconut map cards + ex-coconut choropleth
- Back-to-home links added to all 5 Pages pages

## Scripts Modified/Created
| Script | Change |
|--------|--------|
| `scripts/62_district_calorie_production.py` | Coconut flagging, ex-coconut quadrants, log-scale + ex-coconut choropleths, K/M formatting |
| `scripts/63_calorie_diversity_hover_map.py` | Coconut mode dropdown, badge, log-spaced kcal bins, K/M formatting |
| `scripts/64_generate_timeline_data.py` | Shannon change computation, gainer/loser classification |
| `scripts/65_food_nonfood_hover_map.py` | **NEW** — food vs non-food hover map |
| `scripts/61_generate_notebook.py` | Added sections 9.7 + 9.8 |

## Key Outputs
| File | Location |
|------|----------|
| Merged CSV (updated) | `outputs/crop_diversity_analysis/district_diversity_calorie_merged.csv` |
| Ex-coconut static maps | `outputs/crop_diversity_analysis/*_ex_coconut.png` (4 files) |
| Food/non-food hover map | `outputs/crop_diversity_analysis/food_nonfood_hover_map.html` |
| Timeline GeoJSON (updated) | `docs/data/district_timeline.geojson` |
| Notebook (updated) | `notebooks/crop_diversity_analysis.ipynb` (100 cells) |
| GitHub Pages | `docs/` (5 pages: diversity, calorie-diversity, gallery, timeline, food-nonfood) |

## Data Notes
- 154 coconut-dominant districts is a lot — threshold is >50% of kcal from coconut. Many are not extreme outliers (e.g., Nadia WB at 9K kcal/ha, 53% coconut). Could tighten to require both >50% coconut AND >100K kcal/ha if the current threshold seems too broad.
- Bottom-tail districts (Botad 752, Barmer 813 kcal/ha) are likely data quality issues, not real. The P2-P98 winsorization handles this for the ex-coconut choropleth.
- Regenerating hover maps (scripts 60, 63, 65) overwrites the back-to-home links in docs/ — need to re-add after regeneration.

## What's Next
1. **Kharif vs Rabi diversity** — separate seasonal analysis (data has season column)
2. **Herfindahl-Hirschman Index** — additional concentration metric
3. **Per-capita calorie analysis** — need district population data for kcal/person/day
4. **Paper draft** — structure discussed but deferred
5. **Tighten coconut threshold** — consider dual condition (>50% share AND >100K kcal/ha) to avoid flagging low-kcal districts
6. **Persist back links in scripts** — modify scripts 60/63/65 to inject the home link into generated HTML so it survives regeneration
7. **Dashboard performance** — Dash dashboard still laggy with mapbox; could switch to local shapefiles
