# Crop Diversity & Agro-Biodiversity in Indian Agriculture (1997-2021)

District-level analysis of crop diversification patterns across 725 Indian districts, 54 crops, and 24 agricultural years — examining the relationship between irrigation infrastructure and agricultural biodiversity.

## Interactive Maps

**[Crop Diversity Map](https://mridlll.github.io/crop-diversity/diversity.html)** — hover over any district to see diversity indices, irrigation status, dominant crops, and crop composition.

**[Calorie-Diversity Map](https://mridlll.github.io/crop-diversity/calorie-diversity.html)** — hover over any district to see diversity indices alongside caloric productivity and quadrant classification.

**[Diversity Timeline](https://mridlll.github.io/crop-diversity/timeline.html)** — animated year-by-year diversity timeline (1997-2020).

**[Food vs Non-Food Map](https://mridlll.github.io/crop-diversity/food-nonfood.html)** — food vs non-food crop breakdown hover map.

## Key Findings

1. **Semi-irrigated districts are the most diverse** (ABI 0.69) — more than both rainfed (0.62) and fully irrigated (0.67) districts
2. **Diversity is declining nationally**: 357 districts losing diversity vs 233 gaining; median Shannon index declined by 0.05 over two decades
3. **Irrigation drives monoculture**: irrigated districts are 53% wheat-dominant; rainfed are 60% rice-dominant; semi-irrigated maintain mixed cropping
4. **Karnataka is the most diverse state** (ABI 0.85), **Punjab the least** among major states (0.44)
5. **Irrigated districts allocate 67% of area to cereals** vs 53% in semi-irrigated, with pulses and oilseeds crowded out

## Indices

| Index | What It Measures | Range |
|-------|-----------------|-------|
| **Shannon (H')** | Information entropy — sensitivity to rare crops | 0 to ~2.5 |
| **Simpson (1-D)** | Probability two random hectares differ in crop | 0 to 1 |
| **Crop Richness (S)** | Count of distinct crops grown | 1 to 50+ |
| **Agro-Biodiversity Index (ABI)** | Normalized composite of above three | 0 to 1 |

ABI is computed as the equal-weighted mean of min-max normalized Shannon, Simpson, and Richness indices. Full methodology with worked examples and robustness checks is in the [Jupyter notebook](notebooks/crop_diversity_analysis.ipynb).

## Calorie-Diversity Overlay

**[Explore the calorie-diversity interactive map](https://mridlll.github.io/crop-diversity/calorie-diversity.html)** — hover over any district to see diversity indices alongside caloric productivity, with quadrant classification.

District-level calorie production is computed by multiplying crop production (tonnes) by IFCT 2017 energy conversion factors (kcal per 100g), yielding total kcal produced per hectare of gross cropped area. This caloric productivity metric is then overlaid with the Agro-Biodiversity Index to classify districts into four quadrants:

| Quadrant | ABI | Kcal/ha | Description |
|----------|-----|---------|-------------|
| **Diverse & Calorie-Rich** | High | High | Best of both worlds — high biodiversity with strong caloric output |
| **Monoculture Breadbasket** | Low | High | High caloric productivity driven by specialization in few crops |
| **Diverse & Calorie-Poor** | High | Low | Biodiverse cropping but low caloric output per hectare |
| **Vulnerable** | Low | Low | Low diversity and low caloric productivity — most at-risk districts |

**Quadrant distribution** (725 districts):
- Diverse & Calorie-Rich: 23.6%
- Monoculture Breadbasket: 26.3%
- Diverse & Calorie-Poor: 26.2%
- Vulnerable: 23.9%

**Key insight:** The correlation between diversity (ABI) and caloric productivity (kcal/ha) is weakly negative — the assumed tradeoff between biodiversity and food production is overstated. Many districts achieve both high diversity and high caloric output, suggesting that crop diversification need not come at the cost of food security.

### Coconut Exclusion Analysis

154 districts are flagged as coconut-dominant (>50% of kcal derived from coconut). Coconut's extreme caloric density distorts the kcal/ha distribution (range: 752 to 75.5M kcal/ha), so log-scale visualization is used for choropleth maps. When these 154 districts are excluded, the Pearson correlation between ABI and kcal/ha shifts from **-0.18 to +0.02**, effectively eliminating the apparent diversity-productivity tradeoff. Ex-coconut maps and scatter plots are provided as separate outputs (suffixed `_ex_coconut`).

## Repository Structure

```
scripts/
  57_crop_diversity_agro_biodiversity.py   # Core computation of all indices
  58_crop_diversity_dashboard.py           # Plotly Dash interactive dashboard
  59_crop_diversity_static_maps.py         # Publication-quality static maps + GIFs
  60_crop_diversity_hover_map.py           # Folium interactive hover map
  61_generate_notebook.py                  # Generates the analysis notebook
  62_district_calorie_production.py        # Kcal/ha computation using IFCT 2017 factors
  63_calorie_diversity_hover_map.py        # Calorie-diversity quadrant interactive map
  64_generate_timeline_data.py             # Generates GeoJSON for animated timeline page
  65_food_nonfood_hover_map.py             # Food vs non-food crop breakdown hover map

notebooks/
  crop_diversity_analysis.ipynb            # Full analysis with methodology & results

outputs/crop_diversity_analysis/
  district_diversity_indices.csv           # Main dataset: 725 districts, all indices
  district_diversity_calorie_merged.csv    # Diversity + caloric productivity merged
  district_year_diversity_panel.csv        # Yearly panel (14,136 observations)
  district_diversity_change.csv            # Early vs Late period comparison
  state_diversity_summary.csv              # State-level rankings
  crop_diversity_hover_map.html            # Self-contained interactive map
  calorie_diversity_hover_map.html         # Calorie-diversity quadrant map
  kcal_per_hectare_choropleth.png          # Kcal/ha choropleth
  quadrant_map.png                         # Four-quadrant classification map
  bivariate_abi_kcal_map.png              # Bivariate ABI × kcal map
  abi_vs_kcal_scatter.png                  # ABI vs kcal/ha scatter plot
  kcal_per_hectare_choropleth_ex_coconut.png  # Ex-coconut linear choropleth (P2-P98 winsorized)
  quadrant_map_ex_coconut.png              # Quadrant map with coconut-excluded thresholds
  bivariate_abi_kcal_map_ex_coconut.png    # Bivariate map with ex-coconut terciles
  abi_vs_kcal_scatter_ex_coconut.png       # Scatter plot with coconut districts as hollow diamonds
  food_nonfood_hover_map.html              # Food vs non-food interactive map
  maps/                                    # 9 static maps (PNG)
```

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| **[ISB India Data Portal](https://www.india-data-portal.org/)** | District-level area, production, and yield for 54 crops | 725 districts, 1997-2021 |
| **Government of India (scraped)** | District-level gross irrigated area as % of gross cropped area | 503 districts classified |
| **IFCT 2017 (NIN, Hyderabad)** | Energy conversion factors (kcal per 100g) for Indian food items | 54 crops mapped |
| **Census of India 2011** | District boundary shapefiles | 735 districts |

### Crop Production Data

Area, production, and yield data for all major crops at the district-year-season level was sourced from the **Indian School of Business (ISB) India Data Portal**, which aggregates data originally published by the Ministry of Agriculture & Farmers Welfare, Government of India. The dataset covers 54 crops across Kharif, Rabi, and whole-year seasons for agricultural years 1997-98 through 2020-21.

### Irrigation Data

Irrigation statistics were **scraped from multiple Government of India sources** including the Minor Irrigation Census, agricultural census publications, and state-level irrigation department data. No single consolidated dataset existed at the district level — the data was assembled through web scraping and manual compilation from official government portals. Districts were classified into three regimes:

- **Rainfed** (<40% gross irrigated area): 240 districts
- **Semi-Irrigated** (40-60%): 101 districts
- **Irrigated** (>60%): 162 districts

### Shapefiles

Census 2011 district boundaries (735 districts). District name matching between data sources achieved **96.6% coverage** (710/735 districts) through normalized matching + 96 manual corrections for spelling, transliteration, and administrative reorganization differences (e.g., Telangana bifurcation, district renaming like Prayagraj/Allahabad, Gurugram/Gurgaon).

## Static Maps

Nine publication-quality maps are included in `outputs/crop_diversity_analysis/maps/`:

| Map | Description |
|-----|-------------|
| Shannon Index | District-level Shannon diversity (H') |
| Simpson Index | District-level Simpson diversity (1-D) |
| Crop Richness | Number of distinct crops per district |
| ABI | Agro-Biodiversity Index composite |
| Irrigation Regime | Rainfed / Semi-Irrigated / Irrigated classification |
| Diversity Change | Early (pre-2005) vs Late (post-2015) Shannon change |
| Combined Panel (2x2) | All four indices side-by-side |
| Irrigation x Diversity (1x3) | ABI by irrigation regime |
| Top/Bottom 20 | Most and least diverse districts |

## Reproducing the Analysis

```bash
# 1. Compute diversity indices (requires raw crop data + shapefiles)
python scripts/57_crop_diversity_agro_biodiversity.py

# 2. Generate static maps and animated GIFs
python scripts/59_crop_diversity_static_maps.py

# 3. Generate interactive hover map
python scripts/60_crop_diversity_hover_map.py

# 4. Generate Jupyter notebook
python scripts/61_generate_notebook.py

# 5. Compute district-level calorie production (requires IFCT 2017 factors)
python scripts/62_district_calorie_production.py

# 6. Generate calorie-diversity interactive map
python scripts/63_calorie_diversity_hover_map.py

# 7. Generate GeoJSON data for animated timeline page
python scripts/64_generate_timeline_data.py

# 8. Generate food vs non-food crop breakdown hover map
python scripts/65_food_nonfood_hover_map.py

# 9. Launch interactive dashboard (opens at http://127.0.0.1:8050)
python scripts/58_crop_diversity_dashboard.py
```

### Requirements

```
pandas, geopandas, numpy, matplotlib, plotly, dash, dash-bootstrap-components, folium, imageio, pillow
```

## References

1. Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423.
2. Simpson, E.H. (1949). "Measurement of Diversity." *Nature*, 163, 688.
3. Di Falco, S. & Chavas, J.P. (2009). "On Crop Biodiversity, Risk Exposure, and Food Security in the Highlands of Ethiopia." *American Journal of Agricultural Economics*, 91(3), 599-611.
4. Lin, B.B. (2011). "Resilience in Agriculture through Crop Diversification." *BioScience*, 61(3), 183-193.
5. Birthal, P.S. et al. (2015). "Crop Diversification and Resilience of Agriculture to Climatic Shocks." *Indian Journal of Agricultural Economics*, 70(1).

## License

This work is independent research by Mridul.
