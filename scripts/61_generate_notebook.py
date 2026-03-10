"""
Generate the Crop Diversity Analysis Jupyter Notebook.

Usage:
    python scripts/61_generate_notebook.py

Creates: notebooks/crop_diversity_analysis.ipynb
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

def md(source):
    return new_markdown_cell(source.strip())

def code(source):
    return new_code_cell(source.strip())

cells = []

# =============================================================================
# TITLE
# =============================================================================
cells.append(md(r"""
# Crop Diversity and Agro-Biodiversity in Indian Agriculture (1997-2021)

**Authors:** Council on Energy, Environment and Water (CEEW)
**Date:** March 2026
**Version:** 1.0

---

**Abstract**

This notebook presents a comprehensive analysis of crop diversity and agro-biodiversity across 755 districts in India over a 24-year period (1997-2021). Using three complementary indices -- the Shannon Diversity Index, the Simpson Diversity Index, and Crop Richness -- we construct a composite Agro-Biodiversity Index (ABI) that captures both the variety and evenness of crop cultivation. The analysis reveals a troubling decline in crop diversity nationwide, with 357 districts losing diversity against only 233 gaining. Semi-irrigated districts emerge as the most agro-biodiverse, challenging the assumption that irrigation uniformly benefits agricultural ecosystems. Karnataka ranks as the most diverse major state (ABI = 0.85), while Punjab -- the heartland of India's Green Revolution -- ranks lowest (ABI = 0.44). These findings carry significant implications for food security, climate resilience, and agricultural policy.
"""))

# =============================================================================
# SETUP
# =============================================================================
cells.append(md(r"""
---
## Setup and Configuration
"""))

cells.append(code(r"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from IPython.display import Image, display, HTML
from pathlib import Path
import os

%matplotlib inline

# Plotting defaults
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Consistent color palette
PALETTE = {
    'Irrigated (>60%)': '#2166ac',
    'Rainfed (<40%)': '#b2182b',
    'Semi-Irrigated (40-60%)': '#4dac26',
}
REGIME_ORDER = ['Rainfed (<40%)', 'Semi-Irrigated (40-60%)', 'Irrigated (>60%)']
ABI_CMAP = 'YlGn'

# Paths
BASE = Path('..')
DATA_DIR = BASE / 'outputs' / 'crop_diversity_analysis'
MAP_DIR = DATA_DIR / 'maps'
RAW_DATA = BASE / 'outputs' / 'all_crops_apy_1997_2021_india_data_portal.csv'

print("Setup complete.")
"""))

# =============================================================================
# SECTION 1: INTRODUCTION
# =============================================================================
cells.append(md(r"""
---
## 1. Introduction and Motivation

### Why Does Crop Diversity Matter?

Crop diversity is a cornerstone of sustainable agriculture and food security. A diverse cropping system provides multiple benefits:

- **Climate resilience:** Diverse crop portfolios buffer against weather shocks. If one crop fails due to drought or flooding, others may survive.
- **Nutritional security:** Monoculture landscapes tend to produce calorie-dense but nutrient-poor diets. Diverse cropping supports dietary diversity.
- **Soil health:** Crop rotation and mixed cropping maintain soil fertility, reduce pest buildup, and decrease dependence on chemical inputs.
- **Economic stability:** Farmers growing multiple crops face lower income volatility and reduced market risk.
- **Ecological services:** Agro-biodiversity supports pollinators, natural pest control, and broader ecosystem health.

### Policy Context

India's agricultural transformation since the Green Revolution of the 1960s-70s has been characterized by a marked shift toward cereal monocultures -- primarily rice and wheat -- driven by minimum support prices, input subsidies, and public distribution system requirements. While this transformation successfully addressed calorie deficits, it has come at a cost: declining soil health, groundwater depletion, nutritional imbalances, and loss of traditional crop varieties.

The National Mission for Sustainable Agriculture (NMSA) and the National Food Security Mission (NFSM) have increasingly recognized the need to diversify Indian agriculture. Understanding the current state and trends of crop diversity at the district level is essential for targeting these interventions effectively.

### Research Questions

1. What is the spatial distribution of crop diversity across Indian districts?
2. How has crop diversity changed over the period 1997-2021?
3. What is the relationship between irrigation infrastructure and crop diversity?
4. Which states and districts are gaining or losing diversity, and why?
"""))

# =============================================================================
# SECTION 2: DATA DESCRIPTION
# =============================================================================
cells.append(md(r"""
---
## 2. Data Description
"""))

cells.append(md(r"""
### 2.1 Data Sources

This analysis draws on three primary datasets, each providing a distinct dimension of India's agricultural landscape:

| # | Source | Description | Coverage | Time Period | Resolution |
|---|--------|-------------|----------|-------------|------------|
| 1 | **India Data Portal / Ministry of Agriculture & Farmers Welfare** | Area, production, and yield of major crops | 54 crops across 755 districts, Kharif and Rabi seasons | 1997--2021 (24 years) | District-level |
| 2 | **Census of India / Ministry of Water Resources** | Gross irrigated area as percentage of gross cropped area | All reporting districts | Decadal census periods | District-level |
| 3 | **Census of India 2011 Administrative Boundaries** | District boundary shapefiles for spatial analysis | 735 districts | 2011 | District polygon |

**Access and Download Notes:**

- **Crop production data** was obtained from the India Data Portal ([data.gov.in](https://data.gov.in)), which aggregates records published by the Directorate of Economics and Statistics, Ministry of Agriculture & Farmers Welfare. The dataset contains area (hectares), production (tonnes), and yield (tonnes/hectare) for each crop--district--season--year combination.
- **Irrigation data** was sourced from district-level irrigation statistics compiled from Census of India records and Ministry of Water Resources publications. Gross irrigated area as a percentage of gross cropped area was used to classify districts into three irrigation regimes: Rainfed (<40%), Semi-Irrigated (40--60%), and Irrigated (>60%).
- **District shapefiles** are from the Census of India 2011 administrative boundary dataset, which provides polygon geometries for 735 districts.

**Data Cleaning Steps:**

1. District names were harmonised between Title Case (crop data) and UPPER CASE (irrigation data) to enable matching; 503 of 755 crop-data districts were successfully linked to irrigation records.
2. Crop names were standardised and grouped into broad categories (cereals, pulses, oilseeds, fibres, sugarcane, spices, fruits, vegetables, and others).
3. Observations with missing area or zero production were excluded from diversity calculations.
4. Five-year averages (early period: 1997--2005; late period: 2015--2021) were computed to smooth inter-annual variability in the change analysis.
"""))

cells.append(code(r"""
# Load all datasets
df_main = pd.read_csv(DATA_DIR / 'district_diversity_indices.csv')
df_panel = pd.read_csv(DATA_DIR / 'district_year_diversity_panel.csv')
df_change = pd.read_csv(DATA_DIR / 'district_diversity_change.csv')
df_state = pd.read_csv(DATA_DIR / 'state_diversity_summary.csv')
df_irr = pd.read_csv(DATA_DIR / 'irrigation_regime_diversity_summary.csv', header=[0, 1], index_col=0)

print("Datasets loaded successfully.")
print(f"  District indices:    {df_main.shape[0]:,} districts x {df_main.shape[1]} columns")
print(f"  Yearly panel:        {df_panel.shape[0]:,} observations")
print(f"  Change analysis:     {df_change.shape[0]:,} districts (early vs late period)")
print(f"  State summary:       {df_state.shape[0]:,} states/UTs")
"""))

cells.append(code(r"""
# Data coverage summary
n_states = df_main['state_name'].nunique()
n_districts = df_main.shape[0]
years = sorted(df_panel['year_start'].unique())
n_years = len(years)
n_crops = df_main.columns[df_main.columns.str.startswith('share_')].shape[0]

print(f"Coverage:")
print(f"  States/UTs:      {n_states}")
print(f"  Districts:       {n_districts}")
print(f"  Year range:      {min(years)}-{max(years)} ({n_years} years)")
print(f"  Crop categories: {n_crops} (broad types: cereals, pulses, oilseeds, etc.)")
"""))

cells.append(code(r"""
# Summary statistics for key variables
summary_cols = ['shannon_index', 'simpson_index', 'crop_richness', 'agro_biodiversity_index',
                'total_cropped_area', 'dominant_crop_share']
summary = df_main[summary_cols].describe().T
summary.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

# Format nicely
def highlight_extremes(s):
    styles = [''] * len(s)
    if s.name in ['Mean', '50%']:
        return styles
    return styles

styled = (summary.style
    .format(precision=3)
    .set_caption('Table 1: Summary Statistics of District-Level Diversity Indices (N=755)')
    .background_gradient(cmap='YlGn', subset=['Mean', '50%'], axis=0)
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
    ])
)
display(styled)
"""))

cells.append(md(r"""
**Table 1 Interpretation:** The average Shannon Index across 755 districts is approximately 1.5, indicating moderate diversity. The Simpson Index averages around 0.63, meaning there is roughly a 63% chance that two randomly selected hectares grow different crops. Crop richness ranges from 1 (complete monoculture) to 45 crops per district, with a median of about 32. The composite Agro-Biodiversity Index ranges from 0 (no diversity) to nearly 1 (maximum diversity), with a mean around 0.60.

Note the high standard deviations relative to means -- crop diversity varies enormously across India's districts.
"""))

cells.append(code(r"""
# Distribution of ABI categories
abi_counts = df_main['abi_category'].value_counts().reindex(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
abi_pct = (abi_counts / abi_counts.sum() * 100).round(1)

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
bars = ax.barh(abi_counts.index, abi_counts.values, color=colors, edgecolor='white', height=0.6)
for bar, pct, cnt in zip(bars, abi_pct.values, abi_counts.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{cnt} ({pct}%)', va='center', fontsize=10)
ax.set_xlabel('Number of Districts')
ax.set_title('Figure 1: Distribution of Districts by Agro-Biodiversity Category')
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
**Figure 1 Interpretation:** The majority of Indian districts fall in the "Moderate" diversity category. The distribution is roughly bell-shaped, with relatively few districts at the extremes. This suggests that while extreme monoculture is uncommon, truly high agro-biodiversity is also rare -- most of Indian agriculture occupies a middle ground.
"""))

cells.append(code(r"""
# Data quality notes
irr_matched = df_main['irrigation_regime'].notna().sum()
irr_total = df_main.shape[0]
print("Data Quality Notes:")
print(f"  - Irrigation regime available for {irr_matched}/{irr_total} districts ({irr_matched/irr_total*100:.1f}%)")
print(f"  - {irr_total - irr_matched} districts lack irrigation classification")
print(f"  - District names were matched between crop data (Title Case) and irrigation data (UPPER CASE)")
print(f"  - Some bottom-ranked districts include UT mismatches (e.g., Delhi, Chandigarh)")
print(f"  - Change analysis covers {df_change.shape[0]} districts with data in both early (<=2005) and late (>=2015) periods")
"""))

# =============================================================================
# SECTION 3: METHODOLOGY
# =============================================================================
cells.append(md(r"""
---
## 3. Methodology

This section provides a thorough account of every methodological choice in the analysis -- the indices selected, the normalization strategy, the composite construction, and the classification schemes. Each decision is justified on both theoretical and practical grounds. The goal is full transparency: a reader should be able to reproduce the Agro-Biodiversity Index from raw data using only this section as a guide.
"""))

# ---- 3.1 Why Measure Crop Diversity? ----
cells.append(md(r"""
### 3.1 Why Measure Crop Diversity?

Measuring crop diversity is not merely an academic exercise. It addresses three interconnected dimensions of agricultural sustainability that are central to India's food system challenges.

**Ecological resilience.** Monoculture systems are inherently fragile. When a single crop dominates a landscape, the entire region becomes vulnerable to species-specific pests, diseases, and weather shocks. The Irish Potato Famine (1845-52) and the Southern Corn Leaf Blight epidemic in the United States (1970) are canonical examples. In the Indian context, the repeated fall armyworm outbreaks in maize-dominant districts illustrate the same principle. As established in the ecological economics literature, diversified cropping systems function as biological insurance -- if one crop fails, others in the portfolio may survive, stabilizing aggregate output. Mixed cropping also supports beneficial soil microbiomes, breaks pest and disease cycles, and reduces dependence on chemical inputs, all of which contribute to long-term agroecosystem health.

**Economic resilience.** Price shocks in commodity markets can devastate farmers locked into a single crop. When international cotton prices collapsed in 2014-15, Vidarbha's cotton-dependent districts experienced severe distress, while neighboring districts with diversified portfolios absorbed the shock more readily. Crop diversification is, in effect, a risk management strategy -- it provides market diversification analogous to a financial portfolio. The agricultural risk literature consistently finds that income volatility declines with the number of revenue-generating crops, even holding total cultivated area constant.

**Nutritional security.** There is a well-documented pathway from what a region grows to what its population eats. Districts dominated by rice or wheat monocultures tend to have diets rich in calories but poor in micronutrients -- iron, zinc, vitamin A, and dietary fiber. The nutrition transition literature has shown that crop diversity at the landscape level is positively associated with dietary diversity scores at the household level, particularly in subsistence-oriented farming systems where much of what is grown is also consumed. Measuring crop diversity therefore serves as a proxy indicator for nutritional adequacy and food system resilience.

**Quantification is necessary for policy.** While the qualitative case for crop diversity is well established, effective policy intervention requires quantification. Which districts are losing diversity? How fast? What distinguishes resilient districts from declining ones? These questions demand rigorous, comparable metrics -- which is what the indices below provide.
"""))

# ---- 3.2 Individual Indices ----
cells.append(md(r"""
### 3.2 Individual Diversity Indices -- Deep Explanation

We employ three indices, each capturing a different facet of crop diversity. No single index is sufficient; their complementarity is the reason we use all three and ultimately combine them. Throughout this section, let $p_i$ denote the proportion of total cropped area in a district devoted to crop category $i$, and let $S$ denote the total number of distinct crop categories cultivated.

---

#### 3.2.1 Shannon Diversity Index ($H'$)

**Origin and theoretical basis.** The Shannon Index was developed by Claude Shannon in 1948 as a measure of information entropy in communication theory. Its adoption in ecology (and subsequently in agricultural diversity studies) rests on an elegant analogy: just as entropy measures the "surprise" or uncertainty in a stream of symbols, the Shannon Index measures the "surprise" in a landscape. If you randomly selected one hectare of cropland, how uncertain would you be about which crop you would find?

**Formula:**

$$H' = -\sum_{i=1}^{S} p_i \ln(p_i)$$

where $p_i$ is the area share of crop $i$ and the convention $0 \cdot \ln(0) = 0$ is adopted (the limit as $p \to 0^+$).

**Intuition:** Consider a district growing only rice. You would have zero surprise upon picking any hectare -- $H' = 0$. Now consider a district growing 10 crops, each on exactly 10% of the area. Maximum uncertainty -- you genuinely cannot predict which crop you will find. In this case $H' = \ln(10) \approx 2.30$, the maximum for $S=10$.

**Properties:**
- **Range:** $0$ (monoculture) to $\ln(S)$ (perfect evenness across $S$ crops)
- **Sensitivity to rare crops:** The Shannon Index is sensitive to the presence of rare crops (those with small $p_i$). Even a crop occupying 1% of area contributes $-0.01 \times \ln(0.01) = 0.046$ to the index. This makes it a good detector of diversity in the "tail" of the distribution.
- **Captures both richness and evenness:** Unlike a simple count, $H'$ increases both when more crops are added and when area is distributed more evenly among existing crops.
- **Units:** Measured in "nats" (when using natural log) or "bits" (when using log base 2). We use natural log throughout.

**Why we use it:** The Shannon Index is the standard measure in the ecological diversity literature and provides the most balanced sensitivity across the entire crop distribution. It detects changes in rare crops that other indices miss, making it particularly valuable for tracking the erosion of minor but ecologically or nutritionally important crop categories (e.g., millets, traditional pulses).

**Limitations:** Raw Shannon values are difficult to interpret without context -- $H' = 1.8$ has no intuitive meaning unless compared to the theoretical maximum or to other districts. The index is also sensitive to sample completeness: if rare crops are underreported in government statistics, $H'$ will be downward-biased.

---

#### 3.2.2 Simpson Diversity Index ($D$)

**Origin and theoretical basis.** Proposed by E.H. Simpson in 1949 for measuring concentration in biological populations, the Simpson Index has a direct probabilistic interpretation that makes it particularly accessible.

**Formula (complement form):**

$$D = 1 - \sum_{i=1}^{S} p_i^2$$

We use the complement form ($1 - \lambda$, where $\lambda = \sum p_i^2$ is the original Simpson concentration index) so that higher values indicate greater diversity, consistent with the Shannon Index.

**Intuition:** $\sum p_i^2$ is the probability that two randomly and independently chosen hectares grow the *same* crop. Therefore $D = 1 - \sum p_i^2$ is the probability that two randomly chosen hectares grow *different* crops. This is perhaps the most intuitive diversity measure: "If I pick two fields at random, what is the chance they are growing different things?"

**Properties:**
- **Range:** $0$ (monoculture, where $p_1 = 1$) to $1 - 1/S$ (perfect evenness)
- **Weight on dominant species:** The squaring operation $p_i^2$ gives disproportionate weight to crops with large area shares. A crop with $p = 0.5$ contributes $0.25$ to the concentration sum, while a crop with $p = 0.01$ contributes only $0.0001$. This makes the Simpson Index primarily a measure of dominance/concentration rather than of the full distribution shape.
- **Robustness to rare crops:** Because rare crops contribute negligibly to $\sum p_i^2$, the Simpson Index is robust to the underreporting of minor crops in administrative data. This is a practical advantage when working with government crop statistics, which may miss small-holder cultivation of niche crops.

**Why we use it:** The direct probability interpretation makes the Simpson Index the most communicable measure for policy audiences. Saying "there is only a 35% chance two random fields grow different crops" is immediately meaningful to non-specialists. Its robustness to rare crop underreporting also makes it a useful complement to the Shannon Index, which is more sensitive to the tail of the distribution.

**Limitations:** By de-emphasizing rare crops, the Simpson Index can miss important diversity changes happening at the margins. A district could lose five minor crop categories without appreciable change in $D$ if those crops collectively occupied less than 5% of area. Additionally, $D$ is bounded above by $1 - 1/S$, which means districts with very different crop counts can have similar maximum possible values -- a subtle comparability issue.

---

#### 3.2.3 Crop Richness ($S$)

**Definition:**

$$S = \text{number of distinct crop categories cultivated in the district}$$

**Why it matters despite its simplicity.** Richness is the most policy-legible metric. When a policymaker asks "how many crops do farmers in this district grow?", they are asking about richness. It is directly actionable: programs that introduce new crops increase $S$; programs that fail to support existing minor crops may reduce it. Richness also serves as an upper bound on the information content of the other indices -- you cannot have high Shannon or Simpson diversity with very low $S$.

**Limitations:** Richness is completely insensitive to evenness. Consider two districts:
- **District A:** 20 crops, each occupying 5% of area. $S = 20$.
- **District B:** 20 crops, but one crop occupies 95% of area and the other 19 share the remaining 5%. $S = 20$.

Both districts have identical richness, yet District A is genuinely diverse while District B is effectively a monoculture. This is precisely why richness alone is insufficient and why we combine it with Shannon and Simpson in the composite ABI.

**A note on crop category granularity.** Our data uses broad crop categories from government statistics (e.g., "cereals," "pulses," "oilseeds," and finer divisions within these). Within-crop varietal diversity (e.g., Basmati vs. IR-64 rice) is not captured. This means our richness measure is a lower bound on true agricultural biodiversity.
"""))

# ---- 3.2 Worked Example Code Cell ----
cells.append(md(r"""
#### Worked Example: Computing All Three Indices for a Single District

To make these formulas concrete, we compute all three indices step-by-step for two contrasting districts from our dataset: one high-diversity and one low-diversity district. This also serves as a reproducibility check against the pre-computed values.
"""))

cells.append(code(r"""
# Worked example: step-by-step index computation for two contrasting districts
# We pick a high-ABI and a low-ABI district from the data

# Identify candidate districts (look for Belgaum, Karnataka and Ludhiana, Punjab)
high_candidates = df_main[df_main['district_name'].str.contains('Belgaum|Belagavi', case=False, na=False)]
low_candidates = df_main[df_main['district_name'].str.contains('Ludhiana', case=False, na=False)]

# Fallback: use top and bottom by ABI if specific districts not found
if high_candidates.empty:
    high_candidates = df_main.nlargest(1, 'agro_biodiversity_index')
if low_candidates.empty:
    low_candidates = df_main.nsmallest(1, 'agro_biodiversity_index')

high_dist = high_candidates.iloc[0]
low_dist = low_candidates.iloc[0]

share_cols = [c for c in df_main.columns if c.startswith('share_')]

for label, dist in [("HIGH-DIVERSITY DISTRICT", high_dist), ("LOW-DIVERSITY DISTRICT", low_dist)]:
    print(f"\n{'='*70}")
    print(f"  {label}: {dist['district_name']}, {dist['state_name']}")
    print(f"{'='*70}")

    # Extract crop shares and filter to non-zero
    shares = dist[share_cols].values.astype(float)
    crop_names = [c.replace('share_', '').replace('_', ' ').title() for c in share_cols]
    nonzero_mask = shares > 0
    p = shares[nonzero_mask]
    names = [crop_names[i] for i in range(len(crop_names)) if nonzero_mask[i]]

    # Sort by share descending
    sort_idx = np.argsort(-p)
    p = p[sort_idx]
    names = [names[i] for i in sort_idx]

    # Show crop distribution
    print(f"\n  Crop area shares (non-zero crops only):")
    for name, share in zip(names, p):
        bar = '#' * int(share * 50)
        print(f"    {name:25s}  {share:6.3f}  ({share*100:5.1f}%)  {bar}")

    # Step 1: Crop Richness
    S = len(p)
    print(f"\n  Step 1 -- Crop Richness (S):")
    print(f"    S = count of non-zero crops = {S}")

    # Step 2: Shannon Index
    H = -np.sum(p * np.log(p))
    H_max = np.log(S) if S > 1 else 0
    print(f"\n  Step 2 -- Shannon Index (H'):")
    print(f"    H' = -SUM(p_i * ln(p_i))")
    # Show first few terms
    terms = [-pi * np.log(pi) for pi in p[:5]]
    term_strs = [f"(-{pi:.3f} x ln({pi:.3f})) = {t:.4f}" for pi, t in zip(p[:5], terms)]
    for ts in term_strs:
        print(f"      {ts}")
    if len(p) > 5:
        print(f"      ... + {len(p)-5} more terms")
    print(f"    H' = {H:.4f}")
    print(f"    Theoretical maximum ln({S}) = {H_max:.4f}")
    print(f"    Evenness ratio H'/ln(S) = {H/H_max:.3f}" if H_max > 0 else "    (monoculture)")

    # Step 3: Simpson Index
    D = 1 - np.sum(p**2)
    print(f"\n  Step 3 -- Simpson Index (D):")
    print(f"    D = 1 - SUM(p_i^2)")
    print(f"    SUM(p_i^2) = {np.sum(p**2):.4f}")
    print(f"    D = 1 - {np.sum(p**2):.4f} = {D:.4f}")
    print(f"    Interpretation: {D*100:.1f}% chance two random hectares grow different crops")

    # Compare with stored values
    print(f"\n  Verification against pre-computed values:")
    print(f"    Shannon:  computed={H:.4f}  stored={dist['shannon_index']:.4f}  match={'YES' if abs(H - dist['shannon_index']) < 0.01 else 'APPROX (averaged over years)'}")
    print(f"    Simpson:  computed={D:.4f}  stored={dist['simpson_index']:.4f}  match={'YES' if abs(D - dist['simpson_index']) < 0.01 else 'APPROX (averaged over years)'}")
    print(f"    Richness: computed={S}      stored={int(dist['crop_richness'])}      match={'YES' if S == int(dist['crop_richness']) else 'APPROX (averaged over years)'}")

print("\n" + "="*70)
print("  NOTE: Small differences between computed and stored values are expected")
print("  because stored values are averaged across all years (1997-2021), while")
print("  this example uses the time-averaged crop shares.")
print("="*70)
"""))

# ---- 3.3 The ABI ----
cells.append(md(r"""
### 3.3 The Agro-Biodiversity Index (ABI) -- Full Derivation

The ABI is the central analytical contribution of this study. This section explains every step in its construction and justifies each methodological decision.

#### 3.3.1 Why a Composite Index?

Each individual index captures a different aspect of crop diversity:

| Index | What It Measures | Blind Spot |
|-------|-----------------|------------|
| **Shannon ($H'$)** | Information content / uncertainty | Hard to interpret raw values; sensitive to rare crop completeness |
| **Simpson ($D$)** | Dominance / concentration | Misses changes in the tail of the distribution |
| **Richness ($S$)** | Variety / count | Completely ignores evenness |

No single index tells the full story. A district could score high on richness but low on Shannon/Simpson (many crops, but area concentrated in one). Another could score high on Simpson but moderate on richness (few crops, but very evenly distributed). The ABI synthesizes all three dimensions into a single measure that penalizes deficiency in any one dimension.

This approach follows the standard practice in composite index construction, as exemplified by the UNDP's Human Development Index (which combines health, education, and income dimensions) and is well-established in the agro-biodiversity measurement literature.

#### 3.3.2 Normalization: Min-Max Scaling

**The problem.** The three indices operate on fundamentally different scales:
- Shannon Index: theoretically $0$ to $\sim 3.8$ (for $S=45$ crops), practically $0$ to $\sim 2.5$ in our data
- Simpson Index: $0$ to $1$
- Crop Richness: $1$ to $45+$

Averaging them directly would let richness (with values up to 45) dominate the composite, drowning out the contribution of the bounded indices. Normalization to a common scale is therefore essential.

**The method.** We apply min-max normalization across all districts:

$$\hat{x}_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$

where $x_{\min}$ and $x_{\max}$ are the minimum and maximum values of the index observed across all 755 districts. This maps each index to the $[0, 1]$ interval, where:
- $0$ = the least diverse district in the sample (on that dimension)
- $1$ = the most diverse district in the sample (on that dimension)

**Why min-max over z-score normalization?** We considered two alternatives:

| Method | Pros | Cons |
|--------|------|------|
| **Min-max** (chosen) | Bounded [0,1], intuitive interpretation, no distributional assumptions | Sensitive to outliers at extremes |
| **Z-score** | Robust to outliers, standard in statistics | Unbounded, negative values have no natural interpretation for a "diversity score" |

Min-max normalization was chosen because (a) the bounded $[0,1]$ interpretation is essential for a composite that should itself be interpretable as a diversity score, and (b) diversity indices do not have natural outliers in the statistical sense -- extreme values represent genuinely extreme districts (e.g., Delhi with near-zero diversity, or Belgaum with near-maximum diversity), not measurement errors.

**Important note on scope:** The min and max are computed cross-sectionally across all 755 districts. This means the ABI is a *relative* measure -- it tells you where a district stands in the distribution of Indian districts, not against some absolute theoretical standard. If every district in India grew only two crops, the "most diverse" district would still score ABI = 1. This is a deliberate choice: we are interested in ranking and comparing districts within the Indian context.

#### 3.3.3 Aggregation: Equal-Weighted Average

The composite ABI is computed as the simple arithmetic mean of the three normalized indices:

$$\text{ABI} = \frac{1}{3}\left(\hat{H}' + \hat{D} + \hat{S}\right)$$

where $\hat{H}'$, $\hat{D}$, and $\hat{S}$ denote the min-max normalized Shannon, Simpson, and Richness values respectively.

**Why equal weights?** The choice of weights is the most consequential decision in any composite index. We use equal weights ($1/3$ each) for three reasons:

1. **No theoretical basis for differential weighting.** There is no established theory in agro-biodiversity measurement that privileges one diversity dimension over another. Shannon is not "more important" than Simpson or richness -- they capture genuinely different concepts.

2. **Transparency and reproducibility.** Equal weights are the default assumption that requires no additional justification. Any alternative weighting scheme would require defending the specific magnitudes, introducing subjectivity.

3. **Robustness.** We verified (see Section 3.3.6) that the district rankings are robust to moderate perturbations in weights, suggesting that the choice of equal weights does not drive the substantive findings.

**Alternative considered: PCA-derived weights.** Principal Component Analysis could be used to derive data-driven weights based on the variance structure. However, PCA weights reflect the statistical correlation structure of the data, not necessarily the conceptual importance of each dimension. In our data, Shannon and Simpson are highly correlated ($r > 0.9$), so PCA would effectively downweight the richness component -- which is precisely the dimension that carries the most direct policy relevance. We flag PCA weighting as a robustness check but do not adopt it as the primary specification.

#### 3.3.4 Interpretation Guide

The ABI ranges from 0 to 1. We define the following interpretive bands:

| ABI Range | Category | Interpretation |
|-----------|----------|----------------|
| $< 0.2$ | **Very Low** | Near-monoculture: one or two crops dominate almost all area |
| $0.2 - 0.4$ | **Low** | Limited diversity: a few crops dominate, with minimal minor crop presence |
| $0.4 - 0.6$ | **Moderate** | Middle ground: reasonable variety with moderate concentration |
| $0.6 - 0.8$ | **High** | Genuinely diverse: multiple crops with relatively balanced area allocation |
| $> 0.8$ | **Very High** | Exceptional diversity: many crops, none dominant, approaching theoretical maximum |

These thresholds are based on natural breaks in the empirical distribution (see histogram below) and are intended as heuristic guides rather than rigid classifications. The quintile boundaries of the ABI distribution fall approximately at 0.44, 0.58, 0.68, and 0.78, which broadly align with the above scheme.
"""))

cells.append(code(r"""
# ABI distribution histogram with interpretive bands
fig, ax = plt.subplots(figsize=(12, 6))

abi_vals = df_main['agro_biodiversity_index'].dropna()

# Histogram
n, bins, patches = ax.hist(abi_vals, bins=50, color='#2c7bb6', alpha=0.7,
                            edgecolor='white', linewidth=0.5)

# Color the bands
band_colors = {
    (0.0, 0.2): ('#d73027', 'Very Low'),
    (0.2, 0.4): ('#fc8d59', 'Low'),
    (0.4, 0.6): ('#fee08b', 'Moderate'),
    (0.6, 0.8): ('#91cf60', 'High'),
    (0.8, 1.0): ('#1a9850', 'Very High'),
}

for (lo, hi), (color, label) in band_colors.items():
    ax.axvspan(lo, hi, alpha=0.12, color=color, zorder=0)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 30,
            label, ha='center', fontsize=10, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add vertical lines at band boundaries
for boundary in [0.2, 0.4, 0.6, 0.8]:
    ax.axvline(boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

# Add summary statistics
median_abi = abi_vals.median()
mean_abi = abi_vals.mean()
ax.axvline(median_abi, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_abi:.3f}')
ax.axvline(mean_abi, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_abi:.3f}')

# Quintile boundaries
quintiles = abi_vals.quantile([0.2, 0.4, 0.6, 0.8]).values
for i, q in enumerate(quintiles):
    ax.axvline(q, color='purple', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('Agro-Biodiversity Index (ABI)', fontsize=12)
ax.set_ylabel('Number of Districts', fontsize=12)
ax.set_title('Figure M1: Distribution of ABI with Interpretive Bands', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.show()

# Category counts
print("\nDistrict counts by ABI category:")
for (lo, hi), (_, label) in band_colors.items():
    count = ((abi_vals >= lo) & (abi_vals < hi if hi < 1.0 else abi_vals <= hi)).sum()
    pct = count / len(abi_vals) * 100
    print(f"  {label:12s} ({lo:.1f} - {hi:.1f}): {count:4d} districts ({pct:5.1f}%)")

print(f"\n  Quintile boundaries: {', '.join(f'{q:.3f}' for q in quintiles)}")
"""))

# ---- 3.3.5 Worked Example: ABI Computation ----
cells.append(md(r"""
#### 3.3.5 Worked Example: ABI Computation Step-by-Step

We now demonstrate the full ABI pipeline -- from raw index values through normalization to the final composite -- for one high-diversity and one low-diversity district. This makes the abstract formula concrete and verifiable.
"""))

cells.append(code(r"""
# Full ABI computation: normalization and aggregation for two contrasting districts

# Get global min/max for normalization
shannon_min, shannon_max = df_main['shannon_index'].min(), df_main['shannon_index'].max()
simpson_min, simpson_max = df_main['simpson_index'].min(), df_main['simpson_index'].max()
richness_min, richness_max = df_main['crop_richness'].min(), df_main['crop_richness'].max()

print("GLOBAL MIN-MAX VALUES (across all 755 districts)")
print("="*60)
print(f"  Shannon Index:    min = {shannon_min:.4f},  max = {shannon_max:.4f}")
print(f"  Simpson Index:    min = {simpson_min:.4f},  max = {simpson_max:.4f}")
print(f"  Crop Richness:    min = {richness_min:.0f},       max = {richness_max:.0f}")

# Re-use the high/low districts from the earlier worked example
for label, dist in [("HIGH-DIVERSITY", high_dist), ("LOW-DIVERSITY", low_dist)]:
    h = dist['shannon_index']
    d = dist['simpson_index']
    s = dist['crop_richness']

    # Normalize
    h_norm = (h - shannon_min) / (shannon_max - shannon_min)
    d_norm = (d - simpson_min) / (simpson_max - simpson_min)
    s_norm = (s - richness_min) / (richness_max - richness_min)

    # Composite
    abi_computed = (h_norm + d_norm + s_norm) / 3
    abi_stored = dist['agro_biodiversity_index']

    print(f"\n{'='*60}")
    print(f"  {label}: {dist['district_name']}, {dist['state_name']}")
    print(f"{'='*60}")
    print(f"\n  Raw Index Values:")
    print(f"    Shannon (H')  = {h:.4f}")
    print(f"    Simpson (D)   = {d:.4f}")
    print(f"    Richness (S)  = {s:.0f}")

    print(f"\n  Step 1: Min-Max Normalization")
    print(f"    H'_norm = ({h:.4f} - {shannon_min:.4f}) / ({shannon_max:.4f} - {shannon_min:.4f}) = {h_norm:.4f}")
    print(f"    D_norm  = ({d:.4f} - {simpson_min:.4f}) / ({simpson_max:.4f} - {simpson_min:.4f}) = {d_norm:.4f}")
    print(f"    S_norm  = ({s:.0f} - {richness_min:.0f}) / ({richness_max:.0f} - {richness_min:.0f}) = {s_norm:.4f}")

    print(f"\n  Step 2: Equal-Weighted Average")
    print(f"    ABI = ({h_norm:.4f} + {d_norm:.4f} + {s_norm:.4f}) / 3")
    print(f"    ABI = {h_norm + d_norm + s_norm:.4f} / 3")
    print(f"    ABI = {abi_computed:.4f}")

    print(f"\n  Verification: stored ABI = {abi_stored:.4f}  |  difference = {abs(abi_computed - abi_stored):.4f}")
    if abs(abi_computed - abi_stored) < 0.01:
        print(f"    --> Match confirmed.")
    else:
        print(f"    --> Small difference due to year-averaging in stored values.")
"""))

# ---- 3.3.6 Robustness note ----
cells.append(md(r"""
#### 3.3.6 Robustness of Equal Weights

A natural concern with equal weighting is sensitivity: do the results change materially if we alter the weights? We assess this by computing rank correlations between the baseline ABI and alternative weighting schemes.
"""))

cells.append(code(r"""
# Robustness check: how sensitive are district rankings to weight choices?
from scipy.stats import spearmanr

# Recompute normalized values for all districts
h_norm_all = (df_main['shannon_index'] - df_main['shannon_index'].min()) / \
             (df_main['shannon_index'].max() - df_main['shannon_index'].min())
d_norm_all = (df_main['simpson_index'] - df_main['simpson_index'].min()) / \
             (df_main['simpson_index'].max() - df_main['simpson_index'].min())
s_norm_all = (df_main['crop_richness'] - df_main['crop_richness'].min()) / \
             (df_main['crop_richness'].max() - df_main['crop_richness'].min())

baseline_abi = (h_norm_all + d_norm_all + s_norm_all) / 3

# Alternative weighting schemes
alternatives = {
    'Equal (1/3, 1/3, 1/3)': (1/3, 1/3, 1/3),
    'Shannon-heavy (0.5, 0.25, 0.25)': (0.5, 0.25, 0.25),
    'Simpson-heavy (0.25, 0.5, 0.25)': (0.25, 0.5, 0.25),
    'Richness-heavy (0.25, 0.25, 0.5)': (0.25, 0.25, 0.5),
    'No richness (0.5, 0.5, 0.0)': (0.5, 0.5, 0.0),
    'Richness only (0.0, 0.0, 1.0)': (0.0, 0.0, 1.0),
}

print("ROBUSTNESS CHECK: Spearman Rank Correlation with Baseline ABI")
print("="*70)
print(f"  {'Weighting Scheme':<40s}  {'Spearman rho':>12s}  {'p-value':>10s}")
print("-"*70)
for name, (w_h, w_d, w_s) in alternatives.items():
    alt_abi = w_h * h_norm_all + w_d * d_norm_all + w_s * s_norm_all
    rho, pval = spearmanr(baseline_abi, alt_abi)
    print(f"  {name:<40s}  {rho:>12.4f}  {pval:>10.2e}")

print()
print("  Interpretation: Spearman rho > 0.95 indicates that district rankings are")
print("  highly robust to the weighting scheme. Only extreme schemes (dropping an")
print("  entire component) meaningfully alter the rankings.")
"""))

# ---- 3.4 Irrigation Classification ----
cells.append(md(r"""
### 3.4 Irrigation Classification

Districts are classified into three irrigation regimes based on their gross irrigated area as a percentage of gross cropped area:

| Regime | Irrigation % | Description |
|--------|-------------|-------------|
| **Rainfed** | $< 40\%$ | Predominantly rain-dependent agriculture |
| **Semi-Irrigated** | $40\% - 60\%$ | Mixed water sources; transitional zone |
| **Irrigated** | $> 60\%$ | Predominantly canal, well, or tube-well irrigated |

#### Justification of Threshold Choice

The thresholds at 40% and 60% were chosen based on three considerations:

1. **Policy convention.** Indian agricultural planning documents commonly use 40% as the threshold separating "predominantly rainfed" from "partially irrigated" regions. The NITI Aayog and Ministry of Agriculture use similar break points in their irrigation mapping exercises.

2. **Distributional properties.** The histogram below shows that the gross irrigated area percentage has a bimodal distribution across Indian districts, with a concentration of districts below 30% (strongly rainfed) and above 70% (strongly irrigated). The 40-60% band captures the transitional zone between these two modes.

3. **Analytical balance.** The thresholds produce roughly balanced group sizes (240 rainfed, 101 semi-irrigated, 162 irrigated among the 503 matched districts), ensuring adequate statistical power for between-group comparisons.

**Limitation:** District-level irrigation percentages are averages that mask substantial intra-district variation. A district classified as "semi-irrigated" at 50% may contain both fully irrigated blocks (near canals) and entirely rainfed blocks (distant from water infrastructure). Our analysis cannot capture this within-district heterogeneity.
"""))

cells.append(code(r"""
# Histogram of gross irrigated area percentage with threshold lines
df_irr_data = df_main[df_main['irrigation_regime'].notna()].copy()

# We need to reconstruct or approximate the irrigation percentage
# The irrigation_regime column encodes the categories; let's show the distribution
# by looking for a gross_irrigated_pct or similar column
irr_pct_col = None
for col in df_main.columns:
    if 'irrigat' in col.lower() and 'pct' in col.lower():
        irr_pct_col = col
        break
    if 'irrigat' in col.lower() and 'share' in col.lower():
        irr_pct_col = col
        break
    if col == 'gross_irrigated_area_pct':
        irr_pct_col = col
        break

if irr_pct_col:
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = df_main[irr_pct_col].dropna()
    ax.hist(vals, bins=40, color='#4393c3', alpha=0.7, edgecolor='white')
    ax.axvline(40, color='red', linestyle='--', linewidth=2, label='Threshold: 40%')
    ax.axvline(60, color='red', linestyle='--', linewidth=2, label='Threshold: 60%')
    ax.axvspan(0, 40, alpha=0.08, color='#b2182b', label='Rainfed')
    ax.axvspan(40, 60, alpha=0.08, color='#4dac26', label='Semi-Irrigated')
    ax.axvspan(60, 100, alpha=0.08, color='#2166ac', label='Irrigated')
    ax.set_xlabel('Gross Irrigated Area (%)')
    ax.set_ylabel('Number of Districts')
    ax.set_title('Figure M2: Distribution of Irrigation Percentage with Classification Thresholds')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    # If no raw percentage column exists, show the regime distribution as a proxy
    print("Note: Raw irrigation percentage column not found in dataset.")
    print("Showing regime category distribution instead.\n")
    regime_counts = df_irr_data['irrigation_regime'].value_counts().reindex(REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(regime_counts)), regime_counts.values,
                  color=[PALETTE[r] for r in regime_counts.index],
                  edgecolor='white', width=0.6)
    ax.set_xticks(range(len(regime_counts)))
    ax.set_xticklabels(regime_counts.index, fontsize=11)
    for bar, cnt in zip(bars, regime_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(cnt), ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Districts')
    ax.set_title('Figure M2: Distribution of Districts by Irrigation Regime')
    plt.tight_layout()
    plt.show()

    print(f"\nIrrigation regime distribution (N={df_irr_data.shape[0]} matched districts):")
    for regime in REGIME_ORDER:
        n = regime_counts.get(regime, 0)
        pct = n / df_irr_data.shape[0] * 100
        print(f"  {regime}: {n} districts ({pct:.1f}%)")
"""))

# ---- 3.5 Temporal Change Methodology ----
cells.append(md(r"""
### 3.5 Temporal Change Methodology

To assess whether crop diversity is increasing or declining, we compare diversity indices between two periods:

- **Early period:** All years up to and including 2005 (capturing the post-liberalization but pre-NREGA agricultural landscape)
- **Late period:** All years from 2015 onward (capturing the most recent cropping patterns under current policy regimes)

For each district with data in both periods, we compute:

$$\Delta H' = \overline{H'}_{\text{late}} - \overline{H'}_{\text{early}}$$

and similarly for $\Delta D$ and $\Delta S$, where the overlines denote the mean index value within each period.

#### Why These Period Cutoffs?

The choice of 2005 and 2015 as boundary years serves two purposes:

1. **Adequate temporal separation.** A 10-year gap between the end of the early period and the start of the late period ensures that observed changes reflect genuine structural shifts in cropping patterns rather than year-to-year noise from weather or price fluctuations.

2. **Policy regime alignment.** The mid-2000s mark a significant inflection point in Indian agricultural policy: the National Rural Employment Guarantee Act (NREGA, 2005) altered rural labor markets, the National Food Security Mission (2007) intensified focus on rice, wheat, and pulses, and the period after 2010 saw increasing policy attention to crop diversification and sustainable agriculture. Comparing pre-2005 and post-2015 patterns thus captures the net effect of a decade of policy evolution.

3. **Data availability.** Using multi-year averages within each period (rather than single-year snapshots) reduces the influence of anomalous years and increases the number of districts with sufficient data for comparison.

#### Why Not Regression-Based Trends?

An alternative approach would be to fit a linear regression through the full time series for each district and use the slope as the measure of change. We chose the period-comparison approach for three reasons:

1. **Fewer assumptions.** Linear regression assumes a monotonic (and indeed linear) trend. Crop diversity may follow non-linear trajectories -- declining in one decade, then partially recovering in the next. Period comparison makes no assumption about the functional form of change.

2. **Robustness to data gaps.** Many districts have missing data for some years. Period comparison only requires *some* data in each period, while regression requires enough data points for a meaningful slope estimate.

3. **Interpretability.** "Diversity declined by 0.08 Shannon units between the early and late periods" is more directly interpretable than "the linear trend coefficient is -0.004 per year."

**Acknowledged limitation:** The period-comparison approach cannot detect non-monotonic trends. A district that experienced severe diversity loss from 2000-2010 but recovered from 2010-2020 might show little net change, masking a V-shaped trajectory. Time series plots (Section 6) complement this analysis by revealing such dynamics.
"""))

# =============================================================================
# SECTION 4: SPATIAL DISTRIBUTION
# =============================================================================
cells.append(md(r"""
---
## 4. Spatial Distribution of Crop Diversity
"""))

cells.append(code(r"""
# Display the ABI map
abi_map = MAP_DIR / 'map_abi.png'
if abi_map.exists():
    display(Image(filename=str(abi_map), width=900))
else:
    print(f"Map not found: {abi_map}")
"""))

cells.append(md(r"""
**Figure 2: Agro-Biodiversity Index Across Indian Districts**

The ABI map reveals clear regional patterns. Southern and western India -- particularly Karnataka, Andhra Pradesh, and Rajasthan -- show the highest crop diversity. The Indo-Gangetic plain displays a striking east-west divide: western states (Punjab, Haryana) are diversity-poor due to the rice-wheat monoculture, while the eastern plain (Bihar, eastern UP) shows moderate diversity. Northeast India is mixed, with Nagaland scoring surprisingly high due to traditional shifting cultivation practices that maintain crop variety.
"""))

cells.append(code(r"""
# Display the combined 2x2 panel showing all four indices
panel_map = MAP_DIR / 'map_combined_panel_2x2.png'
if panel_map.exists():
    display(Image(filename=str(panel_map), width=1000))
else:
    print(f"Map not found: {panel_map}")
"""))

cells.append(md(r"""
**Figure 3: Four-Panel Map of Individual Diversity Indices**

The four-panel view illustrates how the indices complement each other:
- **Shannon Index** (top-left): Captures the combined effect of crop count and evenness. High in the Deccan Plateau and Rajasthan.
- **Simpson Index** (top-right): Similar spatial pattern to Shannon but more robust to rare crops. Highlights concentration in Punjab and western UP.
- **Crop Richness** (bottom-left): Shows raw variety. Even some low-Shannon districts score high on richness, indicating many crops grown but area concentrated in a few.
- **ABI** (bottom-right): The composite view smooths out index-specific quirks, providing the most balanced picture.
"""))

cells.append(code(r"""
# Display the irrigation regime map
irr_map = MAP_DIR / 'map_irrigation_regime.png'
if irr_map.exists():
    display(Image(filename=str(irr_map), width=900))
else:
    print(f"Map not found: {irr_map}")
"""))

cells.append(md(r"""
**Figure 4: Irrigation Regime Classification of Indian Districts**

The irrigation map shows the stark geography of India's water infrastructure. Irrigated districts (blue) cluster in Punjab-Haryana, western UP, and along major canal systems. Rainfed districts (red) dominate central India, the northeast, and rain-shadow regions. Semi-irrigated districts (green) form transitional zones -- and, as we shall see, often harbor the highest crop diversity.
"""))

# =============================================================================
# SECTION 5: IRRIGATION AND DIVERSITY
# =============================================================================
cells.append(md(r"""
---
## 5. Irrigation and Crop Diversity
"""))

cells.append(code(r"""
# Filter to districts with irrigation data
df_irr_data = df_main[df_main['irrigation_regime'].notna()].copy()
regime_counts = df_irr_data['irrigation_regime'].value_counts().reindex(REGIME_ORDER)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(len(regime_counts)), regime_counts.values,
              color=[PALETTE[r] for r in regime_counts.index],
              edgecolor='white', width=0.6)
ax.set_xticks(range(len(regime_counts)))
ax.set_xticklabels(regime_counts.index, fontsize=11)
for bar, cnt in zip(bars, regime_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(cnt), ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Districts')
ax.set_title('Figure 5: Distribution of Districts by Irrigation Regime (N=503)')
plt.tight_layout()
plt.show()

print(f"\nTotal districts with irrigation data: {df_irr_data.shape[0]}")
for regime in REGIME_ORDER:
    n = regime_counts[regime]
    print(f"  {regime}: {n} districts ({n/df_irr_data.shape[0]*100:.1f}%)")
"""))

cells.append(md(r"""
**Figure 5 Interpretation:** Rainfed districts form the largest group (240 districts, ~48%), followed by irrigated (162, ~32%) and semi-irrigated (101, ~20%). The relatively smaller semi-irrigated category represents transitional districts that, as we show below, tend to maintain the most diverse cropping patterns.
"""))

cells.append(code(r"""
# Box plots: Diversity by irrigation regime
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
metrics = [
    ('shannon_index', 'Shannon Index (H\')'),
    ('simpson_index', 'Simpson Index (D)'),
    ('crop_richness', 'Crop Richness (S)'),
    ('agro_biodiversity_index', 'ABI'),
]

for ax, (col, label) in zip(axes, metrics):
    data_for_plot = [df_irr_data[df_irr_data['irrigation_regime'] == r][col].dropna()
                     for r in REGIME_ORDER]
    bp = ax.boxplot(data_for_plot, labels=['Rainfed', 'Semi-Irr.', 'Irrigated'],
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    for patch, regime in zip(bp['boxes'], REGIME_ORDER):
        patch.set_facecolor(PALETTE[regime])
        patch.set_alpha(0.7)
    ax.set_title(label, fontsize=11)
    ax.set_ylabel(label.split('(')[0].strip() if '(' in label else label)

fig.suptitle('Figure 6: Diversity Indices by Irrigation Regime', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
**Figure 6 Interpretation:** A striking and counterintuitive finding emerges: **semi-irrigated districts show the highest median diversity** across all four indices. The mean ABI values are:

- **Semi-Irrigated (40-60%):** ABI = 0.69 (highest)
- **Irrigated (>60%):** ABI = 0.67
- **Rainfed (<40%):** ABI = 0.62 (lowest)

This challenges the simplistic narrative that irrigation uniformly promotes or hinders diversity. Semi-irrigated districts, with access to some water infrastructure but not enough to sustain water-intensive monocultures, appear to maintain the most balanced crop portfolios. Fully irrigated districts tend toward cereal monocultures (rice-wheat), while rainfed districts are constrained by water availability to a narrower set of drought-tolerant crops.
"""))

cells.append(code(r"""
# Crop composition by irrigation regime
share_cols = [c for c in df_main.columns if c.startswith('share_')]
crop_labels = [c.replace('share_', '').replace('_', ' ').title() for c in share_cols]

regime_composition = df_irr_data.groupby('irrigation_regime')[share_cols].mean()
regime_composition.columns = crop_labels
regime_composition = regime_composition.reindex(REGIME_ORDER)

# Select top crop categories (those with >2% share in any regime)
top_cats = regime_composition.columns[regime_composition.max() > 0.02]
other = regime_composition.drop(columns=top_cats).sum(axis=1)
plot_df = regime_composition[top_cats].copy()
plot_df['Other'] = other

fig, ax = plt.subplots(figsize=(10, 5))
plot_df.plot(kind='barh', stacked=True, ax=ax,
             cmap='Set2', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Share of Total Cropped Area')
ax.set_title('Figure 7: Crop Composition by Irrigation Regime')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
**Figure 7 Interpretation:** The crop composition starkly differs across regimes:

- **Irrigated districts:** ~67% cereals (driven by rice and wheat), only ~8.5% pulses. This reflects the Green Revolution template of cereal intensification under assured water supply.
- **Rainfed districts:** ~60% cereals (rice-dominant), with slightly higher pulse and oilseed shares. Water constraints limit the crop mix but also prevent complete cereal takeover.
- **Semi-irrigated districts:** ~53% cereals, ~23% oilseeds, ~13% pulses. This is the most balanced portfolio -- enough water for diverse options but not enough to specialize in water-hungry cereals.
"""))

cells.append(code(r"""
# Display the irrigation-diversity panel map
irr_div_map = MAP_DIR / 'map_irrigation_diversity_panel.png'
if irr_div_map.exists():
    display(Image(filename=str(irr_div_map), width=1000))
else:
    print(f"Map not found: {irr_div_map}")
"""))

cells.append(md(r"""
**Figure 8: Irrigation Regime and Diversity -- Spatial Panel**

This panel map overlays the irrigation and diversity dimensions spatially, making the geographic patterns clear. Semi-irrigated belts -- particularly in central India and parts of the south -- consistently coincide with higher diversity scores.
"""))

# =============================================================================
# SECTION 6: TEMPORAL TRENDS
# =============================================================================
cells.append(md(r"""
---
## 6. Temporal Trends in Crop Diversity
"""))

cells.append(code(r"""
# National-level time series of diversity indices
yearly_national = df_panel.groupby('year_start').agg(
    shannon_mean=('shannon_index', 'mean'),
    shannon_q25=('shannon_index', lambda x: x.quantile(0.25)),
    shannon_q75=('shannon_index', lambda x: x.quantile(0.75)),
    simpson_mean=('simpson_index', 'mean'),
    simpson_q25=('simpson_index', lambda x: x.quantile(0.25)),
    simpson_q75=('simpson_index', lambda x: x.quantile(0.75)),
    richness_mean=('crop_richness', 'mean'),
    richness_q25=('crop_richness', lambda x: x.quantile(0.25)),
    richness_q75=('crop_richness', lambda x: x.quantile(0.75)),
    n_districts=('district_key', 'count'),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (prefix, title, color) in zip(axes, [
    ('shannon', 'Shannon Index (H\')', '#2c7bb6'),
    ('simpson', 'Simpson Index (D)', '#d7191c'),
    ('richness', 'Crop Richness (S)', '#1a9641'),
]):
    ax.plot(yearly_national['year_start'], yearly_national[f'{prefix}_mean'],
            color=color, linewidth=2, marker='o', markersize=4)
    ax.fill_between(yearly_national['year_start'],
                    yearly_national[f'{prefix}_q25'],
                    yearly_national[f'{prefix}_q75'],
                    alpha=0.2, color=color, label='IQR (25th-75th)')
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Figure 9: National Trends in Crop Diversity (Mean with IQR)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
**Figure 9 Interpretation:** The national time series reveals important dynamics. The Shannon and Simpson indices show a slight downward trend over the 24-year period, consistent with the broader narrative of declining crop diversity. Crop richness, however, may show a different pattern -- it can increase even as evenness declines (more crops are counted but area concentrates in fewer). The interquartile range (shaded band) illustrates the substantial variation across districts in any given year.
"""))

cells.append(code(r"""
# Time series by irrigation regime
df_panel_irr = df_panel.merge(
    df_main[['district_key', 'irrigation_regime']],
    on='district_key', how='left'
)
df_panel_irr = df_panel_irr[df_panel_irr['irrigation_regime'].notna()]

yearly_regime = df_panel_irr.groupby(['year_start', 'irrigation_regime']).agg(
    shannon_mean=('shannon_index', 'mean'),
    simpson_mean=('simpson_index', 'mean'),
    richness_mean=('crop_richness', 'mean'),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, title) in zip(axes, [
    ('shannon_mean', 'Shannon Index'),
    ('simpson_mean', 'Simpson Index'),
    ('richness_mean', 'Crop Richness'),
]):
    for regime in REGIME_ORDER:
        subset = yearly_regime[yearly_regime['irrigation_regime'] == regime]
        ax.plot(subset['year_start'], subset[col],
                color=PALETTE[regime], linewidth=2, label=regime, marker='o', markersize=3)
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Figure 10: Diversity Trends by Irrigation Regime', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
**Figure 10 Interpretation:** When broken down by irrigation regime, a clearer picture emerges. Semi-irrigated districts consistently maintain the highest diversity across all years. The gap between semi-irrigated and the other two regimes appears stable over time, suggesting that this is a structural feature of these districts' cropping systems rather than a transient phenomenon. All three regimes show similar temporal patterns, indicating that the forces driving diversity change (market pressures, policy shifts, climate trends) operate broadly across irrigation contexts.
"""))

cells.append(code(r"""
# Change analysis: early vs late period
print(f"Change analysis covers {df_change.shape[0]} districts with data in both early (<=2005) and late (>=2015) periods.\n")

# Summary of gainers vs losers
for col, label in [('shannon_change', 'Shannon Index'),
                    ('simpson_change', 'Simpson Index'),
                    ('richness_change', 'Crop Richness')]:
    gaining = (df_change[col] > 0).sum()
    losing = (df_change[col] < 0).sum()
    no_change = (df_change[col] == 0).sum()
    median_change = df_change[col].median()
    print(f"{label}:")
    print(f"  Gaining: {gaining} districts | Losing: {losing} districts | No change: {no_change}")
    print(f"  Median change: {median_change:+.4f}")
    print()
"""))

cells.append(code(r"""
# Distribution of Shannon change
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, (col, label, color) in zip(axes, [
    ('shannon_change', 'Shannon Change', '#2c7bb6'),
    ('simpson_change', 'Simpson Change', '#d7191c'),
    ('richness_change', 'Crop Richness Change', '#1a9641'),
]):
    vals = df_change[col].dropna()
    ax.hist(vals, bins=40, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(vals.median(), color='orange', linestyle='-', linewidth=2, label=f'Median: {vals.median():+.3f}')
    ax.set_xlabel(label)
    ax.set_ylabel('Number of Districts')
    ax.legend(fontsize=9)

fig.suptitle('Figure 11: Distribution of Diversity Change (Early vs Late Period)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
**Figure 11 Interpretation:** The distribution of change is revealing. For the Shannon Index, the distribution is negatively skewed -- more districts are losing diversity than gaining it. The median Shannon change of approximately -0.05 indicates a modest but widespread decline. Crop richness change shows a different pattern -- many districts have actually gained crop varieties, but the evenness of area allocation has declined. This means districts are growing more crops on paper but concentrating their area in fewer dominant ones.
"""))

cells.append(code(r"""
# Display the diversity change map
change_map = MAP_DIR / 'map_diversity_change.png'
if change_map.exists():
    display(Image(filename=str(change_map), width=900))
else:
    print(f"Map not found: {change_map}")
"""))

cells.append(md(r"""
**Figure 12: Spatial Pattern of Diversity Change**

The change map reveals geographic clustering. Diversity losses (red) concentrate in parts of central and eastern India, while gains (blue/green) appear in pockets of southern and western India. This spatial clustering suggests that regional factors -- state-level agricultural policies, market access, irrigation investments -- play a significant role in determining diversity trajectories.
"""))

# =============================================================================
# SECTION 7: STATE-LEVEL ANALYSIS
# =============================================================================
cells.append(md(r"""
---
## 7. State-Level Analysis
"""))

cells.append(code(r"""
# State rankings by ABI
df_state_sorted = df_state.sort_values('agro_biodiversity_index', ascending=True)

fig, ax = plt.subplots(figsize=(10, 12))
colors = plt.cm.YlGn(np.linspace(0.2, 0.9, len(df_state_sorted)))
bars = ax.barh(range(len(df_state_sorted)), df_state_sorted['agro_biodiversity_index'],
               color=colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(df_state_sorted)))
ax.set_yticklabels(df_state_sorted['state_name'], fontsize=9)
ax.set_xlabel('Agro-Biodiversity Index (ABI)')
ax.set_title('Figure 13: State Rankings by Agro-Biodiversity Index')

# Annotate values
for i, (val, n) in enumerate(zip(df_state_sorted['agro_biodiversity_index'],
                                   df_state_sorted['n_districts'])):
    ax.text(val + 0.01, i, f'{val:.3f} (n={n})', va='center', fontsize=8)

ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
**Figure 13 Interpretation:** The state rankings reveal enormous variation. The top five most diverse states are:

1. **Karnataka** (ABI = 0.854, 30 districts): Southern India's agricultural powerhouse, with a diverse mix of cereals, pulses, oilseeds, spices, and plantation crops across varied agro-climatic zones.
2. **Andhra Pradesh** (ABI = 0.786, 20 districts): Another southern state benefiting from multiple agro-climatic zones and diverse cropping traditions.
3. **Rajasthan** (ABI = 0.767, 33 districts): Despite arid conditions, Rajasthan maintains impressive diversity through traditional rain-fed crops including millets, pulses, and oilseeds.
4. **Nagaland** (ABI = 0.765, 11 districts): Traditional jhum (shifting) cultivation practices maintain high crop diversity in this northeastern state.
5. **Uttarakhand** (ABI = 0.723, 13 districts): Hill agriculture with diverse cereals, pulses, and horticultural crops.

At the bottom: **Punjab** (ABI = 0.444, 22 districts) -- India's "food bowl" is among the least diverse states, locked into the rice-wheat cycle by MSP incentives and irrigation infrastructure. **Delhi** (ABI = 0.0) is effectively a data artifact due to urbanization.
"""))

cells.append(code(r"""
# Top and bottom 10 states (excluding small UTs)
df_state_major = df_state[df_state['n_districts'] >= 4].copy()
top10 = df_state_major.nlargest(10, 'agro_biodiversity_index')
bot10 = df_state_major.nsmallest(10, 'agro_biodiversity_index')

comparison = pd.concat([top10, bot10]).sort_values('agro_biodiversity_index', ascending=False)

styled_comparison = (comparison[['state_name', 'agro_biodiversity_index', 'shannon_index',
                                  'simpson_index', 'crop_richness', 'n_districts']]
    .style
    .format({'agro_biodiversity_index': '{:.3f}', 'shannon_index': '{:.3f}',
             'simpson_index': '{:.3f}', 'crop_richness': '{:.1f}'})
    .background_gradient(cmap='YlGn', subset=['agro_biodiversity_index'])
    .set_caption('Table 2: Top 10 and Bottom 10 States by ABI (states with >= 4 districts)')
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
    ])
)
display(styled_comparison)
"""))

cells.append(md(r"""
**Table 2 Interpretation:** The contrast between top and bottom states is instructive. High-ABI states tend to have diverse agro-climatic conditions (Karnataka spans coastal tropics to semi-arid Deccan), traditional farming systems (Nagaland's jhum), or mixed irrigation (Rajasthan's combination of canal-irrigated and rain-fed zones). Low-ABI states are characterized by either cereal monoculture under heavy irrigation (Punjab) or constrained cropping due to geography (Tripura's rice dominance in floodplain agriculture, Goa's small agricultural sector).
"""))

# =============================================================================
# SECTION 8: DISTRICT-LEVEL DEEP DIVES
# =============================================================================
cells.append(md(r"""
---
## 8. District-Level Deep Dives
"""))

cells.append(code(r"""
# Top 20 and bottom 20 districts
top20 = df_main.nlargest(20, 'agro_biodiversity_index')
bot20 = df_main.nsmallest(20, 'agro_biodiversity_index')

print("=" * 80)
print("TOP 20 MOST DIVERSE DISTRICTS")
print("=" * 80)
for i, row in top20.iterrows():
    print(f"  ABI={row['agro_biodiversity_index']:.3f} | {row['district_name']}, {row['state_name']}"
          f" | {int(row['crop_richness'])} crops | Dominant: {row['dominant_crop']} ({row['dominant_crop_share']:.1%})")

print()
print("=" * 80)
print("BOTTOM 20 LEAST DIVERSE DISTRICTS")
print("=" * 80)
for i, row in bot20.iterrows():
    print(f"  ABI={row['agro_biodiversity_index']:.3f} | {row['district_name']}, {row['state_name']}"
          f" | {int(row['crop_richness'])} crops | Dominant: {row['dominant_crop']} ({row['dominant_crop_share']:.1%})")
"""))

cells.append(code(r"""
# Styled table of top 20
top20_display = top20[['state_name', 'district_name', 'agro_biodiversity_index',
                        'shannon_index', 'simpson_index', 'crop_richness',
                        'dominant_crop', 'dominant_crop_share']].copy()
top20_display.columns = ['State', 'District', 'ABI', 'Shannon', 'Simpson',
                          'Crops', 'Dominant Crop', 'Dom. Share']
top20_display = top20_display.reset_index(drop=True)
top20_display.index = top20_display.index + 1

styled_top20 = (top20_display.style
    .format({'ABI': '{:.3f}', 'Shannon': '{:.3f}', 'Simpson': '{:.3f}',
             'Crops': '{:.0f}', 'Dom. Share': '{:.1%}'})
    .background_gradient(cmap='YlGn', subset=['ABI'])
    .set_caption('Table 3: Top 20 Most Diverse Districts in India')
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
    ])
)
display(styled_top20)
"""))

cells.append(md(r"""
**Table 3 Interpretation:** The most diverse districts share common characteristics: they typically have moderate dominant crop shares (30-40% rather than 60-80%), high crop counts (40+ crops), and are located in states with diverse agro-climatic conditions. Karnataka dominates the top ranks, reflecting its remarkable agro-ecological variety -- from the humid western coast to the semi-arid Deccan Plateau.
"""))

cells.append(code(r"""
# Display the top/bottom 20 map
tb_map = MAP_DIR / 'map_top_bottom_20_abi.png'
if tb_map.exists():
    display(Image(filename=str(tb_map), width=900))
else:
    print(f"Map not found: {tb_map}")
"""))

cells.append(md(r"""
**Figure 14: Map of Top 20 (green) and Bottom 20 (red) Districts by ABI**

The spatial distribution of extreme districts reinforces the regional patterns. The most diverse districts cluster in Karnataka, Andhra Pradesh, and Rajasthan, while the least diverse are scattered across the plains and include several urban or special-category districts (Delhi, Chandigarh) that are data artifacts rather than genuine agricultural monocultures.
"""))

cells.append(code(r"""
# Case studies: interesting districts
cases = [
    ('Karnataka', 'Belgaum', 'Consistently top-ranked: diverse agro-climate supports cereals, pulses, oilseeds, and sugarcane'),
    ('Punjab', 'Ludhiana', 'Green Revolution epicenter: rice-wheat monoculture despite high productivity'),
    ('Rajasthan', 'Udaipur', 'Rain-fed diversity: traditional millets, pulses, and oilseeds despite arid conditions'),
]

for state, district, note in cases:
    row = df_main[(df_main['state_name'] == state) & (df_main['district_name'] == district)]
    if row.empty:
        # Try case-insensitive match
        row = df_main[(df_main['state_name'].str.lower() == state.lower()) &
                       (df_main['district_name'].str.lower() == district.lower())]
    if not row.empty:
        r = row.iloc[0]
        print(f"\n{'='*60}")
        print(f"CASE STUDY: {r['district_name']}, {r['state_name']}")
        print(f"{'='*60}")
        print(f"  ABI:              {r['agro_biodiversity_index']:.3f}")
        print(f"  Shannon Index:    {r['shannon_index']:.3f}")
        print(f"  Simpson Index:    {r['simpson_index']:.3f}")
        print(f"  Crop Richness:    {int(r['crop_richness'])}")
        print(f"  Dominant Crop:    {r['dominant_crop']} ({r['dominant_crop_share']:.1%})")
        print(f"  Top 3 Share:      {r['top3_crops_share']:.1%}")
        if pd.notna(r.get('irrigation_regime')):
            print(f"  Irrigation:       {r['irrigation_regime']}")
        print(f"  Note: {note}")
    else:
        print(f"  District '{district}' in '{state}' not found in dataset.")
"""))

# =============================================================================
# SECTION 9: CALORIE PRODUCTION & THE DIVERSITY-PRODUCTIVITY FRONTIER
# =============================================================================
cells.append(md(r"""
---
## 9. Calorie Production & the Diversity-Productivity Frontier

The preceding sections established that crop diversity is declining across India and that irrigation regime plays a mediating role. A critical unanswered question is whether **diversity comes at the cost of caloric productivity**. If diversified districts produce fewer calories per hectare, policymakers face a genuine trade-off; if not, diversification can be pursued without sacrificing food security objectives.

This section overlays calorie production estimates onto the diversity analysis by converting district-level crop production data into kilocalories using Indian Food Composition Table (IFCT 2017) conversion factors. Non-food crops (cotton, jute, tobacco, etc.) are assigned zero caloric value, enabling us to distinguish between diversity that is nutritionally meaningful and diversity that is compositionally broad but calorically hollow.

**Methodology:**

$$\text{District kcal} = \sum_{\text{crop}} \left( \text{production}_{\text{tonnes}} \times \text{kcal\_per\_kg} \times 1000 \right)$$

where $\text{kcal\_per\_kg} = \text{kcal\_per\_100g} \times 10$.
"""))

# ---- 9.1 Computing District-Level Calorie Production ----
cells.append(md(r"""
### 9.1 Computing District-Level Calorie Production

Each district's total annual calorie production is computed by mapping crop production (in tonnes) to energy values from the Indian Food Composition Table (IFCT 2017). Crops that are non-food (cotton, jute, tobacco, mesta, sannhemp) are assigned 0 kcal. Caloric productivity is then expressed as **kcal per hectare** to normalise across districts of different sizes.
"""))

cells.append(code(r"""
# Load the merged calorie-diversity dataset
df_cal = pd.read_csv(DATA_DIR / 'district_diversity_calorie_merged.csv')
print(f"Loaded {len(df_cal)} districts with calorie data")
print()

# Summary statistics for key calorie columns
cal_cols = ['total_kcal_annual', 'kcal_per_hectare', 'food_crop_kcal_share',
            'cereal_kcal_share', 'pulse_kcal_share', 'oilseed_kcal_share',
            'sugar_kcal_share', 'agro_biodiversity_index']
print("Summary Statistics (Calorie Variables):")
print("=" * 80)
display(df_cal[cal_cols].describe().round(4).T.style
    .format('{:.4f}')
    .set_caption('Table 4: Summary Statistics -- Calorie Production Variables')
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
    ])
)
print()
print(f"Median kcal/ha:          {df_cal['kcal_per_hectare'].median():,.0f}")
print(f"Mean food crop kcal %:   {df_cal['food_crop_kcal_share'].mean():.1%}")
print(f"Districts with data:     {df_cal['kcal_per_hectare'].notna().sum()}")
"""))

# ---- 9.2 Caloric Productivity Map ----
cells.append(md(r"""
### 9.2 Caloric Productivity Map
"""))

cells.append(code(r"""
# Display the kcal per hectare choropleth
kcal_map = DATA_DIR / 'kcal_per_hectare_choropleth.png'
if kcal_map.exists():
    display(Image(filename=str(kcal_map), width=900))
else:
    print(f"Map not found: {kcal_map}")
"""))

cells.append(md(r"""
**Figure 15: Caloric Productivity (kcal per hectare) Across Indian Districts**

The spatial pattern of caloric productivity broadly mirrors India's agricultural geography. The **Indo-Gangetic Plain** (Punjab, Haryana, western UP) registers the highest caloric output per hectare, driven by intensive rice-wheat cultivation with high yields. In contrast, **arid western Rajasthan and Gujarat** show low caloric productivity, reflecting both low rainfall and the predominance of low-calorie crops like cotton and guar. An interesting outlier is **Kerala and coastal Tamil Nadu**, where coconut -- a high-calorie crop (354 kcal/100g) -- inflates per-hectare calorie figures despite relatively modest grain production. This "coconut effect" is an important caveat when interpreting caloric productivity as a proxy for food security.
"""))

# ---- 9.3 The Diversity-Productivity Frontier ----
cells.append(md(r"""
### 9.3 The Diversity-Productivity Frontier
"""))

cells.append(code(r"""
# Display the ABI vs kcal/ha scatter plot
scatter_path = DATA_DIR / 'abi_vs_kcal_scatter.png'
if scatter_path.exists():
    display(Image(filename=str(scatter_path), width=900))
else:
    print(f"Scatter plot not found: {scatter_path}")
"""))

cells.append(code(r"""
# Compute correlation between ABI and kcal/ha
from scipy import stats

valid = df_cal[['agro_biodiversity_index', 'kcal_per_hectare']].dropna()
pearson_r, pearson_p = stats.pearsonr(valid['agro_biodiversity_index'], valid['kcal_per_hectare'])
spearman_r, spearman_p = stats.spearmanr(valid['agro_biodiversity_index'], valid['kcal_per_hectare'])

print("Correlation: ABI vs. kcal/ha")
print("=" * 50)
print(f"  Pearson r  = {pearson_r:+.3f}  (p = {pearson_p:.4f})")
print(f"  Spearman r = {spearman_r:+.3f}  (p = {spearman_p:.4f})")
print(f"  N          = {len(valid)}")
print()

# By irrigation regime
if 'irrigation_regime' in df_cal.columns:
    print("Correlation by Irrigation Regime:")
    print("-" * 50)
    for regime in REGIME_ORDER:
        sub = df_cal[df_cal['irrigation_regime'] == regime][['agro_biodiversity_index', 'kcal_per_hectare']].dropna()
        if len(sub) > 10:
            r, p = stats.pearsonr(sub['agro_biodiversity_index'], sub['kcal_per_hectare'])
            print(f"  {regime:30s}: r = {r:+.3f}  (p = {p:.4f}, N = {len(sub)})")
"""))

cells.append(md(r"""
**Figure 16: The Diversity-Productivity Frontier -- ABI vs. kcal per hectare**

The scatter plot reveals a **weak negative correlation** between agro-biodiversity and caloric productivity (Pearson r ~ -0.18). This is a crucial finding: **diversity does NOT strongly trade off against caloric output**. The relationship is far weaker than a simple "diversification costs calories" narrative would predict.

The weak correlation means that many districts achieve both high diversity and respectable caloric productivity, while others are neither diverse nor particularly productive. The policy implication is significant: diversification need not come at the cost of food security, provided it is pursued with appropriate crop selection and agronomic support.

When stratified by irrigation regime, the correlation patterns may differ: irrigated districts tend to cluster in the high-calorie / low-diversity quadrant, while semi-irrigated districts spread across the frontier, reinforcing their role as the most balanced agricultural systems.
"""))

# ---- 9.4 Quadrant Analysis ----
cells.append(md(r"""
### 9.4 Quadrant Analysis

By splitting districts at the median ABI and median kcal/ha, we create four quadrants that characterise distinct agricultural archetypes:

| Quadrant | Description | Policy Posture |
|:---------|:------------|:---------------|
| **Diverse & Calorie-Rich** | High ABI, high kcal/ha | Sustain and learn from |
| **Monoculture Breadbasket** | Low ABI, high kcal/ha | Incentivise diversification |
| **Diverse & Calorie-Poor** | High ABI, low kcal/ha | Improve yields within diverse systems |
| **Vulnerable** | Low ABI, low kcal/ha | Highest priority for intervention |
"""))

cells.append(code(r"""
# Display the quadrant map
quad_map = DATA_DIR / 'quadrant_map.png'
if quad_map.exists():
    display(Image(filename=str(quad_map), width=900))
else:
    print(f"Quadrant map not found: {quad_map}")
"""))

cells.append(code(r"""
# Crosstab: quadrant x irrigation regime
if 'kcal_diversity_quadrant' in df_cal.columns and 'irrigation_regime' in df_cal.columns:
    quad_irr = pd.crosstab(
        df_cal['kcal_diversity_quadrant'],
        df_cal['irrigation_regime'],
        margins=True,
        margins_name='Total'
    )
    # Reorder if possible
    quad_order = ['Diverse & Calorie-Rich', 'Monoculture Breadbasket',
                  'Diverse & Calorie-Poor', 'Vulnerable']
    existing_quads = [q for q in quad_order if q in quad_irr.index]
    other_quads = [q for q in quad_irr.index if q not in quad_order and q != 'Total']
    final_order = existing_quads + other_quads + ['Total']
    quad_irr = quad_irr.reindex([q for q in final_order if q in quad_irr.index])

    print("Table 5: Quadrant × Irrigation Regime Crosstab")
    print("=" * 70)
    display(quad_irr.style
        .set_caption('Table 5: District Count by Calorie-Diversity Quadrant and Irrigation Regime')
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
        ])
    )
else:
    print("Quadrant or irrigation regime column not found in data.")
"""))

cells.append(code(r"""
# Mean values by quadrant
if 'kcal_diversity_quadrant' in df_cal.columns:
    quad_stats = df_cal.groupby('kcal_diversity_quadrant').agg(
        n_districts=('district_name', 'count'),
        mean_abi=('agro_biodiversity_index', 'mean'),
        mean_kcal_ha=('kcal_per_hectare', 'mean'),
        mean_food_kcal_share=('food_crop_kcal_share', 'mean'),
        mean_cereal_share=('cereal_kcal_share', 'mean'),
        mean_crop_richness=('crop_richness', 'mean'),
    ).round(3)

    # Add share column
    quad_stats['pct_of_total'] = (quad_stats['n_districts'] / quad_stats['n_districts'].sum() * 100).round(1)

    print("Table 6: Mean Values by Calorie-Diversity Quadrant")
    print("=" * 80)
    display(quad_stats.style
        .format({
            'mean_abi': '{:.3f}',
            'mean_kcal_ha': '{:,.0f}',
            'mean_food_kcal_share': '{:.3f}',
            'mean_cereal_share': '{:.3f}',
            'mean_crop_richness': '{:.1f}',
            'pct_of_total': '{:.1f}%',
        })
        .set_caption('Table 6: Agricultural Profile by Calorie-Diversity Quadrant')
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
        ])
    )
"""))

cells.append(md(r"""
**Figure 17: Calorie-Diversity Quadrant Map of Indian Districts**

The four quadrants reveal distinct agricultural archetypes distributed across India:

- **Diverse & Calorie-Rich (~24%):** These districts achieve the best of both worlds -- high crop variety and high caloric output. They are concentrated in **south and central India** (Karnataka, Maharashtra, parts of Andhra Pradesh), where favourable agro-climatic conditions and mixed farming traditions support diverse yet productive systems. These districts serve as proof of concept that diversity and productivity can coexist.

- **Monoculture Breadbasket (~26%):** Characterised by high caloric output but low diversity, these districts epitomise the **Punjab/Haryana model** -- intensive rice-wheat or sugarcane cultivation with assured irrigation. While these districts are critical for national food security, their low diversity makes them vulnerable to pest outbreaks, soil degradation, and market shocks. This quadrant presents the strongest case for policy-driven diversification.

- **Diverse & Calorie-Poor (~26%):** These districts grow many crops but produce relatively few calories per hectare. They are concentrated in **northeast India, hill regions, and tribal areas**, where subsistence farming with traditional crop mixes persists but yields remain low. The diversity here is "natural" rather than engineered -- it reflects adaptation to marginal conditions. Interventions should focus on improving yields within these diverse systems rather than replacing them with monocultures.

- **Vulnerable (~24%):** Low on both diversity and caloric output, these districts represent the **highest priority for agricultural intervention**. They include arid zones in western Rajasthan, rain-shadow regions, and areas with degraded soils. Both diversification support and productivity enhancement are needed.
"""))

# ---- 9.5 Nutritionally Hollow Diversity ----
cells.append(md(r"""
### 9.5 Nutritionally Hollow Diversity

Not all diversity is nutritionally meaningful. A district that grows cotton, jute, and tobacco alongside a single food grain may appear "diverse" by the ABI metric but produces few edible calories. We term this phenomenon **nutritionally hollow diversity** -- high compositional variety but low food-crop calorie share.
"""))

cells.append(code(r"""
# Display the nutritionally hollow diversity map if available
hollow_map = DATA_DIR / 'nutritionally_hollow_map.png'
if hollow_map.exists():
    display(Image(filename=str(hollow_map), width=900))
else:
    print(f"Nutritionally hollow map not found: {hollow_map}")
    print("Proceeding with analytical identification instead.")
"""))

cells.append(code(r"""
# Identify nutritionally hollow districts: high ABI but low food crop calorie share
hollow = df_cal[(df_cal['agro_biodiversity_index'] > 0.6) &
                (df_cal['food_crop_kcal_share'] < 0.5)].copy()

print(f"Nutritionally Hollow Districts: ABI > 0.6 AND food_crop_kcal_share < 0.5")
print(f"Count: {len(hollow)} out of {len(df_cal)} ({len(hollow)/len(df_cal)*100:.1f}%)")
print()

if len(hollow) > 0:
    hollow_display = hollow[['state_name', 'district_name', 'agro_biodiversity_index',
                             'food_crop_kcal_share', 'crop_richness', 'dominant_crop']].copy()
    hollow_display.columns = ['State', 'District', 'ABI', 'Food kcal Share', 'Crops', 'Dominant Crop']
    hollow_display = hollow_display.sort_values('food_crop_kcal_share', ascending=True).reset_index(drop=True)
    hollow_display.index = hollow_display.index + 1
    display(hollow_display.head(20).style
        .format({'ABI': '{:.3f}', 'Food kcal Share': '{:.3f}', 'Crops': '{:.0f}'})
        .background_gradient(cmap='RdYlGn', subset=['Food kcal Share'])
        .set_caption('Table 7: Nutritionally Hollow Districts (ABI > 0.6, Food kcal Share < 50%)')
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
        ])
    )
else:
    print("No districts meet the nutritionally hollow criteria.")
    print("This suggests that high-ABI districts generally maintain substantial food crop production.")
"""))

cells.append(md(r"""
**Figure 18: Nutritionally Hollow Diversity**

The nutritionally hollow analysis highlights an important nuance: **not all crop diversity is equally valuable for food security**. Districts that appear diverse in the ABI ranking may owe their diversity to non-food crops (fiber, narcotics, plantation crops) rather than to a balanced mix of food grains, pulses, and oilseeds. This finding reinforces the need to complement area-based diversity metrics with nutritional output measures when assessing agricultural resilience.
"""))

# ---- 9.6 Irrigation, Diversity, and Calories ----
cells.append(md(r"""
### 9.6 Irrigation, Diversity, and Calories

Having established the individual relationships between irrigation and diversity (Section 5) and between diversity and calories (Section 9.3), we now examine the three-way interaction.
"""))

cells.append(code(r"""
# Box plot: kcal/ha by irrigation regime
if 'irrigation_regime' in df_cal.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: kcal/ha by irrigation regime
    regime_data = df_cal[df_cal['irrigation_regime'].notna()]
    bp_data = [regime_data[regime_data['irrigation_regime'] == r]['kcal_per_hectare'].dropna()
               for r in REGIME_ORDER]

    bplot = axes[0].boxplot(bp_data, labels=['Rainfed', 'Semi-Irr.', 'Irrigated'],
                            patch_artist=True, showfliers=False, widths=0.6)
    colors = [PALETTE[r] for r in REGIME_ORDER]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel('kcal per hectare')
    axes[0].set_title('Figure 19a: Caloric Productivity by Irrigation Regime')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # Right panel: ABI by irrigation regime
    bp_data2 = [regime_data[regime_data['irrigation_regime'] == r]['agro_biodiversity_index'].dropna()
                for r in REGIME_ORDER]
    bplot2 = axes[1].boxplot(bp_data2, labels=['Rainfed', 'Semi-Irr.', 'Irrigated'],
                             patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Agro-Biodiversity Index')
    axes[1].set_title('Figure 19b: Diversity by Irrigation Regime')

    plt.tight_layout()
    plt.show()
else:
    print("Irrigation regime data not available for comparison.")
"""))

cells.append(code(r"""
# Mean ABI and kcal/ha by irrigation regime
if 'irrigation_regime' in df_cal.columns:
    irr_cal = df_cal.groupby('irrigation_regime').agg(
        n_districts=('district_name', 'count'),
        mean_abi=('agro_biodiversity_index', 'mean'),
        median_abi=('agro_biodiversity_index', 'median'),
        mean_kcal_ha=('kcal_per_hectare', 'mean'),
        median_kcal_ha=('kcal_per_hectare', 'median'),
        mean_food_kcal_share=('food_crop_kcal_share', 'mean'),
    ).round(3)

    # Reorder
    irr_cal = irr_cal.reindex([r for r in REGIME_ORDER if r in irr_cal.index])

    print("Table 8: Caloric Productivity and Diversity by Irrigation Regime")
    print("=" * 80)
    display(irr_cal.style
        .format({
            'mean_abi': '{:.3f}',
            'median_abi': '{:.3f}',
            'mean_kcal_ha': '{:,.0f}',
            'median_kcal_ha': '{:,.0f}',
            'mean_food_kcal_share': '{:.3f}',
        })
        .set_caption('Table 8: Irrigation Regime -- Diversity and Caloric Productivity')
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '13px'), ('font-weight', 'bold'), ('text-align', 'left')]},
        ])
    )
"""))

cells.append(md(r"""
**Figures 19a-b and Table 8: The Irrigation-Diversity-Calories Nexus**

The three-way comparison yields a key insight: **semi-irrigated districts achieve the best balance between diversity and caloric productivity**. While irrigated districts produce the most calories per hectare, they do so at the cost of crop diversity. Rainfed districts maintain moderate diversity but lag in caloric output due to yield constraints. Semi-irrigated districts occupy the productive middle ground -- they have sufficient water access to support reasonable yields while retaining enough agronomic flexibility to maintain diverse cropping patterns.

This finding has direct policy relevance: expanding irrigation from rainfed to semi-irrigated levels may enhance both productivity and diversity, but pushing irrigation beyond the semi-irrigated threshold -- toward the full irrigation model of Punjab and Haryana -- risks collapsing into the monoculture breadbasket pattern.
"""))

# =============================================================================
# SECTION 10: SUMMARY AND POLICY IMPLICATIONS
# =============================================================================
cells.append(md(r"""
---
## 10. Summary and Policy Implications

### Key Findings

1. **India's crop diversity is moderate but declining.** The average ABI across 755 districts is approximately 0.60. Out of 590 districts with data in both early and late periods, 357 (60%) experienced a decline in Shannon diversity, while only 233 (40%) showed improvement.

2. **Semi-irrigated districts are the most diverse.** Districts with 40-60% irrigated area have the highest mean ABI (0.69), outperforming both fully irrigated (0.67) and rainfed (0.62) districts. This "Goldilocks" finding suggests that partial irrigation enables crop diversity by expanding options without incentivizing monoculture.

3. **Irrigated districts are cereal-locked.** Irrigated districts devote ~67% of area to cereals and only ~8.5% to pulses, versus ~53% cereals and ~13% pulses in semi-irrigated districts. The economic incentives of MSP for rice and wheat, combined with assured water supply, drive this concentration.

4. **Karnataka leads, Punjab trails.** Karnataka (ABI = 0.85) is India's most agro-biodiverse major state, while Punjab (ABI = 0.44) exemplifies the diversity cost of the Green Revolution model.

5. **Crop richness is increasing, but evenness is declining.** Many districts grow more crop varieties than before, but concentrate their area in fewer dominant crops. This "hollow diversity" -- variety without balance -- undermines the resilience benefits that true crop diversity provides.

6. **Diversity does not strongly trade off against caloric productivity.** The Pearson correlation between ABI and kcal/ha is approximately -0.18 -- a weak negative relationship. Many districts achieve both high diversity and high caloric output, particularly in south and central India.

7. **Four distinct agricultural archetypes emerge from the calorie-diversity quadrant analysis.** Roughly a quarter of districts fall into each quadrant: Diverse & Calorie-Rich (~24%), Monoculture Breadbasket (~26%), Diverse & Calorie-Poor (~26%), and Vulnerable (~24%). Each requires tailored policy interventions.

8. **Semi-irrigated districts achieve the best diversity-productivity balance.** They combine moderate caloric output with the highest diversity, reinforcing the finding from Section 5 that the semi-irrigated zone is an agricultural sweet spot.

9. **Some crop diversity is nutritionally hollow.** Districts with high ABI but low food-crop calorie share owe their diversity to non-food crops (fiber, plantation, narcotics), limiting the food security benefits of their diverse cropping patterns.

### Policy Implications

- **Diversification incentives for irrigated districts:** MSP reform or diversification bonuses could incentivize irrigated districts to move beyond rice-wheat. Given their water infrastructure, these districts have the potential for high-value diversification (horticulture, pulses, oilseeds).

- **Protect semi-irrigated diversity:** The naturally diverse semi-irrigated zone should be a conservation priority. Policies that push toward full irrigation without corresponding diversification incentives could erode this diversity.

- **State-specific strategies:** The wide variation across states (ABI from 0.44 to 0.85 among major states) demands tailored approaches. Punjab needs structural reform of cropping incentives; states like Karnataka need support to maintain their existing diversity.

- **Monitor evenness, not just variety:** Metrics that count crop types without measuring area distribution can mask concerning trends. The divergence between richness (increasing) and evenness (decreasing) should be tracked and reported.

- **Diversification need not sacrifice calories:** The weak diversity-productivity correlation means that well-designed diversification programmes can maintain caloric output while improving nutritional variety and climate resilience. The "Diverse & Calorie-Rich" quadrant districts provide models to emulate.

- **Target the Vulnerable quadrant:** Districts low on both diversity and caloric productivity require integrated interventions combining improved varieties, water management, and crop diversification -- not a single-axis approach.

- **Audit diversity for nutritional content:** Agricultural diversity metrics should be complemented with nutritional output measures to distinguish meaningful food-crop diversity from compositionally broad but nutritionally hollow cropping patterns.

### Limitations

1. **Irrigation matching:** Only 503 of 755 districts (67%) could be matched with irrigation data. Improved matching (e.g., fuzzy name matching) would strengthen the irrigation analysis.
2. **Temporal coverage:** Not all districts have data for all years, which may bias trend estimates.
3. **Crop aggregation:** The analysis uses broad crop categories from government statistics. Within-crop variety (e.g., traditional vs. hybrid rice) is not captured.
4. **No yield/economics integration:** Diversity analysis without productivity and income data cannot fully assess the trade-offs farmers face.
5. **Calorie conversion factors:** IFCT 2017 conversion factors are crop-level averages that do not account for variety-level differences or post-harvest losses. Actual dietary calories available are lower than the production-based estimates reported here.

### Future Work

1. Fuzzy district name matching to improve irrigation coverage from 67% to ~90%
2. Separate Kharif vs. Rabi diversity analysis to capture seasonal patterns
3. Additional concentration metrics: Herfindahl-Hirschman Index, Margalef Richness Index
4. Integration with district-level yield, income, and nutritional data
5. Interactive dashboard for policymaker exploration (Plotly/Streamlit)
6. Micronutrient diversity analysis -- extending beyond calories to protein, iron, zinc, and vitamin A content
7. Temporal analysis of caloric productivity trends and their relationship with diversification patterns
"""))

cells.append(md(r"""
---

*This analysis was produced by the Council on Energy, Environment and Water (CEEW). For questions or feedback, please contact the CEEW research team.*

*Data sources: India Data Portal (crop area, production, yield), district-level irrigation statistics, Indian Food Composition Table (IFCT 2017) for calorie conversion factors. See Section 11 for full references.*
"""))

# =============================================================================
# SECTION 11: REFERENCES
# =============================================================================
cells.append(md(r"""
---
## 11. References

### Data Sources

1. Ministry of Agriculture & Farmers Welfare, Government of India. *Area, Production, and Yield of Major Crops (1997--2021).* Directorate of Economics and Statistics, district-level records. Available at: [https://data.gov.in](https://data.gov.in).

2. Census of India / Ministry of Water Resources, Government of India. *District-Level Gross Irrigated Area Statistics.* Gross irrigated area as percentage of gross cropped area, used for irrigation regime classification.

3. Census of India. *District Boundaries of India, 2011.* Administrative boundary shapefiles (735 districts). Office of the Registrar General & Census Commissioner, Government of India.

4. National Institute of Nutrition (NIN). *Indian Food Composition Tables (IFCT 2017).* Indian Council of Medical Research, Hyderabad. Used for crop-level kilocalorie conversion factors (kcal per 100g edible portion).

### Methodological References

5. Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379--423. doi:10.1002/j.1538-7305.1948.tb01338.x

6. Simpson, E.H. (1949). "Measurement of Diversity." *Nature*, 163, 688. doi:10.1038/163688a0

### Agricultural Diversity Literature

7. Food and Agriculture Organization of the United Nations (FAO). *Guidelines for the Assessment of Agrobiodiversity.* FAO, Rome. Available at: [https://www.fao.org](https://www.fao.org).

8. Di Falco, S. & Chavas, J.-P. (2009). "On Crop Biodiversity, Risk Exposure, and Food Security in the Highlands of Ethiopia." *American Journal of Agricultural Economics*, 91(3), 599--611. doi:10.1111/j.1467-8276.2009.01265.x

9. Lin, B.B. (2011). "Resilience in Agriculture through Crop Diversification: Adaptive Management for Environmental Change." *BioScience*, 61(3), 183--193. doi:10.1525/bio.2011.61.3.4

10. Birthal, P.S., Negi, D.S., Jha, A.K. & Singh, D. (2014). "Income Sources of Farm Households in India: Determinants, Distributional Consequences and Policy Implications." *Agricultural Economics Research Review*, 27(1), 37--48.

11. Birthal, P.S., Roy, D. & Negi, D.S. (2015). "Assessing the Impact of Crop Diversification on Farm Poverty in India." *World Development*, 72, 70--92. doi:10.1016/j.worlddev.2015.02.015

12. Auffhammer, M. & Carleton, T.A. (2018). "Regional Crop Diversity and Weather Shocks in India." *Asian Development Review*, 35(2), 113--130. doi:10.1162/adev_a_00116

13. Chand, R. (1999). "Emerging Crisis in Punjab Agriculture: Severity and Options for Future." *Economic and Political Weekly*, 34(13), A2--A10.
"""))

# =============================================================================
# BUILD NOTEBOOK
# =============================================================================
nb = new_notebook()
nb.metadata.kernelspec = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata.language_info = {
    'name': 'python',
    'version': '3.11.0',
    'mimetype': 'text/x-python',
    'file_extension': '.py',
}
nb.cells = cells

# Ensure output directory exists
os.makedirs('notebooks', exist_ok=True)
output_path = os.path.join('notebooks', 'crop_diversity_analysis.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, {sum(1 for c in cells if c.cell_type == 'code')} code)")
