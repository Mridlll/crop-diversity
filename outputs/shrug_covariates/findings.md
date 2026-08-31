# Irrigation and agrobiodiversity, rebuilt on SHRUG

Analysis set: 697 districts across 30 states.
155 of them sit on a shared pre-2011 parent district.

## 1. The headline finding, old measure and new

| measure | rainfed ABI | n | semi ABI | n | irrigated ABI | n | inverted U? |
|---|---|---|---|---|---|---|---|
| scraped irrigation_pct (original) | 0.639 | 217 | 0.684 | 104 | 0.667 | 166 | yes |
| SHRUG Antyodaya 2019 | 0.611 | 188 | 0.652 | 238 | 0.656 | 271 | NO |
| SHRUG Antyodaya, parent-weighted | 0.615 | 188 | 0.671 | 238 | 0.663 | 271 | yes |
| SHRUG Antyodaya, drop shared parents | 0.622 | 152 | 0.690 | 169 | 0.669 | 221 | yes |
| PC11 Village Directory 2011 | 0.632 | 306 | 0.692 | 122 | 0.637 | 228 | yes |

## 2. Shape of the relationship, without imposing cut-points

| decile of irrigation | n | mean irrigation | ABI | Shannon | richness | cereal share |
|---|---|---|---|---|---|---|
| 1 | 70 | 0.166 | 0.574 | 1.369 | 27.9 | 0.720 |
| 2 | 70 | 0.309 | 0.634 | 1.514 | 32.1 | 0.667 |
| 3 | 69 | 0.390 | 0.642 | 1.595 | 30.3 | 0.596 |
| 4 | 70 | 0.446 | 0.675 | 1.710 | 31.7 | 0.609 |
| 5 | 71 | 0.502 | 0.592 | 1.465 | 27.2 | 0.568 |
| 6 | 68 | 0.554 | 0.676 | 1.659 | 32.9 | 0.604 |
| 7 | 70 | 0.631 | 0.677 | 1.685 | 30.2 | 0.562 |
| 8 | 69 | 0.701 | 0.689 | 1.593 | 34.3 | 0.608 |
| 9 | 70 | 0.812 | 0.670 | 1.526 | 33.3 | 0.741 |
| 10 | 70 | 0.922 | 0.595 | 1.307 | 28.4 | 0.787 |

Linear:    ABI = 0.6143 +0.0512 x irrigation   (p = 0.0476, R2 = 0.005)
Quadratic: linear term +0.6020 (p=0.0000), squared term -0.5017 (p=0.0000), R2 = 0.035
Turning point at irrigation = 0.600 (concave, so an interior maximum).

With state fixed effects: linear +0.3242 (p=0.0056), squared -0.4155 (p=0.0001).

## 3. Irrigation source, holding the level of irrigation constant

Village-share of each dominant irrigation source, mean across districts:
canal 0.170, groundwater 0.507, surface 0.115, other 0.207.

| canal quartile | n | irrigation | ABI | cereal share | pulse share | richness |
|---|---|---|---|---|---|---|
| Q1 least canal | 175 | 0.540 | 0.639 | 0.653 | 0.091 | 31.0 |
| Q2 | 174 | 0.532 | 0.641 | 0.625 | 0.094 | 31.5 |
| Q3 | 175 | 0.515 | 0.669 | 0.647 | 0.103 | 31.2 |
| Q4 most canal | 173 | 0.587 | 0.620 | 0.660 | 0.103 | 29.6 |

State fixed effects, groundwater is the omitted source, SEs clustered on parent district.

| outcome | n | irrigation level | canal (vs groundwater) | surface (vs groundwater) | R2 |
|---|---|---|---|---|---|
| agro_biodiversity_index | 697 | -0.109*** | -0.038 | -0.219*** | 0.490 |
| shannon_index | 697 | -0.595*** | -0.010 | -0.772*** | 0.456 |
| crop_richness | 697 | +2.070 | -4.350*** | -1.526 | 0.598 |
| share_cereals | 697 | -0.034 | +0.025 | +0.237*** | 0.578 |
| share_pulses | 697 | -0.133*** | +0.076*** | -0.128*** | 0.387 |

Significance: *** p<0.01, ** p<0.05, * p<0.10.

## 4. The other four dimensions

Outcome is the Agro-Biodiversity Index. `raw` is bivariate. `adjusted` adds
irrigation share, log mean holding size and state fixed effects.

| dimension | n | raw | adjusted |
|---|---|---|---|
| irrigation share | 697 | +0.051** | -0.091** |
| canal village share | 697 | -0.070** | -0.029 |
| mean holding (log ha) | 697 | -0.007 | -0.002 |
| cultivator share of ag workers | 697 | -0.109*** | -0.073* |
| landless share (SECC) | - | - | - |
| mandi village share | 697 | -0.241** | +0.024 |
| weekly haat village share | 697 | -0.011 | +0.022 |
| regular market village share | 697 | -0.165*** | -0.112** |
| FPO village share | 697 | +0.060* | +0.122** |
| cold storage village share | 697 | +0.270*** | +0.026 |
| SC population share | 697 | +0.245*** | +0.207** |
| ST population share | 697 | -0.055** | +0.020 |
| organic farmer share | 697 | +0.019 | -0.053 |
| cropping intensity | 697 | +0.129*** | +0.189*** |