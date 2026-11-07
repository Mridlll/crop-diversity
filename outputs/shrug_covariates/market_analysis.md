# The market layer: mandis, haats and fertiliser shops

## Status after referee (Nov 2026)

Written 7 November 2026. Site basis since script 112: population shares (version W, the share of a district's 2011 rural population living in a village with the facility). This note uses village shares (version U). Sources: `outputs/revision_nov8/market_pop_weighted_results.md`, `market_pw_coefs.csv`, `market_pw_robust_haat.csv`, `market_pw_interaction.csv`; state clustering in `outputs/revision_nov8/irrigation_v2/irrigation_v2_results.md`.

- Section D opening, "Diversity peaks near 37 percent irrigation and falls after": **wrong and out of date**. The 37 percent is an ABI-era turning point. The current result is flat-then-falling: within state, the effective number of crops is flat across the less irrigated districts and falls once irrigation passes about half of sown area (fifths +0.12, -0.55, -1.05, -2.14 against the least irrigated fifth; knot 0.30, about -0.4 crops per 10 points above it). Corrected in place below.
- Weekly haat on the count of crops, U +7.365 -> **W +6.276 (1.231)**, p < 0.001; six adjusted versions +5.22 to +6.91; one SD of haat share (0.15) is about 0.9 crops on a mean of 21.4. State-clustered p 0.000 to 0.007 in every window. With the six crops removed for the 2020-22 extension it falls to +4.61 to +4.86. Survives. Haat on D1: W +0.553 (0.717), p 0.441; on evenness W **-0.046 (0.028), p 0.103**, so the negative evenness reading is not statistically significant on population shares.
- FPO, "lift D1 without lifting D0": on population shares FPOs go with both. **D1 W +1.886 (0.790), p 0.017; D0 W +3.380 (1.494), p 0.024**. State-clustered D1 p 0.156 (site window), 0.110 and 0.094 in the recent windows. Refereed-weakened.
- Fertiliser shops on pulse share, U -0.172 -> **W -0.123 (0.035)**; cereal share W -0.216 (0.073), p 0.003. State-clustered pulse p 0.027 over the full period, 0.104 and 0.108 in the recent windows. Refereed-weakened.
- Mandis on cereal share, U -0.235 (p 0.054) -> **W -0.153 (0.112), p 0.172**. The opening hypothesis (more mandis, more cereals) remains refuted.
- Section D mandi interaction, U p 0.030 -> **W: curvature -8.21 at low mandi density against -4.06 at high, difference p 0.060**; fertiliser shop p 0.95. Exploratory; significance depends on weighting.
- Reweighting changed 31 signs and 20 calls at 5 percent across 280 cells; the headline claims above are the ones that moved.

The original analysis below is kept unchanged for the record, apart from the sentences marked [corrected Nov 2026]. Where it disagrees with this block, this block and `outputs/ANALYSIS_INDEX.md` hold.

---

Analysis set: 606 districts, 30 states.
Crop-category outcomes available: cereals, drugs_and_narcotics, fiber_crops, fodder, fruits, oilseeds, pulses, spices, sugar, vegetable.

Diversity outcomes are the Hill numbers, not the ABI: D0 is the count of crops,
D1 (exp of Shannon) and D2 (inverse Simpson) are effective counts weighted toward
commoner crops, and evenness is D1/D0. All are in units of crops, so a coefficient
reads as a change in the number of crops. The ABI is not used here at all.

Adjusted models control for irrigation and its square, log mean holding size,
ST population share, log nightlights, non-farm establishment density, a
connectivity index and state fixed effects. Errors cluster on the pre-2011
parent district.

## A. What exists where

Share of a district's villages having each facility.

| facility | mean | p10 | p90 | districts with none |
|---|---|---|---|---|
| mandi | 0.030 | 0.003 | 0.063 | 40 |
| regular market | 0.098 | 0.009 | 0.214 | 6 |
| weekly haat | 0.143 | 0.009 | 0.303 | 30 |
| fertiliser shop | 0.175 | 0.033 | 0.398 | 15 |
| seed centre | 0.098 | 0.024 | 0.203 | 14 |
| soil testing | 0.043 | 0.006 | 0.075 | 39 |
| custom hiring | 0.092 | 0.014 | 0.227 | 13 |
| cold storage | 0.084 | 0.019 | 0.171 | 14 |
| farm-gate processing | 0.114 | 0.028 | 0.194 | 13 |
| FPO | 0.231 | 0.056 | 0.485 | 5 |

Mandis are rare: the median district has one in 2 to 3 villages per hundred, and
40 districts report none at all. Haats are five times commoner. That asymmetry is
the point of the exercise rather than a nuisance.

How the facilities correlate with each other:

```
                      mandi  regular m  weekly ha  fertilise  seed cent  soil test  custom hi  cold stor  farm-gate   FPO
mandi                  1.00       0.34      -0.15       0.43       0.47       0.41       0.46       0.33       0.44  0.46
regular market         0.34       1.00      -0.21       0.72       0.57       0.53       0.65       0.24       0.67  0.55
weekly haat           -0.15      -0.21       1.00      -0.02      -0.10      -0.04      -0.09       0.06      -0.09  0.05
fertiliser shop        0.43       0.72      -0.02       1.00       0.70       0.65       0.80       0.42       0.74  0.76
seed centre            0.47       0.57      -0.10       0.70       1.00       0.82       0.76       0.56       0.73  0.69
soil testing           0.41       0.53      -0.04       0.65       0.82       1.00       0.73       0.54       0.70  0.63
custom hiring          0.46       0.65      -0.09       0.80       0.76       0.73       1.00       0.55       0.82  0.83
cold storage           0.33       0.24       0.06       0.42       0.56       0.54       0.55       1.00       0.58  0.54
farm-gate processing   0.44       0.67      -0.09       0.74       0.73       0.70       0.82       0.58       1.00  0.74
FPO                    0.46       0.55       0.05       0.76       0.69       0.63       0.83       0.54       0.74  1.00
```

Correlation of each with the development controls:

| facility | log nightlights | establishments per 1000 | connectivity |
|---|---|---|---|
| mandi | +0.16 | +0.18 | +0.30 |
| regular market | +0.10 | +0.33 | +0.54 |
| weekly haat | +0.04 | -0.16 | +0.07 |
| fertiliser shop | +0.20 | +0.35 | +0.74 |
| seed centre | +0.11 | +0.21 | +0.53 |
| soil testing | +0.10 | +0.14 | +0.49 |
| custom hiring | +0.12 | +0.32 | +0.63 |
| cold storage | +0.04 | +0.03 | +0.33 |
| farm-gate processing | +0.12 | +0.23 | +0.55 |
| FPO | +0.17 | +0.33 | +0.70 |

Everything correlates with development, which is exactly why the adjusted models
carry nightlights, establishment density and connectivity.

## B. Output-market type and what a district grows

Three output-market variables entered together, so each is read against the
other two. Outcomes are the four Hill measures plus every crop-category share.

Read D0 against D1. D0 counts crops; D1 counts *effective* crops, so a crop
grown on a sliver of land barely moves it. A facility that lifts D0 but not D1
is associated with more crops grown at the margin rather than a genuinely more
balanced cropping pattern. That distinction is invisible in a composite index.

| outcome | mandi | weekly haat | regular market |
|---|---|---|---|
| D1_exp_shannon | +0.110 | +0.369 | -0.597 |
| D0_richness | +2.808 | +7.286*** | -1.227 |
| D2_inv_simpson | +0.196 | +0.162 | -0.508 |
| evenness_D1_D0 | -0.035 | -0.073* | -0.013 |
| share: cereals | -0.243* | -0.063 | +0.009 |
| share: drugs_and_narcotics | +0.007 | -0.005 | +0.028 |
| share: fiber_crops | +0.187* | +0.051 | -0.002 |
| share: fodder | -0.005 | -0.002 | +0.005 |
| share: fruits | +0.031 | +0.012 | +0.013 |
| share: oilseeds | -0.056 | +0.023 | +0.125 |
| share: pulses | -0.102 | -0.093** | -0.122*** (q) |
| share: spices | +0.174 | -0.024 | +0.011 |
| share: sugar | -0.013 | +0.111*** (q) | -0.074** |
| share: vegetable | +0.021 | -0.010 | +0.007 |

Stars are uncorrected p values. `(q)` marks results surviving a
Benjamini-Hochberg correction at 5 percent across the 10 crop-category
outcomes, applied separately for each market type.

**The haat result, read properly.** Weekly haats go with +7.3 crops on D0
(p < 0.01) but only +0.37 on D1 (p = 0.66), and evenness is *negative*
at -0.073 (p = 0.05). So haat districts grow more crops, and they grow them on
small patches around the same dominant staple. That is a real finding about
smallholder marketing, and it is much weaker than 'haats make districts diverse'.
An index that averages richness and evenness together would have reported the
strong version.

## C. Input supply

Fertiliser shops, seed centres, soil testing and custom hiring, entered together.

| outcome | fertiliser shop | seed centre | soil testing | custom hiring |
|---|---|---|---|---|
| D1_exp_shannon | +0.482 | -1.164 | +1.019 | -0.026 |
| D0_richness | -0.567 | +5.588** | -1.441 | +0.282 |
| D2_inv_simpson | +0.445 | -1.018 | +0.817 | -0.084 |
| evenness_D1_D0 | +0.004 | -0.101 | +0.069 | -0.007 |
| share: cereals | -0.232** | -0.044 | +0.228 | +0.015 |
| share: drugs_and_narcotics | +0.008 | +0.004 | +0.028 | -0.017 |
| share: fiber_crops | +0.140** | -0.159** | +0.073 | +0.114 |
| share: fodder | -0.007 | -0.009 | +0.005 | -0.019 |
| share: fruits | +0.012 | +0.027 | -0.040 | +0.034* |
| share: oilseeds | +0.096 | +0.134 | -0.201 | -0.050 |
| share: pulses | -0.187*** (q) | +0.053 | -0.055 | +0.028 |
| share: spices | +0.075 | -0.153 | +0.036 | +0.045 |
| share: sugar | +0.062* | +0.132* | -0.154** | -0.097* |
| share: vegetable | +0.032 | +0.014 | +0.080* | -0.054** |

Post-harvest infrastructure, entered on its own:

| outcome | cold storage | farm-gate processing | FPO |
|---|---|---|---|
| D1_exp_shannon | -0.314 | -1.059 | +2.371** |
| D0_richness | +4.772 | +2.202 | +0.443 |
| share: vegetable | -0.007 | -0.085*** | +0.058** |
| share: fruits | -0.037* | +0.030 | +0.033** |
| share: spices | -0.076 | +0.011 | +0.081 |
| share: cereals | +0.234 | +0.080 | -0.223*** |

**FPOs run the opposite way to haats.** They lift D1 without lifting D0, meaning
they go with area spread more evenly across the crops a district already grows,
rather than with extra crops appearing at the margin. Haats add crops; FPOs
rebalance area. Only one index family can tell those apart.

## D. Does market infrastructure explain the irrigation downslope?

[corrected Nov 2026] Within state, diversity is flat across the less irrigated districts and falls once irrigation passes about half of sown area (the 37 percent peak first written here was an ABI-era number). The standard story
about the fall is Punjab: assured procurement plus dense input supply makes the
cereal package the only rational choice. If that is right, the downslope should
be steeper where mandis and fertiliser shops are dense.

Test: interact irrigation and its square with each infrastructure measure,
then read the fitted curve at a low and a high value of that measure.

`curvature` is the coefficient on irrigation squared. More negative means a
sharper peak and a steeper fall. `low` and `high` are one standard deviation
below and above the mean of the infrastructure measure.

| infrastructure | n | curvature at low | curvature at high | interaction | p |
|---|---|---|---|---|---|
| mandi village share | 606 | -9.007 | -3.637 | +2.685** | 0.0301 |
| fertiliser shop village share | 606 | -8.185 | -5.102 | +1.541 | 0.1442 |
| output-market index | 606 | -8.480 | -4.382 | +2.049* | 0.0983 |
| input-supply index | 606 | -8.740 | -4.264 | +2.238* | 0.0564 |
| post-harvest index | 606 | -8.639 | -4.352 | +2.143* | 0.0826 |

## E. A market typology

Districts split at the median on output-market density and on input-supply
density, giving four types.

| type | n | irrigation | D1 | D0 | cereal | pulses | vegetables |
|---|---|---|---|---|---|---|---|
| output high, input high | 183 | 0.595 | 4.90 | 21.4 | 0.617 | 0.082 | 0.019 |
| output high, input low | 120 | 0.512 | 3.78 | 20.9 | 0.780 | 0.080 | 0.023 |
| output low, input high | 120 | 0.554 | 6.02 | 23.4 | 0.560 | 0.133 | 0.010 |
| output low, input low | 183 | 0.515 | 5.06 | 20.4 | 0.697 | 0.116 | 0.018 |

Adjusted differences against 'output high, input high' as the reference:

  output high, input low             -0.1201 (p = 0.5460)
  output low, input high             +0.0395 (p = 0.8945)
  output low, input low              -0.0605 (p = 0.7989)

## F. Checks on the market layer

### F1. Collinearity

The facilities in section A correlate 0.65 to 0.83 with one another. Entering
them together, as sections B and C do, splits shared variance in a way that
makes individual coefficients unstable and can flip signs. Variance inflation
factors for each block, with the controls included:

**output market**: mandi 1.2, weekly_haat 1.2, regular_market 1.7, irr_share 1.6, log_viirs 2.0, estab_per_1000pop 1.4, idx_connectivity 2.0, pca_st_share 1.7
**input supply**: fert_shop 4.1, seed_centre 3.9, soil_test 3.4, custohire 4.1, irr_share 1.7, log_viirs 2.0, estab_per_1000pop 1.4, idx_connectivity 2.7, pca_st_share 1.8

A VIF above 5 is the usual warning line and above 10 the usual stop line.

### F2. Each facility entered on its own

The honest way to read collinear regressors. Every row is a separate model with
that facility as the only market variable, plus the full control set.

| facility | D1 effective crops | D0 richness | cereal share | pulse share |
|---|---|---|---|---|
| mandi | +0.230 | +1.897 | -0.235* | -0.048 |
| regular market | -0.683 | -2.982 | +0.044 | -0.093*** |
| weekly haat | +0.459 | +7.365*** | -0.054 | -0.069 |
| fertiliser shop | +0.304 | +1.077 | -0.193** | -0.172*** |
| seed centre | -0.350 | +4.581** | -0.013 | -0.056 |
| soil testing | +0.319 | +2.653 | +0.112 | -0.076 |
| custom hiring | +0.078 | +1.708 | -0.036 | -0.056 |
| cold storage | +0.684 | +6.328** | +0.128 | -0.171*** |
| farm-gate processing | +0.169 | +3.986** | +0.030 | -0.053 |
| FPO | +1.919** | +2.376 | -0.139* | -0.069* |

### F3. What the development controls are doing

The weekly-haat result is the one worth stress-testing, since it is the only
facility that does not load on the common infrastructure factor.

| specification | n | haat coefficient on D0 richness | p |
|---|---|---|---|
| bivariate | 606 | +16.405*** | 0.0000 |
| state FE only | 606 | +7.205*** | 0.0000 |
| + irrigation | 606 | +7.001*** | 0.0000 |
| full controls | 606 | +7.365*** | 0.0000 |
| full, drop shared parents | 542 | +7.920*** | 0.0000 |
| full, drop no-haat districts | 576 | +6.464*** | 0.0000 |
| full, with all other facilities | 606 | +6.878*** | 0.0000 |

And the same for the haat measured per 100,000 people rather than as a village share:

  haats per 100k population: +0.0554 (p = 0.0068, n = 606)

Scale: a one standard deviation rise in haat village share (0.117) goes with
+0.9 crops, against a mean richness of 21.4.

### F4. What did not hold

Written down so it is not quietly dropped.

- The opening hypothesis was that dense mandi networks push districts toward
  cereals. The coefficient is **negative** (-0.243), so within a state and at a
  given level of irrigation, more mandis goes with *fewer* cereals, not more.
- The second hypothesis was that the irrigation downslope is steeper where market
  and input infrastructure is dense. Every interaction in section D is **positive**,
  meaning the curve is flatter there, and only the mandi one approaches
  significance (p = 0.030). The Punjab story does not show up in this
  cross-section.
- The market typology in section E has large raw differences that vanish entirely
  once the controls go in. All three adjusted contrasts are null.