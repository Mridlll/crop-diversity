# The market layer: mandis, haats and fertiliser shops

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
| D1_exp_shannon | +0.113 | +0.398 | -0.598 |
| D0_richness | +2.820 | +7.269*** | -1.123 |
| D2_inv_simpson | +0.174 | +0.185 | -0.512 |
| evenness_D1_D0 | -0.037 | -0.071* | -0.016 |
| share: cereals | -0.240* | -0.058 | +0.015 |
| share: drugs_and_narcotics | +0.006 | -0.005 | +0.028 |
| share: fiber_crops | +0.185* | +0.050 | -0.004 |
| share: fodder | -0.005 | -0.002 | +0.005 |
| share: fruits | +0.029 | +0.012 | +0.014 |
| share: oilseeds | -0.058 | +0.017 | +0.120 |
| share: pulses | -0.100 | -0.092** | -0.122*** (q) |
| share: spices | +0.178 | -0.024 | +0.010 |
| share: sugar | -0.015 | +0.111*** (q) | -0.073** |
| share: vegetable | +0.020 | -0.009 | +0.007 |

Stars are uncorrected p values. `(q)` marks results surviving a
Benjamini-Hochberg correction at 5 percent across the 10 crop-category
outcomes, applied separately for each market type.

**The haat result, read properly.** Weekly haats go with +7.3 crops on D0
(p < 0.01) but only +0.40 on D1 (p = 0.63), and evenness is *negative*
at -0.071 (p = 0.05). So haat districts grow more crops, and they grow them on
small patches around the same dominant staple. That is a real finding about
smallholder marketing, and it is much weaker than 'haats make districts diverse'.
An index that averages richness and evenness together would have reported the
strong version.

## C. Input supply

Fertiliser shops, seed centres, soil testing and custom hiring, entered together.

| outcome | fertiliser shop | seed centre | soil testing | custom hiring |
|---|---|---|---|---|
| D1_exp_shannon | +0.449 | -1.214 | +1.044 | +0.033 |
| D0_richness | -0.535 | +5.450** | -1.359 | +0.339 |
| D2_inv_simpson | +0.425 | -1.062 | +0.843 | -0.048 |
| evenness_D1_D0 | +0.001 | -0.098 | +0.067 | -0.003 |
| share: cereals | -0.241** | -0.035 | +0.221 | +0.023 |
| share: drugs_and_narcotics | +0.008 | +0.004 | +0.028 | -0.017 |
| share: fiber_crops | +0.129** | -0.152* | +0.070 | +0.118 |
| share: fodder | -0.006 | -0.009 | +0.005 | -0.019 |
| share: fruits | +0.012 | +0.027 | -0.041 | +0.034* |
| share: oilseeds | +0.121 | +0.117 | -0.189 | -0.068 |
| share: pulses | -0.190*** (q) | +0.052 | -0.053 | +0.029 |
| share: spices | +0.074 | -0.150 | +0.033 | +0.047 |
| share: sugar | +0.061* | +0.131* | -0.154** | -0.095* |
| share: vegetable | +0.032 | +0.015 | +0.079* | -0.052** |

Post-harvest infrastructure, entered on its own:

| outcome | cold storage | farm-gate processing | FPO |
|---|---|---|---|
| D1_exp_shannon | -0.265 | -1.048 | +2.385** |
| D0_richness | +4.786 | +2.156 | +0.474 |
| share: vegetable | -0.006 | -0.083*** | +0.057** |
| share: fruits | -0.038* | +0.031 | +0.032** |
| share: spices | -0.075 | +0.009 | +0.082 |
| share: cereals | +0.244* | +0.081 | -0.222*** |

**FPOs run the opposite way to haats.** They lift D1 without lifting D0, meaning
they go with area spread more evenly across the crops a district already grows,
rather than with extra crops appearing at the margin. Haats add crops; FPOs
rebalance area. Only one index family can tell those apart.

## D. Does market infrastructure explain the irrigation downslope?

Diversity peaks near 37 percent irrigation and falls after. The standard story
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
| mandi village share | 606 | -8.978 | -3.741 | +2.619** | 0.0330 |
| fertiliser shop village share | 606 | -8.198 | -5.136 | +1.531 | 0.1430 |
| output-market index | 606 | -8.460 | -4.467 | +1.996 | 0.1023 |
| input-supply index | 606 | -8.759 | -4.300 | +2.230* | 0.0556 |
| post-harvest index | 606 | -8.626 | -4.433 | +2.097* | 0.0867 |

## E. A market typology

Districts split at the median on output-market density and on input-supply
density, giving four types.

| type | n | irrigation | D1 | D0 | cereal | pulses | vegetables |
|---|---|---|---|---|---|---|---|
| output high, input high | 183 | 0.595 | 4.89 | 21.2 | 0.608 | 0.081 | 0.019 |
| output high, input low | 120 | 0.512 | 3.79 | 20.8 | 0.780 | 0.080 | 0.023 |
| output low, input high | 120 | 0.554 | 5.99 | 23.0 | 0.561 | 0.133 | 0.010 |
| output low, input low | 183 | 0.515 | 5.03 | 20.2 | 0.696 | 0.115 | 0.018 |

Adjusted differences against 'output high, input high' as the reference:

  output high, input low             -0.1179 (p = 0.5502)
  output low, input high             +0.0397 (p = 0.8929)
  output low, input low              -0.0699 (p = 0.7668)

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
| mandi | +0.228 | +1.880 | -0.235* | -0.046 |
| regular market | -0.690 | -2.875 | +0.049 | -0.094*** |
| weekly haat | +0.488 | +7.331*** | -0.050 | -0.068 |
| fertiliser shop | +0.278 | +1.099 | -0.198** | -0.174*** |
| seed centre | -0.379 | +4.527** | -0.010 | -0.057 |
| soil testing | +0.317 | +2.665 | +0.111 | -0.075 |
| custom hiring | +0.111 | +1.753 | -0.032 | -0.056 |
| cold storage | +0.750 | +6.337** | +0.141 | -0.172*** |
| farm-gate processing | +0.204 | +3.962** | +0.035 | -0.053 |
| FPO | +1.949** | +2.394 | -0.134* | -0.066* |

### F3. What the development controls are doing

The weekly-haat result is the one worth stress-testing, since it is the only
facility that does not load on the common infrastructure factor.

| specification | n | haat coefficient on D0 richness | p |
|---|---|---|---|
| bivariate | 606 | +16.994*** | 0.0000 |
| state FE only | 606 | +7.182*** | 0.0000 |
| + irrigation | 606 | +6.989*** | 0.0000 |
| full controls | 606 | +7.331*** | 0.0000 |
| full, drop shared parents | 542 | +8.016*** | 0.0000 |
| full, drop no-haat districts | 576 | +6.440*** | 0.0000 |
| full, with all other facilities | 606 | +6.867*** | 0.0000 |

And the same for the haat measured per 100,000 people rather than as a village share:

  haats per 100k population: +0.0541 (p = 0.0075, n = 606)

Scale: a one standard deviation rise in haat village share (0.117) goes with
+0.9 crops, against a mean richness of 21.2.

### F4. What did not hold

Written down so it is not quietly dropped.

- The opening hypothesis was that dense mandi networks push districts toward
  cereals. The coefficient is **negative** (-0.240), so within a state and at a
  given level of irrigation, more mandis goes with *fewer* cereals, not more.
- The second hypothesis was that the irrigation downslope is steeper where market
  and input infrastructure is dense. Every interaction in section D is **positive**,
  meaning the curve is flatter there, and only the mandi one approaches
  significance (p = 0.033). The Punjab story does not show up in this
  cross-section.
- The market typology in section E has large raw differences that vanish entirely
  once the controls go in. All three adjusted contrasts are null.