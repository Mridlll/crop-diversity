# Irrigation v2: recent window, source, predetermined irrigation, long differences

Scripts 122_irrigation_v2_diversity.py (diversity windows) and 123_irrigation_v2_estimates.py (all estimates). Full tables in irrigation_v2_raw.md; every coefficient in irrigation_v2_coefs.csv and irrigation_v2_markets.csv.

Every regression has state fixed effects, so each comparison is between districts of the same state. Each coefficient carries two standard errors, written (parent; state): the first is clustered on 2011 parent district (the site's choice, 605 clusters), the second on state (30 clusters). Stars use the parent clustering: * p < 0.10, ** p < 0.05, *** p < 0.01. A p-value between 0.05 and 0.10 is called near significance below and is not counted as a result. All results are associations.

The full-period diversity indices rebuilt from the extended file match the site's corrected indices exactly (724 districts, maximum difference 0). The full-period irrigation curve reproduces the site's within-state coefficient (squared term -6.836), and the full-period market rows reproduce market_pw_coefs.csv to 1e-6.

## The 2020-21 crop-list break and how it is handled

Window (b), 2017-18 to 2021-22, reaches into the extension. Mean crops per district-year, unflagged state-years:

| year | D0, all crops | D0, six crops removed | D0 at 0.5% | D1 |
|---|---|---|---|---|
| 2017-18 | 21.70 | 19.70 | 9.85 | 5.049 |
| 2018-19 | 22.21 | 20.14 | 9.53 | 4.808 |
| 2019-20 | 22.66 | 20.58 | 9.59 | 4.780 |
| 2020-21 | 22.23 | 20.43 | 9.43 | 4.678 |
| 2021-22 | 21.84 | 20.00 | 9.28 | 4.613 |

Three versions of window (b) are estimated. "Flags dropped" removes the agreed unreliable state-years (Manipur, Madhya Pradesh and Jammu and Kashmir in all new years; Meghalaya and Nagaland 2021-22; DNH and DD 2021-23; Haryana 2022-23 falls outside the window). "Six crops removed" also drops niger seed, coriander, garlic, turmeric, cashewnut and banana in every year; this is the main version of (b). "State-stable" keeps a crop in a state only if the state reports it in every unflagged year of the window. Window (a), 2015-16 to 2019-20, is shown both on all crops and on the same reduced lists, so a difference between (a) and (b) cannot come from the crop list. Districts need at least 3 of 5 years in a window (10 of 22 for the full period, as on the site). Removing the six crops lowers D0 by about 2 crops everywhere; it barely moves any estimate below.

## Q1. Recent window

### Irrigation bend (quadratic in the Antyodaya 2019-20 irrigated share)

| outcome | window | n | squared term (parent; state) | peak (parent; state) |
|---|---|---|---|---|
| D1 | (c) full, site | 606 | -6.84*** (1.38; 1.65) | 0.28 (0.05; 0.07) |
| D1 | (a) 2015-20, all crops | 598 | -6.78*** (1.46; 1.80) | 0.28 (0.06; 0.07) |
| D1 | (b) 2017-22, flags, six removed | 591 | -6.46*** (1.40; 1.89) | 0.30 (0.05; 0.06) |
| D1 | (b) 2017-22, raw | 598 | -6.16*** (1.49; 2.03) | 0.29 (0.06; 0.07) |
| D0 | (c) full, site | 606 | -11.43*** (2.93; 3.10) | 0.46 (0.04; 0.09) |
| D0 | (a) 2015-20, all crops | 598 | -12.82*** (3.42; 3.21) | 0.41 (0.05; 0.08) |
| D0 | (a) 2015-20, six removed | 598 | -12.91*** (3.03; 2.97) | 0.44 (0.04; 0.07) |
| D0 | (b) 2017-22, flags, six removed | 591 | -12.06*** (3.03; 2.82) | 0.45 (0.04; 0.07) |
| D0 at 0.5% | (c) full, site | 606 | -12.97*** (2.36; 2.96) | 0.36 (0.04; 0.06) |
| D0 at 0.5% | (a) 2015-20, all crops | 598 | -12.48*** (2.52; 3.37) | 0.31 (0.05; 0.07) |
| D0 at 0.5% | (b) 2017-22, flags, six removed | 591 | -12.81*** (2.47; 3.37) | 0.30 (0.05; 0.07) |
| D2 | (c) full, site | 606 | -3.85*** (1.03; 1.29) | 0.22 (0.09; 0.11) |
| D2 | (a) 2015-20, all crops | 598 | -3.85*** (1.11; 1.23) | 0.25 (0.09; 0.09) |
| D2 | (b) 2017-22, flags, six removed | 591 | -3.77*** (1.10; 1.34) | 0.29 (0.08; 0.09) |
| evenness | (c) full, site | 606 | -0.175** (0.077; 0.088) | 0.16 (0.19; 0.22) |
| evenness | (a) 2015-20, all crops | 598 | -0.156* (0.080; 0.088) | 0.19 (0.20; 0.23) |
| evenness | (b) 2017-22, flags, six removed | 591 | -0.165* (0.086; 0.094) | 0.18 (0.21; 0.23) |

On the 590 districts present in every window, the D1 squared terms are -6.80, -6.84 and -6.46 for (c), (a) and (b); the D0 terms -11.51, -12.83 and -12.06.

What changes. The bend in D1, D0, D0 at 0.5% and D2 is the same size in both recent windows as over the full period, and every squared term stays significant under state clustering (largest state-clustered p is 0.010, for D2 in raw (b)). The D1 peak is 0.28 to 0.30 in every version. The thresholded count peaks a little earlier in the recent windows (0.29 to 0.33 against 0.36), a shift of about one standard error. The raw D0 peak sits at 0.41 to 0.45 against 0.46. Evenness is the one measure that weakens: its squared term is -0.155 to -0.166 in the recent windows with p 0.045 to 0.063 (parent) and 0.058 to 0.103 (state), so evenness is near significance only in the recent windows, and its peak was never estimated with any precision (standard error about 0.2 in every version).

### Market headline results (population shares, all site controls, state FE)

| facility | outcome | window | n | b | SE parent | SE state | p parent / state |
|---|---|---|---|---|---|---|---|
| weekly haat | D0 | (c) full, site | 606 | 6.28*** | 1.23 | 1.49 | 0.000 / 0.000 |
| weekly haat | D0 | (a) 2015-20, all crops | 598 | 5.88*** | 1.49 | 1.63 | 0.000 / 0.000 |
| weekly haat | D0 | (a) 2015-20, six removed | 598 | 4.86*** | 1.34 | 1.52 | 0.000 / 0.001 |
| weekly haat | D0 | (b) flags, six removed | 591 | 4.61*** | 1.36 | 1.71 | 0.001 / 0.007 |
| weekly haat | D0 | (b) flags, state-stable | 590 | 5.69*** | 1.49 | 1.74 | 0.000 / 0.001 |
| FPO | D1 | (c) full, site | 606 | 1.89** | 0.79 | 1.33 | 0.017 / 0.156 |
| FPO | D1 | (a) 2015-20, all crops | 598 | 2.25*** | 0.82 | 1.41 | 0.006 / 0.110 |
| FPO | D1 | (b) flags, six removed | 591 | 2.10*** | 0.79 | 1.26 | 0.008 / 0.094 |
| FPO | D0 | (c) full, site | 606 | 3.38** | 1.49 | 1.77 | 0.024 / 0.056 |
| FPO | D0 | (a) 2015-20, all crops | 598 | 3.59** | 1.75 | 2.02 | 0.040 / 0.076 |
| FPO | D0 | (b) flags, six removed | 591 | 4.05** | 1.58 | 1.80 | 0.011 / 0.025 |
| fertiliser shop | pulse share | (c) full, site | 606 | -0.123*** | 0.035 | 0.055 | 0.000 / 0.027 |
| fertiliser shop | pulse share | (a) 2015-20, all crops | 598 | -0.096** | 0.044 | 0.059 | 0.030 / 0.104 |
| fertiliser shop | pulse share | (b) flags, six removed | 591 | -0.089** | 0.045 | 0.055 | 0.049 / 0.108 |
| mandi | cereal share | (c) full, site | 606 | -0.153 | 0.112 | 0.146 | 0.172 / 0.296 |
| mandi | cereal share | (a) 2015-20, all crops | 598 | -0.165 | 0.128 | 0.165 | 0.196 / 0.316 |
| mandi | cereal share | (b) flags, six removed | 591 | -0.158 | 0.135 | 0.167 | 0.241 / 0.343 |

What changes. The weekly haat and count of crops link holds in every window and survives state clustering; it shrinks by about a quarter when the six crops are removed (6.3 to 4.6 to 4.9), so part of it runs through those minor crops, which haat districts grow more of. The haat has no link to D1 in any window (b between -0.34 and 0.55, all p above 0.4). Producer organisations and D1 hold at the parent level in every window (1.9 to 2.4) but under state clustering they were never significant (p 0.16 on the site's own window) and stay near significance at best (p 0.08 to 0.13). FPO and D0 is slightly stronger in window (b) than on the site. The fertiliser shop and pulse share link is about a quarter smaller in the recent windows (-0.09 to -0.10 against -0.12) and loses its state-clustered significance (p 0.10 to 0.11 against 0.027). Mandis show no clear link to the cereal share in any window, as the site already says.

## Q2. Canal or groundwater

2011 census sources (usable districts, Maharashtra excluded) against diversity averaged 2009-10 to 2013-14, terrain controls (mean elevation, ruggedness) and state FE. Groundwater makes up 0.435 of irrigated area in the average district and canal 0.327.

| outcome | spec | n | squared term (parent; state) | peak (parent; state) |
|---|---|---|---|---|
| D1 | A total share | 534 | -5.78*** (1.09; 1.94) | 0.24 (0.06; 0.10) |
| D1 | B + groundwater share of irrigated | 531 | -5.80*** (1.12; 1.94) | 0.24 (0.07; 0.11) |
| D1 | E groundwater-majority districts | 231 | -5.68** (2.33; 3.33) | 0.35 (0.13; 0.18) |
| D1 | E canal-majority districts | 136 | -2.42 (1.63; 1.76) | 0.20 (0.22; 0.31) |
| D1 | E neither above half | 164 | -6.34*** (2.11; 1.69) | 0.11 (0.13; 0.10) |
| D0 | A total share | 534 | -9.54*** (2.54; 4.02) | 0.32 (0.08; 0.13) |
| D0 | E groundwater-majority | 231 | -15.65*** (4.40; 6.67) | 0.52 (0.05; 0.09) |
| D0 | E canal-majority | 136 | -4.49 (4.17; 4.61) | 0.35 (0.18; 0.26) |
| D0 at 0.5% | A total share | 534 | -9.53*** (1.76; 2.74) | 0.29 (0.05; 0.09) |
| D0 at 0.5% | E groundwater-majority | 231 | -12.08*** (3.14; 4.76) | 0.42 (0.06; 0.13) |
| D0 at 0.5% | E canal-majority | 136 | -4.56 (2.87; 2.61) | 0.19 (0.22; 0.29) |
| evenness | A total share | 534 | -0.173*** (0.049; 0.074) | 0.30 (0.09; 0.11) |

Curvature by source mix (spec C, total share and its square each interacted with the groundwater share of irrigated area, n 531). Squared term in a district with no groundwater against one irrigated only by groundwater:

| outcome | no groundwater | all groundwater | difference, p parent / state |
|---|---|---|---|
| D1 | -4.58 (1.59; 1.46) | -7.28 (2.37; 4.54) | -2.70, 0.40 / 0.59 |
| D0 | -6.70 (3.68; 3.64) | -14.71 (5.13; 9.28) | -8.02, 0.26 / 0.46 |
| D0 at 0.5% | -8.71 (2.52; 2.41) | -10.29 (3.66; 6.25) | -1.58, 0.75 / 0.82 |
| evenness | -0.203 (0.083; 0.079) | -0.120 (0.106; 0.156) | +0.08, 0.60 / 0.66 |

Separate source shares of sown area with squares (spec D, n 534, plus tank and river/other shares): the squared groundwater term is -10.55*** (3.29; 5.80) for D0 and -4.87** (2.19; 3.89) for D0 at 0.5%; the squared canal term is -7.11* (3.90; 4.13) and -5.49** (2.46; 3.18). For D1 neither square is significant (canal -2.10, groundwater -2.77*, state p 0.39 and 0.29), and the strongest single term is the tank share: -9.28*** (1.33; 0.93) for D1 and -10.74*** for D0 at 0.5%.

Antyodaya version (villages by main source, recent window 2015-16 to 2019-20, n 598): the groundwater-majority districts (334) show the bend in D1 (-6.88***, state p 0.074) and D0 at 0.5% (-11.59***, state p 0.024); the canal-majority districts are only 40, with no bend and very wide errors. The source-mix interaction is not significant for any outcome (p 0.20 to 0.82).

Plainly: the downturn shows up clearly in groundwater-dominated districts, and it is there in districts with no single dominant source. In canal-dominated districts the squared term is smaller and not significant, but those districts are fewer (136 census, 40 Antyodaya) and the test of a difference between the two curves is far from significant in every outcome (p 0.26 to 0.82). So the data cannot say the downturn comes from groundwater rather than canals; they say it is not confined to canal districts. High-irrigation districts are slightly more groundwater-irrigated (groundwater 0.55 of irrigated area above 80% irrigation against 0.26 below 20%), so groundwater carries most of the observations on the falling side. Tank irrigation has the largest and most robust negative association with D1 of any source.

## Q3. Irrigation measured in 1991

The 1991 share (539 usable districts) correlates 0.79 with the Antyodaya 2019-20 share and averages 0.34. Terrain controls and state FE throughout.

| outcome | window | measure | n | squared term (parent; state) | peak (parent; state) |
|---|---|---|---|---|---|
| D1 | (c) full | 1991 | 539 | -2.58** (1.09; 1.56) | -0.15 (0.29; 0.41) |
| D1 | (c) full | Antyodaya, same districts | 539 | -6.13*** (1.51; 1.89) | 0.25 (0.08; 0.12) |
| D1 | (a) 2015-20 | 1991 | 532 | -3.17*** (1.11; 1.57) | -0.05 (0.21; 0.28) |
| D1 | (b) 2017-22 | 1991 | 532 | -3.52*** (1.09; 1.35) | 0.01 (0.16; 0.19) |
| D0 | (c) full | 1991 | 539 | -8.47*** (2.35; 4.29) | 0.24 (0.09; 0.15) |
| D0 | (c) full | Antyodaya, same districts | 539 | -10.37*** (3.08; 3.54) | 0.40 (0.07; 0.14) |
| D0 | (a) 2015-20 | 1991 | 532 | -10.00*** (2.64; 4.30) | 0.18 (0.10; 0.13) |
| D0 | (b) 2017-22 | 1991 | 532 | -10.29*** (2.29; 3.69) | 0.24 (0.07; 0.09) |
| D0 at 0.5% | (c) full | 1991 | 539 | -5.58*** (1.67; 2.63) | 0.13 (0.12; 0.17) |
| D0 at 0.5% | (a) 2015-20 | 1991 | 532 | -4.94*** (1.75; 3.01) | 0.05 (0.16; 0.25) |
| D0 at 0.5% | (b) 2017-22 | 1991 | 532 | -4.66*** (1.72; 2.91) | -0.01 (0.19; 0.29) |
| evenness | (c) full | 1991 | 539 | -0.082* (0.042; 0.051) | -0.05 (0.31; 0.44) |

Adding the 1991 groundwater share of irrigated area changes nothing (its coefficient is never significant, p 0.12 to 0.95). With separate 1991 canal and groundwater shares, both squared terms are negative for D0 (canal -11.5 to -14.0, groundwater -13.5 to -14.5, each significant at the parent level, state p 0.06 to 0.2). The 1991 electrified wells and tube wells share is not significant except for D0 in window (b) (3.12, p 0.019 parent, 0.175 state).

Plainly: the downward curvature survives when irrigation is measured in 1991, in every window. The squared term on D1 is about half the size it is with the 2019-20 measure on the same districts, and under state clustering it is near significance on the full period (p 0.097) and significant in the recent windows (p 0.044 and 0.009). The rising part does not survive: with the 1991 share, D1 and the thresholded count fall from zero irrigation onward (peaks at or below 0.13, standard errors 0.12 to 0.4), and only the raw count keeps an interior peak (0.18 to 0.24). Part of the leftward shift is units: on these districts a 1991 share of 0 corresponds on average to a 2019-20 share of 0.34, and 0.24 to 0.50, because irrigation spread most where it started lowest. So the 1991 data support the decline at high irrigation, measured before the crop record starts, and are silent on the rise at low irrigation.

## Q4. Long differences

Change in diversity between 1998-99 to 2002-03 and 2009-10 to 2013-14 (and to 2015-16 to 2019-20) on change in irrigated share, state FE, terrain controls. Mean change 1991 to 2011: irrigated share +0.120, of which groundwater +0.075 and canal +0.027. 2001 to 2011: +0.042, groundwater +0.074, canal -0.014; the 2001 groundwater figure omits electrified open wells, so its change to 2011 is overstated and that comparison is the weaker one. A second sample keeps only districts whose mean gross cropped area stays within 2/3 to 1.5 times its early level, to drop districts whose APY boundaries moved.

| census change | diversity to | sample | outcome | n | total change slope (parent; state) | canal | groundwater |
|---|---|---|---|---|---|---|---|
| 1991-2011 | 2009-13 | all | D1 | 454 | 0.16 (0.46; 0.55) | 0.99* (0.51; 0.65) | 0.42 (0.54; 0.67) |
| 1991-2011 | 2009-13 | all | D0 | 454 | 0.14 (1.15; 0.97) | 1.56 (1.37; 1.96) | 2.51* (1.43; 1.59) |
| 1991-2011 | 2009-13 | all | D0 at 0.5% | 454 | -1.76** (0.72; 0.64) | 0.51 (0.91; 1.38) | -0.96 (0.85; 0.89) |
| 1991-2011 | 2009-13 | area-stable | D1 | 403 | 0.28 (0.47; 0.62) | 0.99** (0.48; 0.57) | 0.17 (0.53; 0.76) |
| 1991-2011 | 2009-13 | area-stable | D0 at 0.5% | 403 | -1.74** (0.75; 0.71) | 0.57 (0.88; 1.25) | -1.66** (0.83; 0.95) |
| 1991-2011 | 2015-19 | all | D1 | 447 | 0.42 (0.51; 0.66) | 0.69 (0.54; 0.64) | 0.78 (0.64; 0.79) |
| 1991-2011 | 2015-19 | all | D0 | 447 | 1.04 (1.08; 0.89) | 1.10 (1.35; 1.28) | 3.65** (1.48; 1.85) |
| 1991-2011 | 2015-19 | all | D0 at 0.5% | 447 | -1.47** (0.71; 0.86) | -0.94 (0.92; 1.03) | -0.67 (0.95; 0.97) |
| 1991-2011 | 2015-19 | area-stable | D0 | 352 | 1.57 (1.19; 1.08) | 1.05 (1.40; 1.64) | 3.98*** (1.51; 2.02) |
| 2001-2011 | 2009-13 | all | D1 | 434 | 0.52 (0.34; 0.29) | 0.14 (0.39; 0.43) | -0.65 (0.46; 0.51) |
| 2001-2011 | 2009-13 | all | D0 at 0.5% | 434 | -0.28 (0.65; 0.65) | 0.33 (0.72; 0.44) | -1.55** (0.72; 0.78) |
| 2001-2011 | 2009-13 | area-stable | D0 at 0.5% | 379 | -0.52 (0.70; 0.59) | 0.00 (0.69; 0.40) | -2.51*** (0.81; 0.70) |
| 2001-2011 | 2015-19 | all | D1 | 426 | 0.05 (0.48; 0.31) | -0.14 (0.53; 0.47) | -0.34 (0.69; 0.52) |
| 2001-2011 | 2015-19 | all | D0 at 0.5% | 426 | -0.53 (0.75; 0.54) | -0.17 (0.79; 0.52) | -0.17 (0.94; 0.75) |

Evenness has no slope on any change (all p above 0.2 for the total). When the change in the square is added, D1 shows a concave pattern in the 1991-2011 to 2009-13 comparison on area-stable districts (change in square -2.23***, state p 0.003) and in 2001-2011 to 2009-13 (-2.36**, state p 0.071), but not when diversity runs to 2015-19 (p 0.17 to 0.60).

Plainly: districts where irrigation expanded more did not change their effective number of crops differently from others in the same state (slopes 0.05 to 0.65, none significant). They did lose main crops: the count of crops holding at least 0.5% of area fell by about 1.5 to 1.8 crops per unit change in irrigated share when the change is measured 1991 to 2011 (significant under both clusterings to 2009-13; state p 0.088 to 2015-19). A typical 1991-2011 change of +0.12 corresponds to about 0.2 fewer main crops. The source split points two ways at once: groundwater expansion 1991 to 2011 goes with more crops on the raw count (2.3 to 4.0; significant when diversity runs to 2015-19, near significance to 2009-13), and groundwater expansion in general with fewer on the thresholded count (-0.1 to -2.5, significant in three of eight versions), which reads as minor crops added while area concentrates; canal expansion goes with a higher D1 in the 1991-2011 to 2009-13 comparison only (about 1.0, state p 0.08 to 0.13). The canal minus groundwater difference is significant under both clusterings only for D0 at 0.5% in 2001-2011 to 2009-13 (p 0.047 / 0.018 all districts; 0.012 / 0.001 area-stable), which is the comparison with the overstated groundwater change. With about 40 long-difference tests run, the source contrasts are fragile and are not a result.

## What holds

1. The irrigation bend in D1, D0, D0 at 0.5% and D2 is the same in 2015-16 to 2019-20 and in 2017-18 to 2021-22 (stable crop list, flagged state-years dropped) as over the full period, significant under both clusterings; the D1 peak stays at 0.28 to 0.30 irrigated.
2. Evenness is the weak member: its bend falls to near significance in the recent windows (p 0.05 to 0.10).
3. Weekly haats go with a larger count of crops in every window under both clusterings (4.6 to 6.3 crops from no resident to all residents near a haat), about a quarter smaller once the six dropped crops are excluded; no link to D1.
4. Producer organisations go with higher D1 and D0 in every window at the parent level; under state clustering the D1 link is near significance only, as it already was on the site.
5. Fertiliser shops and a lower pulse share hold in direction and parent-level significance, about a quarter smaller in recent windows, and lose state-clustered significance (p about 0.10). Mandis show no link to the cereal share in any window.
6. The downturn is present in groundwater-majority districts and in mixed districts; canal-majority districts show a weaker, imprecise bend, and no test distinguishes the canal and groundwater curves (p 0.26 to 0.82). Tank irrigation has the strongest negative association with D1.
7. With irrigation measured in 1991, before the crop record, diversity still falls at high irrigation (D1 squared term -2.6 to -3.5, D0 -8.5 to -10.3), in every window; the rise at low irrigation is not visible on the 1991 measure.
8. In long differences, more irrigation expansion 1991-2011 goes with losing main crops (D0 at 0.5%, about -1.5 to -1.8 per unit change) but no change in D1 or evenness. Canal against groundwater differences are fragile and are not a finding.

## Files

outputs/revision_nov8/irrigation_v2/:
- irrigation_v2_results.md (this note), irrigation_v2_raw.md (every table)
- irrigation_v2_coefs.csv (Q1 curves, Q2, Q3, Q4: n, coefficients, both SEs, p, peaks), irrigation_v2_markets.csv (Q1 markets)
- div_windows.csv (district diversity per window and crop-list variant), div_district_year.csv (district-year indices 1998-99 to 2022-23, all crops and six removed, with the state-year flag), div_build_log.md
- irrigation_v2_longdiff_data.csv (district file behind Q2 to Q4)
