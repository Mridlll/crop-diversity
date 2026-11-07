# Irrigation threshold: where the two-piece line bends

Window rule: 1998-99 to 2019-20 (scripts/_years.py). Grid 0.20 to 0.80, step 0.01. Bootstrap: wild cluster by state, Webb weights, 1,999 draws (knot, sup-Wald); slope p 9,999 draws.

## A-C. Main: D1, Antyodaya 2019-20 share, state FE, full period

Check: knot fixed at 0.30 gives slope below +2.911, above -4.101 (referee CSV: +2.91, -4.10)
main | D1 on irr_share: n 606 (G 30); knot 0.43; LR 95% set 0.36 to 0.65; Hansen 7.35 set 0.29 to 0.75; slopes +0.88 below, -5.37 above; 67% of districts above the knot; bootstrap 95% 0.29 to 0.72 (SD 0.100, 0% of draws at a grid edge, 74% at or below 0.50); sup-Wald 12.07, p 0.077

SSR profile (every fifth grid point; LR = n (SSR - SSR min) / SSR min):

| knot | SSR | LR | slope below | slope above |
|---|---|---|---|---|
| 0.20 | 1579.69 | 12.23 | +4.84 | -3.43 |
| 0.25 | 1572.26 | 9.33 | +4.10 | -3.77 |
| 0.30 | 1566.23 | 6.97 | +2.91 | -4.10 |
| 0.35 | 1558.44 | 3.92 | +2.07 | -4.53 |
| 0.40 | 1551.21 | 1.09 | +1.30 | -5.02 |
| 0.43 (min) | 1548.43 | 0.00 | +0.88 | -5.37 |
| 0.45 | 1550.41 | 0.78 | +0.44 | -5.49 |
| 0.50 | 1557.24 | 3.45 | -0.54 | -5.69 |
| 0.55 | 1553.05 | 1.81 | -0.89 | -6.45 |
| 0.60 | 1551.12 | 1.05 | -1.24 | -7.27 |
| 0.65 | 1557.54 | 3.56 | -1.71 | -7.73 |
| 0.70 | 1562.93 | 5.67 | -2.07 | -8.27 |
| 0.75 | 1566.83 | 7.20 | -2.31 | -9.07 |
| 0.80 | 1571.90 | 9.19 | -2.53 | -9.85 |

Straight line SSR 1588.02; best kink SSR 1548.43; LR of kink against line 15.50.

## D. Other indices (same sample and model)

- index D0 | D0 on irr_share: n 606 (G 30); knot 0.38; LR 95% set 0.31 to 0.67; Hansen 7.35 set 0.20 to 0.80; slopes +6.99 below, -3.83 above; 76% of districts above the knot; bootstrap 95% 0.25 to 0.77 (SD 0.135, 2% of draws at a grid edge, 81% at or below 0.50); sup-Wald 8.38, p 0.156
- index D0_05 | D0_05 on irr_share: n 606 (G 30); knot 0.57; LR 95% set 0.43 to 0.63; Hansen 7.35 set 0.39 to 0.67; slopes +0.28 below, -11.02 above; 44% of districts above the knot; bootstrap 95% 0.43 to 0.72 (SD 0.080, 1% of draws at a grid edge, 20% at or below 0.50); sup-Wald 15.40, p 0.042
- index D2 | D2 on irr_share: n 606 (G 30); knot 0.42; LR 95% set 0.30 to 0.73; Hansen 7.35 set 0.21 to 0.80; slopes +0.20 below, -3.49 above; 69% of districts above the knot; bootstrap 95% 0.26 to 0.73 (SD 0.123, 2% of draws at a grid edge, 74% at or below 0.50); sup-Wald 8.16, p 0.151
- index E | E on irr_share: n 606 (G 30); knot 0.59; LR 95% set 0.28 to 0.79; Hansen 7.35 set 0.20 to 0.80; slopes -0.07 below, -0.24 above; 41% of districts above the knot; bootstrap 95% 0.25 to 0.80 (SD 0.130, 7% of draws at a grid edge, 27% at or below 0.50); sup-Wald 4.40, p 0.297

## E. Adversarial checks on the D1 knot

### Recent windows
- window a 2015-16 to 2019-20 | D1 on irr_share: n 598 (G 30); knot 0.58; LR 95% set 0.42 to 0.72 (not connected); Hansen 7.35 set 0.36 to 0.80; slopes -0.87 below, -7.24 above; 43% of districts above the knot; bootstrap 95% 0.42 to 0.73 (SD 0.092, 2% of draws at a grid edge, 27% at or below 0.50); sup-Wald 13.45, p 0.047
- window b 2017-18 to 2021-22 (flags dropped, six crops removed) | D1 on irr_share: n 591 (G 29); knot 0.58; LR 95% set 0.52 to 0.76; Hansen 7.35 set 0.41 to 0.80; slopes -0.54 below, -6.79 above; 43% of districts above the knot; bootstrap 95% 0.42 to 0.77 (SD 0.096, 2% of draws at a grid edge, 32% at or below 0.50); sup-Wald 16.04, p 0.021
- extension only 2020-21 and 2021-22 | D1 on irr_share: n 506 (G 25); knot 0.58; LR 95% set 0.52 to 0.76; Hansen 7.35 set 0.42 to 0.80; slopes -0.56 below, -7.36 above; 43% of districts above the knot; bootstrap 95% 0.42 to 0.80 (SD 0.098, 3% of draws at a grid edge, 33% at or below 0.50); sup-Wald 18.33, p 0.015

### Other irrigation measures (full-period D1)
- measure: census 2011 village directory | D1 on irr_c2011: n 534 (G 25); knot 0.20; LR 95% set 0.20 to 0.65 (not connected); Hansen 7.35 set 0.20 to 0.72; slopes +5.43 below, -4.04 above; 75% of districts above the knot; bootstrap 95% 0.20 to 0.68 (SD 0.128, 46% of draws at a grid edge, 95% at or below 0.50); sup-Wald 10.42, p 0.044
- measure: land-use statistics, 1998-2019 mean | D1 on irr_lus: n 600 (G 28); knot 0.20; LR 95% set 0.20 to 0.28; Hansen 7.35 set 0.20 to 0.33; slopes +8.49 below, -4.11 above; 74% of districts above the knot; bootstrap 95% 0.20 to 0.49 (SD 0.072, 53% of draws at a grid edge, 98% at or below 0.50); sup-Wald 22.90, p 0.000
- measure: census 1991 village directory | D1 on irr_c1991: n 539 (G 24); knot 0.21; LR 95% set 0.20 to 0.37; Hansen 7.35 set 0.20 to 0.80; slopes +0.45 below, -3.90 above; 57% of districts above the knot; bootstrap 95% 0.20 to 0.79 (SD 0.172, 36% of draws at a grid edge, 83% at or below 0.50); sup-Wald 2.06, p 0.478

The measures differ in level (means: Antyodaya 0.55, census 2011 0.47, land-use statistics 0.45, census 1991 0.34), so a knot on one scale is not a knot on another. Two further checks: the grid widened to 0.05 for the measures whose knot sits on the 0.20 edge, and every measure turned into its percentile rank among the districts (knot then read as a rank, grid 0.20 to 0.80).
- measure, grid from 0.05: census 2011 village directory | D1 on irr_c2011: n 534 (G 25); knot 0.19; LR 95% set 0.14 to 0.64 (not connected); Hansen 7.35 set 0.12 to 0.71; slopes +6.00 below, -4.00 above; 76% of districts above the knot; bootstrap 95% 0.11 to 0.57 (SD 0.133, 1% of draws at a grid edge, 95% at or below 0.50); sup-Wald 10.42, p 0.099
- measure, grid from 0.05: land-use statistics, 1998-2019 mean | D1 on irr_lus: n 600 (G 28); knot 0.16; LR 95% set 0.13 to 0.26; Hansen 7.35 set 0.12 to 0.31; slopes +12.71 below, -3.92 above; 78% of districts above the knot; bootstrap 95% 0.12 to 0.40 (SD 0.063, 0% of draws at a grid edge, 99% at or below 0.50); sup-Wald 25.40, p 0.003
- measure, grid from 0.05: census 1991 village directory | D1 on irr_c1991: n 539 (G 24); knot 0.05; LR 95% set 0.05 to 0.07; Hansen 7.35 set 0.05 to 0.09; slopes +42.02 below, -3.68 above; 89% of districts above the knot; bootstrap 95% 0.05 to 0.53 (SD 0.124, 52% of draws at a grid edge, 97% at or below 0.50); sup-Wald 4.68, p 0.249
- percentile rank: Antyodaya 2019-20 | D1 on rank of irr_share: n 606 (G 30); knot 0.60; LR 95% set 0.30 to 0.73 (not connected); Hansen 7.35 set 0.22 to 0.80; slopes -0.98 below, -6.42 above; 40% of districts above the knot; bootstrap 95% 0.33 to 0.75 (SD 0.115, 1% of draws at a grid edge, 24% at or below 0.50); sup-Wald 16.94, p 0.025
- percentile rank: census 2011 village directory | D1 on rank of irr_c2011: n 534 (G 25); knot 0.64; LR 95% set 0.20 to 0.70; Hansen 7.35 set 0.20 to 0.74; slopes -0.97 below, -8.53 above; 36% of districts above the knot; bootstrap 95% 0.31 to 0.80 (SD 0.128, 11% of draws at a grid edge, 10% at or below 0.50); sup-Wald 17.17, p 0.018
- percentile rank: land-use statistics, 1998-2019 mean | D1 on rank of irr_lus: n 600 (G 28); knot 0.33; LR 95% set 0.20 to 0.41; Hansen 7.35 set 0.20 to 0.59; slopes +4.42 below, -4.91 above; 67% of districts above the knot; bootstrap 95% 0.20 to 0.57 (SD 0.084, 7% of draws at a grid edge, 93% at or below 0.50); sup-Wald 24.80, p 0.002
- percentile rank: census 1991 village directory | D1 on rank of irr_c1991: n 539 (G 24); knot 0.20; LR 95% set 0.20 to 0.25; Hansen 7.35 set 0.20 to 0.53; slopes +8.77 below, -3.84 above; 80% of districts above the knot; bootstrap 95% 0.20 to 0.62 (SD 0.101, 51% of draws at a grid edge, 96% at or below 0.50); sup-Wald 14.02, p 0.020

### Controls
- terrain controls (elevation, ruggedness) | D1 on irr_share: n 606 (G 30); knot 0.43; LR 95% set 0.36 to 0.63; Hansen 7.35 set 0.31 to 0.73; slopes +1.25 below, -5.24 above; 67% of districts above the knot; bootstrap 95% 0.29 to 0.71 (SD 0.099, 0% of draws at a grid edge, 75% at or below 0.50); sup-Wald 11.72, p 0.079
- drop districts sharing a 2011 parent | D1 on irr_share: n 542 (G 29); knot 0.59; LR 95% set 0.36 to 0.73; Hansen 7.35 set 0.28 to 0.80; slopes -1.19 below, -6.95 above; 42% of districts above the knot; bootstrap 95% 0.43 to 0.80 (SD 0.098, 3% of draws at a grid edge, 27% at or below 0.50); sup-Wald 13.93, p 0.042

### Leave one state out (D1, full period; LR sets only)
Knot range 0.42 to 0.60, median 0.43; 30 of 30 drops keep it within the main bootstrap interval.
Largest moves: MADHYA PRADESH 0.60; ANDHRA PRADESH 0.60; BIHAR 0.59; TAMIL NADU 0.59; KARNATAKA 0.58

## Choice of knot for the site
Estimated knot 0.43; bootstrap 95% 0.29 to 0.72 (width 0.43); LR 95% set 0.36 to 0.65. Rule: use 0.50 if the bootstrap interval is wider than 0.15 and 0.50 lies inside both sets, otherwise the estimate. Imprecise: True; 0.50 inside: True. Knot used: 0.50.

## F. Slopes at the chosen knot (state-clustered, t(G-1) interval, wild restricted p, 9,999 draws)
- full knot 0.30 D1: below +2.91 (95% -0.64 to +6.47, wild p 0.123); above -4.10 (95% -6.86 to -1.34, wild p 0.020); 85% of districts above
- full knot 0.30 D0_05: below +6.84 (95% +0.52 to +13.16, wild p 0.044); above -5.56 (95% -9.67 to -1.44, wild p 0.031); 85% of districts above
- full knot 0.50 D1: below -0.54 (95% -2.70 to +1.63, wild p 0.607); above -5.69 (95% -9.13 to -2.26, wild p 0.037); 56% of districts above
- full knot 0.50 D0_05: below +1.31 (95% -2.40 to +5.01, wild p 0.537); above -8.97 (95% -13.92 to -4.01, wild p 0.028); 56% of districts above
- w1721 knot 0.30 D1: below +1.13 (95% -3.07 to +5.33, wild p 0.590); above -3.31 (95% -6.04 to -0.59, wild p 0.044); 85% of districts above
- w1721 knot 0.30 D0_05: below +5.94 (95% -0.88 to +12.76, wild p 0.095); above -7.21 (95% -10.90 to -3.52, wild p 0.010); 85% of districts above
- w1721 knot 0.50 D1: below -0.24 (95% -2.11 to +1.63, wild p 0.791); above -5.17 (95% -8.73 to -1.61, wild p 0.032); 56% of districts above
- w1721 knot 0.50 D0_05: below -0.42 (95% -4.24 to +3.41, wild p 0.836); above -10.28 (95% -15.20 to -5.36, wild p 0.011); 56% of districts above
