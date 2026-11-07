# Irrigation panel: raw results

Estimation sample: 11,099 district-years, 600 districts, 599 clusters (2011 parents), 28 states, 1998-99 to 2019-20.
Irrigation is LUS net irrigated / net sown. Mean 0.477, SD 0.311.
Within-district SD 0.089 (8% of total variance); within district after state x year FE: see below.

## A. Cross-section, district means, state baseline

Antyodaya 2019-20 (site): D1 = +3.811(1.408) irr -6.836(1.379) irr^2, p(sq)=0.0000, peak 0.279 (se 0.054), n=606
LUS 1998-2019 mean: D1 = +3.455(1.104) irr -6.322(1.053) irr^2, p(sq)=0.0000, peak 0.273 (se 0.047), n=600
statsmodels check (LUS XS): b2 -6.3224 se 1.0534

## B. Panel, quadratic

| spec | outcome | n | districts | clusters | irr | irr^2 | p(irr^2) | peak (se) |
|---|---|---|---|---|---|---|---|---|
| pooled, state x year FE | D1 | 11,092 | 600 | 599 | +3.552 (0.956) | -6.116 (0.909) | 0.0000 | 0.290 (0.041) |
| pooled, state x year FE | D0 | 11,092 | 600 | 599 | +8.070 (2.404) | -9.673 (2.166) | 0.0000 | 0.417 (0.048) |
| pooled, state x year FE | D0_05 | 11,092 | 600 | 599 | +5.411 (1.464) | -8.925 (1.427) | 0.0000 | 0.303 (0.040) |
| pooled, state x year FE | D2 | 11,092 | 600 | 599 | +2.423 (0.725) | -4.131 (0.672) | 0.0000 | 0.293 (0.047) |
| pooled, state x year FE | even | 11,092 | 600 | 599 | +0.071 (0.049) | -0.150 (0.048) | 0.0021 | 0.238 (0.100) |
| pooled, state x year FE | cereal_share | 11,092 | 600 | 599 | -0.191 (0.098) | +0.218 (0.102) | 0.0333 | 0.439 (0.095) |
| district FE + year FE | D1 | 11,081 | 582 | 581 | +2.756 (0.586) | -2.860 (0.493) | 0.0000 | 0.482 (0.038) |
| district FE + year FE | D0 | 11,081 | 582 | 581 | +4.849 (1.579) | -4.988 (1.442) | 0.0006 | 0.486 (0.072) |
| district FE + year FE | D0_05 | 11,081 | 582 | 581 | +4.264 (0.939) | -4.582 (0.770) | 0.0000 | 0.465 (0.044) |
| district FE + year FE | D2 | 11,081 | 582 | 581 | +2.047 (0.476) | -1.967 (0.401) | 0.0000 | 0.521 (0.041) |
| district FE + year FE | even | 11,081 | 582 | 581 | +0.079 (0.029) | -0.054 (0.024) | 0.0268 | 0.732 (0.126) |
| district FE + year FE | cereal_share | 11,081 | 582 | 581 | -0.220 (0.040) | +0.245 (0.034) | 0.0000 | 0.450 (0.035) |
| district FE + state x year FE | D1 | 11,074 | 582 | 581 | +1.636 (0.537) | -1.777 (0.464) | 0.0001 | 0.460 (0.057) |
| district FE + state x year FE | D0 | 11,074 | 582 | 581 | +1.527 (0.931) | -2.199 (0.910) | 0.0160 | 0.347 (0.109) |
| district FE + state x year FE | D0_05 | 11,074 | 582 | 581 | +2.322 (0.711) | -2.995 (0.627) | 0.0000 | 0.388 (0.056) |
| district FE + state x year FE | D2 | 11,074 | 582 | 581 | +1.463 (0.483) | -1.327 (0.422) | 0.0018 | 0.551 (0.059) |
| district FE + state x year FE | even | 11,074 | 582 | 581 | +0.052 (0.028) | -0.047 (0.023) | 0.0433 | 0.560 (0.099) |
| district FE + state x year FE | cereal_share | 11,074 | 582 | 581 | -0.090 (0.036) | +0.114 (0.032) | 0.0003 | 0.396 (0.076) |

statsmodels dummy check (main D1): irr +1.6303 (nan), irr2 -1.7689 (nan)

Linear within-district slope (district FE + state x year FE):
  D1: -0.225 (0.192), p=0.2409, n=11,074
  residual SD of irrigation after district and state x year FE: 0.068
  D0: -0.776 (0.465), p=0.0955, n=11,074
  D0_05: -0.815 (0.290), p=0.0051, n=11,074
  D2: +0.073 (0.161), p=0.6488, n=11,074
  even: +0.003 (0.010), p=0.7227, n=11,074
  cereal_share: +0.029 (0.015), p=0.0588, n=11,074

## C. Robustness of the main specification (D1)

| variant | n | irr | irr^2 | p(irr^2) | peak (se) | linear slope (se) |
|---|---|---|---|---|---|---|
| main | 11,074 | +1.636 (0.537) | -1.777 (0.464) | 0.0001 | 0.460 (0.057) | -0.225 (0.192) |
| drop irrigation spikes | 10,973 | +1.382 (0.564) | -1.736 (0.497) | 0.0005 | 0.398 (0.070) | -0.429 (0.209) |
| district-spell FE (split at NSA territory breaks) | 10,965 | +1.716 (0.535) | -1.735 (0.469) | 0.0002 | 0.494 (0.053) | -0.145 (0.187) |
| drop district-years with an APY area break | 10,007 | +1.084 (0.540) | -1.344 (0.474) | 0.0047 | 0.403 (0.082) | -0.338 (0.185) |
| 3-year centred mean of irrigation | 11,073 | +1.130 (0.638) | -1.561 (0.572) | 0.0066 | 0.362 (0.095) | -0.509 (0.240) |
| irrigation lagged one year | 10,520 | +0.300 (0.448) | -0.638 (0.416) | 0.1257 | 0.235 (0.218) | -0.366 (0.172) |
| gross irrigated / gross cropped | 10,036 | +2.562 (0.692) | -2.781 (0.582) | 0.0000 | 0.461 (0.054) | -0.170 (0.299) |
| districts with >=15 irrigation years | 10,089 | +1.862 (0.616) | -1.956 (0.520) | 0.0002 | 0.476 (0.055) | -0.238 (0.202) |
| cropped-area weighted | 11,074 | +2.701 (1.250) | -2.508 (1.048) | 0.0170 | 0.538 (0.058) | -0.106 (0.293) |
| drop shared 2011 parents | 9,929 | +1.848 (0.593) | -1.902 (0.510) | 0.0002 | 0.486 (0.057) | -0.166 (0.213) |
| district-spell FE, drop spikes [D0] | 10,871 | +0.915 (1.097) | -1.075 (0.987) | 0.2768 | 0.426 (0.277) | -0.229 (0.602) |
| district-spell FE, drop spikes [D0_05] | 10,871 | +2.572 (0.749) | -3.295 (0.655) | 0.0000 | 0.390 (0.057) | -0.938 (0.345) |
| district-spell FE, drop spikes [D2] | 10,871 | +1.356 (0.530) | -1.259 (0.472) | 0.0078 | 0.538 (0.070) | +0.015 (0.180) |
| district-spell FE, drop spikes [even] | 10,871 | +0.058 (0.025) | -0.058 (0.022) | 0.0079 | 0.502 (0.087) | -0.004 (0.010) |

## D. Binned within-district version (district FE + state x year FE)

Reference bin 0-10%. Coefficients in units of the outcome.

| bin | n obs | districts moving through | D1 | D0 | D0_05 | evenness |
|---|---|---|---|---|---|---|
| 10-20 | 1329 | 197 | +0.169 (0.068) | +0.18 (0.14) | +0.15 (0.09) | +0.0035 (0.0041) |
| 20-30 | 1165 | 218 | +0.376 (0.088) | +0.51 (0.18) | +0.51 (0.13) | +0.0089 (0.0052) |
| 30-40 | 1162 | 215 | +0.476 (0.118) | +0.31 (0.20) | +0.44 (0.15) | +0.0159 (0.0062) |
| 40-50 | 1046 | 192 | +0.388 (0.125) | +0.42 (0.21) | +0.35 (0.17) | +0.0122 (0.0070) |
| 50-60 | 863 | 187 | +0.312 (0.129) | +0.12 (0.24) | +0.15 (0.18) | +0.0133 (0.0071) |
| 60-70 | 956 | 171 | +0.135 (0.140) | +0.04 (0.28) | +0.13 (0.20) | +0.0082 (0.0077) |
| 70-80 | 731 | 164 | +0.128 (0.137) | -0.18 (0.34) | -0.06 (0.22) | +0.0086 (0.0077) |
| 80-100 | 2389 | 133 | +0.058 (0.143) | -0.21 (0.34) | -0.26 (0.22) | +0.0057 (0.0080) |

Districts never leaving the 0-10% bin: 39; districts in one bin throughout: 142 of 600.

## E. Long differences, 1998-2000 to 2017-2019

Districts with both windows: 407; mean change in NIA/NSA +0.117 (SD 0.163); 179 rose by more than 0.10, 16 fell by more than 0.10.
Mean change: D1 +0.111, D0 +3.08, D0_05 +0.12.

| outcome | sample | n | dIrr | d(Irr^2) | p(quad) | peak (se) | linear dIrr (se) |
|---|---|---|---|---|---|---|---|
| D1 | all | 406 | -0.663 (1.499) | +0.195 (1.214) | 0.8723 | 1.698 (6.980) | -0.435 (0.440) p=0.323 |
| D0 | all | 406 | +1.950 (2.558) | -3.768 (1.958) | 0.0550 | 0.259 (0.226) | -2.445 (1.105) p=0.027 |
| D0_05 | all | 406 | -0.482 (2.118) | -1.747 (1.696) | 0.3037 | -0.138 (0.732) | -2.520 (0.777) p=0.001 |
| D2 | all | 406 | -0.479 (1.333) | +0.552 (1.088) | 0.6120 | 0.433 (0.454) | +0.166 (0.370) p=0.655 |
| even | all | 406 | -0.060 (0.072) | +0.067 (0.056) | 0.2315 | 0.452 (0.215) | +0.017 (0.022) p=0.428 |
| cereal_share | all | 406 | -0.154 (0.100) | +0.112 (0.078) | 0.1510 | 0.689 (0.184) | -0.024 (0.038) p=0.537 |
| D1 | no NSA territory break | 288 | -1.232 (1.893) | +0.814 (1.487) | 0.5846 | 0.757 (0.449) | -0.240 (0.574) p=0.676 |
| D0 | no NSA territory break | 288 | +3.849 (3.318) | -4.135 (2.436) | 0.0907 | 0.465 (0.200) | -1.195 (1.502) p=0.427 |
| D0_05 | no NSA territory break | 288 | +0.039 (2.470) | -2.066 (1.943) | 0.2884 | 0.009 (0.590) | -2.482 (0.934) p=0.008 |
| D2 | no NSA territory break | 288 | -1.157 (1.711) | +1.189 (1.358) | 0.3819 | 0.486 (0.242) | +0.294 (0.476) p=0.538 |
| even | no NSA territory break | 288 | -0.132 (0.086) | +0.113 (0.064) | 0.0772 | 0.581 (0.125) | +0.006 (0.027) p=0.813 |
| cereal_share | no NSA territory break | 288 | -0.012 (0.129) | -0.004 (0.097) | 0.9677 | -1.550 (53.686) | -0.017 (0.049) p=0.731 |

Linear long-difference slope of D1 on dIrr, by baseline irrigation group (state FE):
  baseline <20%: -0.432 (1.443), p=0.765, n=89
  baseline 20-40%: +1.916 (1.399), p=0.174, n=94
  baseline 40-60%: -2.123 (1.085), p=0.054, n=72
  baseline >=60%: -0.481 (0.769), p=0.533, n=137

## F. Threats inside the within estimate (D1, district FE + state x year FE)

- main, rainfall sample (n=11,047): irr +1.651 (0.540, p=0.002); irr2 -1.788 (0.467, p=0.000)
- plus monsoon, post-monsoon and next-spring rainfall z-scores (n=11,017): irr +1.695 (0.542, p=0.002); irr2 -1.819 (0.468, p=0.000); z_jjas -0.013 (0.011, p=0.228); z_jjas2 +0.001 (0.003, p=0.734); z_ond -0.012 (0.012, p=0.294); z_jfmam_next -0.019 (0.011, p=0.089)
- fixed-denominator irrigation, NIA / own mean NSA (n=11,074): irr_fix +0.983 (0.498, p=0.049); irr_fix2 -0.842 (0.369, p=0.023)
  peak 0.584 (se 0.107)
- decomposition: slope varies with baseline level vs curvature around own baseline (n=11,074): irr +1.625 (0.535, p=0.002); irr_x_base -3.726 (0.934, p=0.000); dev2 -1.266 (0.636, p=0.047)
- lead (next year's irrigation) added (n=10,392): irr +2.205 (0.480, p=0.000); irr2 -2.095 (0.373, p=0.000); irr_lead -0.670 (0.389, p=0.085); irr_lead2 +0.198 (0.335, p=0.555)
- within variation split: SD of slow (7-year mean) 0.046, SD of fast (deviation) 0.046
- slow component quadratic (n=11,060): irr_slow +0.562 (0.817, p=0.492); irr_slow2 -1.205 (0.724, p=0.097)
  peak 0.233 (se 0.216)
- slow and fast, each with a baseline interaction (n=11,060): irr_slow +0.048 (0.610, p=0.938); slow_x_base -2.009 (1.242, p=0.106); irr_fast +2.260 (0.492, p=0.000); fast_x_base -4.135 (0.805, p=0.000)

Reading the decomposition: a negative `irr_x_base` means irrigation gains raise D1 where
a district started dry and lower it where it started wet, which is the cross-district
version of the hump. A negative `dev2` would be curvature inside a single district's
own path.