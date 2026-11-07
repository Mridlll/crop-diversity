# Population-weighted market facilities

## Data and method

Antyodaya village rows with a district key: 521,223.
Rows with a Census 2011 population: 521,223 (100.00%); zero or missing: 0.
Rows with an Antyodaya 2019-20 population: 521,223 (100.00%); zero or missing: 0.
Village-level correlation of the two population series: 0.405 (log-log 0.903).
Missing facility flags (treated as no facility, as in script 79): {}
Check: rebuilt unweighted village shares against market_covariates.csv, max abs difference 1.11e-16 over 617 districts.
Check: rebuilt unweighted group indices reproduce market_covariates.csv exactly.
Analysis set: 606 districts, 30 states, 605 clusters (shared 2011 parent).
Rural population in the analysis districts, Census 2011 weights: 716 million; villages with weight 0: 0.

## Weighted against unweighted shares, facility by facility

| facility | mean village share | mean pop share | SD village | SD pop | Pearson r | Spearman r | r (Antyodaya pop weights) | districts with none (v / p) | districts gap > 5 pts |
|---|---|---|---|---|---|---|---|---|---|
| mandi | 0.030 | 0.036 | 0.052 | 0.055 | 0.953 | 0.918 | 0.959 | 40 / 40 | 16 |
| regular market | 0.098 | 0.136 | 0.147 | 0.157 | 0.970 | 0.959 | 0.972 | 6 / 6 | 191 |
| weekly haat | 0.143 | 0.190 | 0.117 | 0.150 | 0.953 | 0.966 | 0.951 | 30 / 30 | 255 |
| fertiliser shop | 0.175 | 0.254 | 0.176 | 0.192 | 0.954 | 0.959 | 0.958 | 15 / 15 | 422 |
| seed centre | 0.098 | 0.145 | 0.105 | 0.119 | 0.939 | 0.920 | 0.927 | 14 / 14 | 242 |
| soil testing | 0.043 | 0.056 | 0.077 | 0.081 | 0.947 | 0.911 | 0.973 | 39 / 39 | 28 |
| custom hiring | 0.092 | 0.115 | 0.123 | 0.129 | 0.967 | 0.947 | 0.973 | 13 / 13 | 93 |
| cold storage | 0.084 | 0.121 | 0.081 | 0.098 | 0.921 | 0.940 | 0.910 | 14 / 14 | 187 |
| farm-gate processing | 0.114 | 0.143 | 0.138 | 0.137 | 0.972 | 0.945 | 0.965 | 13 / 13 | 129 |
| FPO | 0.231 | 0.279 | 0.194 | 0.204 | 0.972 | 0.973 | 0.977 | 5 / 5 | 266 |

Correlation among the weighted facilities (the structure that the site's section 2 rests on):

Nine non-haat facilities, pairwise range: village 0.24 to 0.83; population 0.09 to 0.80.
Weekly haat against the other nine: village -0.21 to 0.06; population -0.34 to 0.04.
Haat vs regular market / mandi: village -0.21 / -0.15; population -0.34 / -0.22.

Baseline check: the U version of the solo models reproduces market_results_solo.csv (max |db| 4.4e-16, max |dp| 1.1e-16, 40 cells).
Baseline check: the U version of the output-market block reproduces market_results_output.csv (max |db| 8.9e-16).

## Headline: one facility at a time (the site's main figure and table)

Each row is a separate model with the full control set and state fixed effects, SEs clustered on the shared 2011 parent. 'village' is the site; 'pop' is the Census-2011 population share (version W, controls also population-weighted). Coefficients are the change from no village to all villages (or no residents to all residents) having the facility.

| outcome | facility | village b (SE) | p | pop b (SE) | p | ratio | flag |
|---|---|---|---|---|---|---|---|
| D0_richness | mandi | +1.897 (3.083) | 0.538 | +0.664 (2.668) | 0.803 | 0.35 |  |
| D0_richness | regular market | -2.982 (1.867) | 0.110 | -3.483 (1.696) | 0.040 | 1.17 | SIGNIF gained |
| D0_richness | weekly haat | +7.365 (1.586) | 0.000 | +6.276 (1.231) | 0.000 | 0.85 |  |
| D0_richness | fertiliser shop | +1.077 (1.946) | 0.580 | +2.908 (1.695) | 0.086 | 2.70 |  |
| D0_richness | seed centre | +4.581 (2.004) | 0.022 | +5.009 (1.672) | 0.003 | 1.09 |  |
| D0_richness | soil testing | +2.653 (1.957) | 0.175 | +2.075 (1.866) | 0.266 | 0.78 |  |
| D0_richness | custom hiring | +1.708 (2.486) | 0.492 | +1.730 (2.221) | 0.436 | 1.01 |  |
| D0_richness | cold storage | +6.328 (3.075) | 0.040 | +5.277 (2.506) | 0.035 | 0.83 |  |
| D0_richness | farm-gate processing | +3.986 (2.014) | 0.048 | +4.441 (1.914) | 0.020 | 1.11 |  |
| D0_richness | FPO | +2.376 (1.830) | 0.194 | +3.380 (1.494) | 0.024 | 1.42 | SIGNIF gained |
| D1_exp_shannon | mandi | +0.230 (2.130) | 0.914 | +0.263 (1.847) | 0.887 | 1.15 |  |
| D1_exp_shannon | regular market | -0.683 (0.812) | 0.400 | -0.998 (0.728) | 0.170 | 1.46 |  |
| D1_exp_shannon | weekly haat | +0.459 (0.844) | 0.587 | +0.553 (0.717) | 0.441 | 1.20 |  |
| D1_exp_shannon | fertiliser shop | +0.304 (0.990) | 0.758 | +0.923 (0.873) | 0.291 | 3.03 |  |
| D1_exp_shannon | seed centre | -0.350 (1.012) | 0.729 | +0.544 (0.840) | 0.517 | -1.55 | SIGN |
| D1_exp_shannon | soil testing | +0.319 (0.985) | 0.746 | +0.686 (0.901) | 0.447 | 2.15 |  |
| D1_exp_shannon | custom hiring | +0.078 (1.260) | 0.951 | -0.458 (1.076) | 0.670 | -5.89 | SIGN |
| D1_exp_shannon | cold storage | +0.684 (1.648) | 0.678 | +0.755 (1.277) | 0.554 | 1.10 |  |
| D1_exp_shannon | farm-gate processing | +0.169 (0.914) | 0.853 | +0.089 (0.854) | 0.917 | 0.53 |  |
| D1_exp_shannon | FPO | +1.919 (0.916) | 0.036 | +1.886 (0.790) | 0.017 | 0.98 |  |
| evenness_D1_D0 | mandi | -0.018 (0.126) | 0.886 | +0.011 (0.114) | 0.921 | -0.62 | SIGN |
| evenness_D1_D0 | regular market | +0.005 (0.051) | 0.921 | -0.002 (0.042) | 0.961 | -0.40 | SIGN |
| evenness_D1_D0 | weekly haat | -0.070 (0.036) | 0.052 | -0.046 (0.028) | 0.103 | 0.67 |  |
| evenness_D1_D0 | fertiliser shop | -0.017 (0.052) | 0.741 | -0.005 (0.040) | 0.901 | 0.29 |  |
| evenness_D1_D0 | seed centre | -0.061 (0.063) | 0.333 | -0.015 (0.047) | 0.748 | 0.25 |  |
| evenness_D1_D0 | soil testing | -0.008 (0.065) | 0.896 | +0.017 (0.053) | 0.752 | -1.98 | SIGN |
| evenness_D1_D0 | custom hiring | -0.022 (0.067) | 0.742 | -0.037 (0.059) | 0.525 | 1.70 |  |
| evenness_D1_D0 | cold storage | -0.069 (0.078) | 0.375 | -0.027 (0.062) | 0.667 | 0.39 |  |
| evenness_D1_D0 | farm-gate processing | -0.068 (0.060) | 0.259 | -0.070 (0.053) | 0.190 | 1.03 |  |
| evenness_D1_D0 | FPO | +0.055 (0.046) | 0.235 | +0.048 (0.038) | 0.203 | 0.88 |  |
| share: cereals | mandi | -0.235 (0.122) | 0.054 | -0.153 (0.112) | 0.172 | 0.65 |  |
| share: cereals | regular market | +0.044 (0.092) | 0.632 | +0.049 (0.077) | 0.526 | 1.10 |  |
| share: cereals | weekly haat | -0.054 (0.067) | 0.417 | -0.074 (0.056) | 0.184 | 1.37 |  |
| share: cereals | fertiliser shop | -0.193 (0.087) | 0.026 | -0.216 (0.073) | 0.003 | 1.12 |  |
| share: cereals | seed centre | -0.013 (0.099) | 0.897 | -0.091 (0.086) | 0.287 | 7.05 |  |
| share: cereals | soil testing | +0.112 (0.128) | 0.383 | +0.093 (0.113) | 0.412 | 0.83 |  |
| share: cereals | custom hiring | -0.036 (0.094) | 0.697 | -0.159 (0.097) | 0.099 | 4.37 |  |
| share: cereals | cold storage | +0.128 (0.126) | 0.309 | -0.071 (0.101) | 0.483 | -0.55 | SIGN |
| share: cereals | farm-gate processing | +0.030 (0.100) | 0.766 | -0.039 (0.088) | 0.660 | -1.30 | SIGN |
| share: cereals | FPO | -0.139 (0.073) | 0.057 | -0.120 (0.064) | 0.061 | 0.87 |  |
| share: pulses | mandi | -0.048 (0.066) | 0.464 | -0.026 (0.056) | 0.644 | 0.53 |  |
| share: pulses | regular market | -0.093 (0.033) | 0.004 | -0.082 (0.027) | 0.002 | 0.88 |  |
| share: pulses | weekly haat | -0.069 (0.044) | 0.112 | -0.041 (0.034) | 0.235 | 0.58 |  |
| share: pulses | fertiliser shop | -0.172 (0.043) | 0.000 | -0.123 (0.035) | 0.000 | 0.71 |  |
| share: pulses | seed centre | -0.056 (0.044) | 0.199 | -0.021 (0.039) | 0.595 | 0.37 |  |
| share: pulses | soil testing | -0.076 (0.047) | 0.109 | -0.054 (0.043) | 0.201 | 0.72 |  |
| share: pulses | custom hiring | -0.056 (0.049) | 0.252 | -0.002 (0.060) | 0.976 | 0.03 |  |
| share: pulses | cold storage | -0.171 (0.066) | 0.009 | -0.058 (0.055) | 0.298 | 0.34 | SIGNIF lost |
| share: pulses | farm-gate processing | -0.053 (0.051) | 0.303 | +0.004 (0.056) | 0.939 | -0.08 | SIGN |
| share: pulses | FPO | -0.069 (0.040) | 0.087 | -0.032 (0.038) | 0.393 | 0.47 |  |

## Every coefficient that changes sign or p = 0.05 significance (version W)

Cells compared: 280 (blocks solo, output, input, post-harvest; outcomes D0, D1, D2, evenness and 10 crop shares).
Sign changes: 31. Significance changes at 5 percent: 20. BH q changes at 5 percent: 7.

| block | outcome | facility | village b (p) | pop b (p) | change |
|---|---|---|---|---|---|
| solo | D0_richness | FPO | +2.376 (0.194) | +3.380 (0.024) | signif  |
| solo | D0_richness | regular market | -2.982 (0.110) | -3.483 (0.040) | signif  |
| solo | D1_exp_shannon | custom hiring | +0.078 (0.951) | -0.458 (0.670) | sign  |
| solo | D1_exp_shannon | seed centre | -0.350 (0.729) | +0.544 (0.517) | sign  |
| solo | D2_inv_simpson | seed centre | -0.361 (0.656) | +0.368 (0.584) | sign  |
| solo | evenness_D1_D0 | mandi | -0.018 (0.886) | +0.011 (0.921) | sign  |
| solo | evenness_D1_D0 | regular market | +0.005 (0.921) | -0.002 (0.961) | sign  |
| solo | evenness_D1_D0 | soil testing | -0.008 (0.896) | +0.017 (0.752) | sign  |
| solo | share: cereals | farm-gate processing | +0.030 (0.766) | -0.039 (0.660) | sign  |
| solo | share: cereals | fertiliser shop | -0.193 (0.026) | -0.216 (0.003) | q |
| solo | share: cereals | cold storage | +0.128 (0.309) | -0.071 (0.483) | sign  |
| solo | share: drugs_and_narcotics | cold storage | +0.010 (0.404) | -0.007 (0.204) | sign  |
| solo | share: fiber_crops | fertiliser shop | +0.139 (0.018) | +0.115 (0.008) | q |
| solo | share: fodder | custom hiring | -0.024 (0.117) | -0.033 (0.037) | signif  |
| solo | share: fodder | mandi | -0.006 (0.738) | +0.023 (0.266) | sign  |
| solo | share: fruits | custom hiring | +0.037 (0.030) | +0.026 (0.070) | signif  |
| solo | share: fruits | soil testing | -0.002 (0.920) | +0.001 (0.966) | sign  |
| solo | share: fruits | cold storage | +0.002 (0.929) | -0.004 (0.793) | sign  |
| solo | share: oilseeds | custom hiring | -0.018 (0.866) | +0.104 (0.203) | sign  |
| solo | share: oilseeds | cold storage | +0.054 (0.686) | +0.210 (0.015) | signif  |
| solo | share: pulses | farm-gate processing | -0.053 (0.303) | +0.004 (0.939) | sign  |
| solo | share: pulses | cold storage | -0.171 (0.009) | -0.058 (0.298) | signif  |
| solo | share: sugar | regular market | -0.096 (0.009) | -0.067 (0.017) | q |
| solo | share: vegetable | custom hiring | -0.008 (0.751) | +0.009 (0.697) | sign  |
| solo | share: vegetable | fertiliser shop | +0.037 (0.052) | +0.041 (0.039) | signif  |
| solo | share: vegetable | seed centre | +0.056 (0.084) | +0.049 (0.038) | signif  |
| solo | share: vegetable | cold storage | -0.016 (0.629) | +0.003 (0.907) | sign  |
| output | share: fiber_crops | regular market | -0.002 (0.965) | +0.001 (0.987) | sign  |
| output | share: fodder | mandi | -0.005 (0.787) | +0.021 (0.310) | sign  |
| output | share: pulses | weekly haat | -0.093 (0.042) | -0.068 (0.066) | signif  |
| output | share: spices | regular market | +0.011 (0.761) | -0.006 (0.885) | sign  |
| output | share: sugar | regular market | -0.074 (0.016) | -0.033 (0.203) | signif  |
| input | D0_richness | custom hiring | +0.282 (0.915) | -0.682 (0.770) | sign  |
| input | D0_richness | fertiliser shop | -0.567 (0.793) | +1.737 (0.351) | sign  |
| input | D1_exp_shannon | seed centre | -1.164 (0.456) | +0.201 (0.856) | sign  |
| input | D2_inv_simpson | seed centre | -1.018 (0.406) | +0.171 (0.847) | sign  |
| input | share: cereals | custom hiring | +0.015 (0.905) | -0.092 (0.425) | sign  |
| input | share: cereals | fertiliser shop | -0.232 (0.029) | -0.223 (0.007) | q |
| input | share: cereals | soil testing | +0.228 (0.137) | +0.266 (0.037) | signif  |
| input | share: fiber_crops | fertiliser shop | +0.140 (0.023) | +0.130 (0.005) | q |
| input | share: fiber_crops | seed centre | -0.159 (0.050) | -0.088 (0.094) | signif  |
| input | share: fodder | soil testing | +0.005 (0.775) | -0.005 (0.786) | sign  |
| input | share: oilseeds | custom hiring | -0.050 (0.673) | +0.063 (0.510) | sign  |
| input | share: oilseeds | soil testing | -0.201 (0.199) | -0.228 (0.042) | signif  |
| input | share: sugar | fertiliser shop | +0.062 (0.061) | +0.064 (0.015) | signif q |
| input | share: sugar | soil testing | -0.154 (0.035) | -0.083 (0.079) | signif  |
| input | share: vegetable | custom hiring | -0.054 (0.037) | -0.027 (0.222) | signif  |
| post | D2_inv_simpson | cold storage | -0.096 (0.941) | +0.154 (0.879) | sign  |
| post | share: cereals | FPO | -0.223 (0.007) | -0.126 (0.089) | signif  |
| post | share: cereals | cold storage | +0.234 (0.102) | -0.013 (0.909) | sign  |
| post | share: drugs_and_narcotics | cold storage | +0.001 (0.922) | -0.012 (0.155) | sign  |
| post | share: fruits | FPO | +0.033 (0.021) | +0.020 (0.060) | signif  |
| post | share: oilseeds | cold storage | +0.147 (0.315) | +0.284 (0.005) | signif q |
| post | share: pulses | cold storage | -0.155 (0.029) | -0.059 (0.357) | signif  |
| post | share: spices | farm-gate processing | +0.011 (0.913) | -0.003 (0.971) | sign  |
| post | share: vegetable | cold storage | -0.007 (0.873) | +0.008 (0.779) | sign  |

Same count for the other two weightings: Wf (regressors weighted, controls as on the site): sign changes 31, 5% significance changes 22; WA (Antyodaya 2019-20 population weights): sign changes 37, 5% significance changes 24.

Wf changes in detail:

| block | outcome | facility | village b (p) | Wf b (p) |
|---|---|---|---|---|
| solo | D0_richness | FPO | +2.376 (0.194) | +3.207 (0.036) |
| solo | D0_richness | regular market | -2.982 (0.110) | -3.479 (0.038) |
| solo | D1_exp_shannon | custom hiring | +0.078 (0.951) | -0.466 (0.675) |
| solo | D1_exp_shannon | seed centre | -0.350 (0.729) | +0.626 (0.460) |
| solo | D2_inv_simpson | seed centre | -0.361 (0.656) | +0.391 (0.564) |
| solo | evenness_D1_D0 | fertiliser shop | -0.017 (0.741) | +0.008 (0.838) |
| solo | evenness_D1_D0 | mandi | -0.018 (0.886) | +0.016 (0.888) |
| solo | evenness_D1_D0 | soil testing | -0.008 (0.896) | +0.028 (0.589) |
| solo | share: cereals | farm-gate processing | +0.030 (0.766) | -0.052 (0.552) |
| solo | share: cereals | FPO | -0.139 (0.057) | -0.155 (0.017) |
| solo | share: cereals | cold storage | +0.128 (0.309) | -0.082 (0.424) |
| solo | share: drugs_and_narcotics | cold storage | +0.010 (0.404) | -0.008 (0.181) |
| solo | share: fodder | mandi | -0.006 (0.738) | +0.026 (0.211) |
| solo | share: fruits | custom hiring | +0.037 (0.030) | +0.028 (0.068) |
| solo | share: fruits | soil testing | -0.002 (0.920) | +0.001 (0.963) |
| solo | share: fruits | cold storage | +0.002 (0.929) | -0.004 (0.806) |
| solo | share: oilseeds | custom hiring | -0.018 (0.866) | +0.105 (0.226) |
| solo | share: oilseeds | cold storage | +0.054 (0.686) | +0.215 (0.015) |
| solo | share: pulses | farm-gate processing | -0.053 (0.303) | +0.006 (0.914) |
| solo | share: pulses | cold storage | -0.171 (0.009) | -0.063 (0.268) |
| solo | share: sugar | fertiliser shop | +0.042 (0.174) | +0.066 (0.010) |
| solo | share: vegetable | custom hiring | -0.008 (0.751) | +0.005 (0.813) |
| solo | share: vegetable | fertiliser shop | +0.037 (0.052) | +0.040 (0.036) |
| solo | share: vegetable | seed centre | +0.056 (0.084) | +0.049 (0.041) |
| output | evenness_D1_D0 | mandi | -0.035 (0.787) | +0.003 (0.978) |
| output | share: cereals | regular market | +0.009 (0.930) | -0.023 (0.771) |
| output | share: fiber_crops | regular market | -0.002 (0.965) | +0.004 (0.931) |
| output | share: fodder | mandi | -0.005 (0.787) | +0.028 (0.195) |
| output | share: pulses | weekly haat | -0.093 (0.042) | -0.056 (0.129) |
| output | share: spices | regular market | +0.011 (0.761) | -0.003 (0.935) |
| output | share: sugar | regular market | -0.074 (0.016) | -0.021 (0.394) |
| input | D0_richness | custom hiring | +0.282 (0.915) | -0.539 (0.820) |
| input | D0_richness | fertiliser shop | -0.567 (0.793) | +1.551 (0.400) |
| input | D1_exp_shannon | seed centre | -1.164 (0.456) | +0.089 (0.934) |
| input | D2_inv_simpson | seed centre | -1.018 (0.406) | +0.107 (0.901) |
| input | share: cereals | custom hiring | +0.015 (0.905) | -0.094 (0.417) |
| input | share: cereals | soil testing | +0.228 (0.137) | +0.241 (0.043) |
| input | share: fiber_crops | seed centre | -0.159 (0.050) | -0.088 (0.092) |
| input | share: oilseeds | custom hiring | -0.050 (0.673) | +0.071 (0.475) |
| input | share: oilseeds | soil testing | -0.201 (0.199) | -0.238 (0.032) |
| input | share: sugar | fertiliser shop | +0.062 (0.061) | +0.079 (0.003) |
| input | share: sugar | soil testing | -0.154 (0.035) | -0.074 (0.106) |
| input | share: vegetable | custom hiring | -0.054 (0.037) | -0.029 (0.187) |
| post | D2_inv_simpson | cold storage | -0.096 (0.941) | +0.089 (0.931) |
| post | share: cereals | cold storage | +0.234 (0.102) | -0.008 (0.942) |
| post | share: drugs_and_narcotics | cold storage | +0.001 (0.922) | -0.012 (0.140) |
| post | share: fiber_crops | FPO | +0.089 (0.059) | +0.072 (0.048) |
| post | share: fruits | FPO | +0.033 (0.021) | +0.019 (0.056) |
| post | share: oilseeds | cold storage | +0.147 (0.315) | +0.290 (0.004) |
| post | share: pulses | cold storage | -0.155 (0.029) | -0.072 (0.270) |
| post | share: spices | farm-gate processing | +0.011 (0.913) | -0.002 (0.978) |
| post | share: sugar | FPO | +0.027 (0.329) | +0.047 (0.038) |
| post | share: vegetable | cold storage | -0.007 (0.873) | +0.005 (0.864) |

## Output-market block (mandi, haat, regular market together): diversity outcomes

| outcome | facility | village b (SE) | p | pop b (SE) | p | ratio | flag |
|---|---|---|---|---|---|---|---|
| D0_richness | mandi | +2.808 (3.279) | 0.392 | +2.025 (2.797) | 0.469 | 0.72 |  |
| D0_richness | weekly haat | +7.286 (1.587) | 0.000 | +6.105 (1.283) | 0.000 | 0.84 |  |
| D0_richness | regular market | -1.227 (2.041) | 0.548 | -1.203 (1.860) | 0.518 | 0.98 |  |
| D1_exp_shannon | mandi | +0.110 (2.143) | 0.959 | +0.169 (1.797) | 0.925 | 1.53 |  |
| D1_exp_shannon | weekly haat | +0.369 (0.842) | 0.661 | +0.371 (0.744) | 0.618 | 1.01 |  |
| D1_exp_shannon | regular market | -0.597 (0.827) | 0.470 | -0.857 (0.761) | 0.260 | 1.44 |  |
| D2_inv_simpson | mandi | +0.196 (1.804) | 0.914 | +0.350 (1.496) | 0.815 | 1.79 |  |
| D2_inv_simpson | weekly haat | +0.162 (0.624) | 0.796 | +0.249 (0.568) | 0.661 | 1.54 |  |
| D2_inv_simpson | regular market | -0.508 (0.616) | 0.410 | -0.765 (0.572) | 0.181 | 1.51 |  |
| evenness_D1_D0 | mandi | -0.035 (0.130) | 0.787 | -0.007 (0.117) | 0.951 | 0.20 |  |
| evenness_D1_D0 | weekly haat | -0.073 (0.037) | 0.050 | -0.051 (0.031) | 0.094 | 0.70 |  |
| evenness_D1_D0 | regular market | -0.013 (0.053) | 0.805 | -0.021 (0.044) | 0.641 | 1.56 |  |

## Input-supply block: diversity outcomes and the pulse and cereal shares

| outcome | facility | village b (SE) | p | pop b (SE) | p | ratio | flag |
|---|---|---|---|---|---|---|---|
| D0_richness | fertiliser shop | -0.567 (2.164) | 0.793 | +1.737 (1.864) | 0.351 | -3.06 | SIGN |
| D0_richness | seed centre | +5.588 (2.785) | 0.045 | +5.256 (2.147) | 0.014 | 0.94 |  |
| D0_richness | soil testing | -1.441 (2.547) | 0.572 | -2.032 (2.069) | 0.326 | 1.41 |  |
| D0_richness | custom hiring | +0.282 (2.647) | 0.915 | -0.682 (2.329) | 0.770 | -2.42 | SIGN |
| D1_exp_shannon | fertiliser shop | +0.482 (1.020) | 0.636 | +1.042 (0.913) | 0.254 | 2.16 |  |
| D1_exp_shannon | seed centre | -1.164 (1.562) | 0.456 | +0.201 (1.106) | 0.856 | -0.17 | SIGN |
| D1_exp_shannon | soil testing | +1.019 (1.665) | 0.540 | +0.409 (1.169) | 0.726 | 0.40 |  |
| D1_exp_shannon | custom hiring | -0.026 (1.310) | 0.984 | -1.129 (1.109) | 0.309 | 42.89 |  |
| share: cereals | fertiliser shop | -0.232 (0.106) | 0.029 | -0.223 (0.083) | 0.007 | 0.96 | q gained |
| share: cereals | seed centre | -0.044 (0.139) | 0.752 | -0.073 (0.103) | 0.479 | 1.66 |  |
| share: cereals | soil testing | +0.228 (0.153) | 0.137 | +0.266 (0.128) | 0.037 | 1.17 | SIGNIF gained |
| share: cereals | custom hiring | +0.015 (0.127) | 0.905 | -0.092 (0.116) | 0.425 | -6.10 | SIGN |
| share: pulses | fertiliser shop | -0.187 (0.047) | 0.000 | -0.144 (0.039) | 0.000 | 0.77 |  |
| share: pulses | seed centre | +0.053 (0.063) | 0.400 | +0.048 (0.047) | 0.300 | 0.91 |  |
| share: pulses | soil testing | -0.055 (0.064) | 0.395 | -0.039 (0.046) | 0.397 | 0.71 |  |
| share: pulses | custom hiring | +0.028 (0.050) | 0.579 | +0.056 (0.064) | 0.382 | 2.00 |  |

## Post-harvest block (storage, farm-gate processing, FPO): the site's FPO crop-share result

| outcome | facility | village b (SE) | p | pop b (SE) | p | ratio | flag |
|---|---|---|---|---|---|---|---|
| D0_richness | cold storage | +4.772 (3.233) | 0.140 | +3.092 (2.622) | 0.238 | 0.65 |  |
| D0_richness | farm-gate processing | +2.202 (2.175) | 0.311 | +2.430 (2.044) | 0.235 | 1.10 |  |
| D0_richness | FPO | +0.443 (2.118) | 0.834 | +1.883 (1.766) | 0.286 | 4.26 |  |
| D1_exp_shannon | cold storage | -0.314 (1.674) | 0.851 | -0.019 (1.297) | 0.988 | 0.06 |  |
| D1_exp_shannon | farm-gate processing | -1.059 (0.989) | 0.284 | -1.034 (0.892) | 0.247 | 0.98 |  |
| D1_exp_shannon | FPO | +2.371 (1.021) | 0.020 | +2.204 (0.815) | 0.007 | 0.93 |  |
| share: cereals | cold storage | +0.234 (0.143) | 0.102 | -0.013 (0.115) | 0.909 | -0.06 | SIGN q 0.254->0.925 |
| share: cereals | farm-gate processing | +0.080 (0.107) | 0.457 | +0.031 (0.100) | 0.759 | 0.38 |  q 0.762->0.843 |
| share: cereals | FPO | -0.223 (0.083) | 0.007 | -0.126 (0.074) | 0.089 | 0.57 | SIGNIF lost q 0.069->0.223 |
| share: fruits | cold storage | -0.037 (0.021) | 0.077 | -0.028 (0.015) | 0.061 | 0.75 |  q 0.254->0.204 |
| share: fruits | farm-gate processing | +0.030 (0.024) | 0.212 | +0.027 (0.021) | 0.203 | 0.91 |  q 0.539->0.676 |
| share: fruits | FPO | +0.033 (0.014) | 0.021 | +0.020 (0.011) | 0.060 | 0.62 | SIGNIF lost q 0.107->0.210 |
| share: pulses | cold storage | -0.155 (0.071) | 0.029 | -0.059 (0.064) | 0.357 | 0.38 | SIGNIF lost q 0.147->0.510 |
| share: pulses | farm-gate processing | +0.018 (0.046) | 0.690 | +0.039 (0.057) | 0.490 | 2.13 |  q 0.863->0.700 |
| share: pulses | FPO | -0.038 (0.045) | 0.393 | -0.030 (0.038) | 0.440 | 0.78 |  q 0.561->0.522 |
| share: vegetable | cold storage | -0.007 (0.043) | 0.873 | +0.008 (0.029) | 0.779 | -1.19 | SIGN q 0.922->0.925 |
| share: vegetable | farm-gate processing | -0.085 (0.026) | 0.001 | -0.067 (0.022) | 0.003 | 0.79 |  q 0.012->0.026 |
| share: vegetable | FPO | +0.058 (0.029) | 0.043 | +0.045 (0.023) | 0.048 | 0.77 |  q 0.145->0.210 |

## Crop-share results that survive Benjamini-Hochberg (q < 0.05), by version

- solo / U: regular market on pulses (-9.3 pts, q 0.043); regular market on sugar (-9.6 pts, q 0.045); weekly haat on sugar (+12.4 pts, q 0.003); fertiliser shop on pulses (-17.2 pts, q 0.001)
- solo / W: regular market on pulses (-8.2 pts, q 0.024); weekly haat on sugar (+11.0 pts, q 0.001); fertiliser shop on cereals (-21.6 pts, q 0.016); fertiliser shop on fiber_crops (+11.5 pts, q 0.027); fertiliser shop on pulses (-12.3 pts, q 0.005)
- output / U: regular market on pulses (-12.2 pts, q 0.006); weekly haat on sugar (+11.1 pts, q 0.011)
- output / W: regular market on pulses (-11.1 pts, q 0.005); weekly haat on sugar (+10.1 pts, q 0.003)
- input / U: fertiliser shop on pulses (-18.7 pts, q 0.001)
- input / W: fertiliser shop on cereals (-22.3 pts, q 0.025); fertiliser shop on fiber_crops (+13.0 pts, q 0.025); fertiliser shop on pulses (-14.4 pts, q 0.003); fertiliser shop on sugar (+6.4 pts, q 0.038)
- post / U: farm-gate processing on vegetable (-8.5 pts, q 0.012)
- post / W: cold storage on oilseeds (+28.4 pts, q 0.048); farm-gate processing on vegetable (-6.7 pts, q 0.026)

## Weekly haat on the count of crops: the page's robustness table

| specification | n | village b (SE) [95% CI] | pop b (SE) [95% CI] | Wf b (p) | WA b (p) |
|---|---|---|---|---|---|
| no adjustment | 606 / 606 | +16.40 (2.36) [+11.78, +21.03] | +14.42 (1.83) [+10.83, +18.02] | +14.42 (0.0000) | +15.30 (0.0000) |
| state baseline only | 606 / 606 | +7.21 (1.49) [+4.28, +10.13] | +6.02 (1.21) [+3.65, +8.38] | +6.02 (0.0000) | +6.35 (0.0000) |
| + irrigation | 606 / 606 | +7.00 (1.48) [+4.10, +9.90] | +5.94 (1.19) [+3.61, +8.28] | +5.94 (0.0000) | +6.29 (0.0000) |
| all measures | 606 / 606 | +7.36 (1.59) [+4.26, +10.47] | +6.28 (1.23) [+3.86, +8.69] | +6.19 (0.0000) | +6.52 (0.0000) |
| drop shared parents | 542 / 542 | +7.92 (1.72) [+4.55, +11.29] | +6.91 (1.33) [+4.31, +9.52] | +6.87 (0.0000) | +7.07 (0.0000) |
| drop districts with no haat | 576 / 576 | +6.46 (1.55) [+3.43, +9.50] | +5.64 (1.19) [+3.30, +7.98] | +5.54 (0.0000) | +5.92 (0.0000) |
| with nine other facilities | 606 / 606 | +6.88 (1.61) [+3.72, +10.04] | +5.22 (1.35) [+2.56, +7.87] | +5.15 (0.0001) | +5.84 (0.0000) |

Haats per 100,000 people (a density, already population-normalised, so no weighted version exists): +0.0554 (SE 0.0205, p 0.0068), n 606.
One SD: village share 0.117 -> +0.86 crops (+0.19 SE-scaled); population share 0.150 -> +0.94 crops. Mean D0 21.4.

## Markets and the irrigation downslope

Interaction of the facility measure (standardised) with irrigation squared, D1 outcome. 'curv' is the coefficient on irrigation squared one SD below and above the mean; 'peak' is the irrigation share at the turning point.

| measure | version | curvature low | curvature high | interaction (SE) | p | peak low | peak high |
|---|---|---|---|---|---|---|---|
| mandi | U | -9.01 | -3.64 | +2.69 (1.24) | 0.030 | 0.36 | 0.13 |
| fertiliser shop | U | -8.18 | -5.10 | +1.54 (1.06) | 0.144 | 0.32 | 0.28 |
| output-market index | U | -8.48 | -4.38 | +2.05 (1.24) | 0.098 | 0.33 | 0.26 |
| input-supply index | U | -8.74 | -4.26 | +2.24 (1.17) | 0.056 | 0.34 | 0.23 |
| post-harvest index | U | -8.64 | -4.35 | +2.14 (1.24) | 0.083 | 0.33 | 0.24 |
| mandi | W | -8.21 | -4.06 | +2.08 (1.10) | 0.060 | 0.36 | 0.15 |
| fertiliser shop | W | -6.33 | -6.46 | -0.07 (1.11) | 0.952 | 0.30 | 0.28 |
| output-market index | W | -7.82 | -5.01 | +1.40 (1.19) | 0.237 | 0.33 | 0.27 |
| input-supply index | W | -6.92 | -5.91 | +0.51 (1.27) | 0.690 | 0.32 | 0.28 |
| post-harvest index | W | -7.71 | -5.06 | +1.33 (1.32) | 0.315 | 0.31 | 0.27 |
| mandi | Wf | -8.32 | -4.26 | +2.03 (1.11) | 0.069 | 0.36 | 0.17 |
| fertiliser shop | Wf | -6.55 | -6.55 | -0.00 (1.11) | 1.000 | 0.30 | 0.27 |
| output-market index | Wf | -8.48 | -4.38 | +2.05 (1.24) | 0.098 | 0.33 | 0.26 |
| input-supply index | Wf | -8.74 | -4.26 | +2.24 (1.17) | 0.056 | 0.34 | 0.23 |
| post-harvest index | Wf | -8.64 | -4.35 | +2.14 (1.24) | 0.083 | 0.33 | 0.24 |
| mandi | WA | -8.58 | -4.01 | +2.29 (1.04) | 0.028 | 0.35 | 0.15 |
| fertiliser shop | WA | -6.40 | -6.49 | -0.04 (1.11) | 0.968 | 0.30 | 0.26 |
| output-market index | WA | -8.07 | -4.97 | +1.55 (1.16) | 0.180 | 0.32 | 0.27 |
| input-supply index | WA | -7.06 | -5.93 | +0.56 (1.26) | 0.654 | 0.32 | 0.27 |
| post-harvest index | WA | -7.34 | -5.51 | +0.91 (1.34) | 0.496 | 0.30 | 0.27 |

## The six groups of district

Rule from script 89: facility supply = mean percentile rank across the nine non-haat facilities, cut into thirds; weekly haat share above or below its median. Recomputed with population shares.

| comparison | density band changes | haat half changes | six-group type changes | band moves by 2 |
|---|---|---|---|---|
| Census-2011 population weights | 104 of 606 (17.2%) | 54 | 145 (23.9%) | 0 |
| Antyodaya-2019 population weights | 104 of 606 (17.2%) | 56 | 149 (24.6%) | 0 |

Density-rank correlation, village against population shares: Pearson 0.951, Spearman 0.942.
Cuts (density terciles, haat median): village 0.401, 0.597, 0.126; population 0.398, 0.589, 0.179.

Transition of density bands (rows village, columns population):

```
band_W  bottom  middle  top
band_u                     
bottom     171      31    0
middle      31     150   21
top          0      21  181
```

Six-group sizes, village against population:

| group | village n | population n |
|---|---|---|
| Few | 100 | 101 |
| Haats, little else | 102 | 101 |
| Some, few haats | 91 | 86 |
| Some, with haats | 111 | 116 |
| Served, few haats | 112 | 116 |
| Served, with haats | 90 | 86 |

What the bands show, village against population, unadjusted means of the three outcomes the page quotes:

| group (population definition) | n | D0 | D1 | cereal share | fert-shop village share | fert-shop pop share |
|---|---|---|---|---|---|---|
| Few (village) | 100 | 18.7 | 4.60 | 0.771 | 0.055 | 0.106 |
| Haats, little else (village) | 102 | 21.0 | 4.21 | 0.725 | 0.076 | 0.134 |
| Some, few haats (village) | 91 | 20.2 | 5.03 | 0.636 | 0.136 | 0.212 |
| Some, with haats (village) | 111 | 23.3 | 4.69 | 0.707 | 0.134 | 0.218 |
| Served, few haats (village) | 112 | 20.9 | 5.39 | 0.530 | 0.334 | 0.444 |
| Served, with haats (village) | 90 | 24.3 | 5.87 | 0.605 | 0.309 | 0.409 |
| Few (pop) | 101 | 17.7 | 4.36 | 0.799 | 0.070 | 0.112 |
| Haats, little else (pop) | 101 | 20.6 | 4.25 | 0.753 | 0.075 | 0.125 |
| Some, few haats (pop) | 86 | 19.6 | 4.69 | 0.688 | 0.139 | 0.207 |
| Some, with haats (pop) | 116 | 24.4 | 4.77 | 0.659 | 0.139 | 0.226 |
| Served, few haats (pop) | 116 | 21.3 | 5.37 | 0.517 | 0.340 | 0.445 |
| Served, with haats (pop) | 86 | 24.4 | 6.40 | 0.570 | 0.275 | 0.401 |

Within-state tests (state fixed effects, errors clustered by state, no other controls): outcome on middle-third and top-third dummies (bottom is the reference) and the haat-rich half.

| outcome | version | middle third b (p) | top third b (p) | haat-rich half b (p) | joint p, thirds |
|---|---|---|---|---|---|
| D1_exp_shannon | village | -0.202 (0.296) | +0.126 (0.672) | +0.202 (0.448) | 0.228 |
| D1_exp_shannon | pop | -0.310 (0.142) | +0.129 (0.715) | +0.164 (0.357) | 0.053 |
| share_cereals | village | -0.009 (0.600) | -0.013 (0.683) | +0.009 (0.764) | 0.847 |
| share_cereals | pop | -0.040 (0.001) | -0.042 (0.133) | -0.002 (0.924) | 0.001 |
| D0_richness | village | -0.275 (0.513) | +0.567 (0.232) | +1.337 (0.003) | 0.137 |
| D0_richness | pop | +0.006 (0.983) | +0.792 (0.132) | +1.135 (0.006) | 0.207 |
| evenness_D1_D0 | village | -0.004 (0.625) | -0.003 (0.785) | -0.008 (0.410) | 0.884 |
| evenness_D1_D0 | pop | -0.011 (0.248) | -0.005 (0.750) | -0.004 (0.605) | 0.443 |