# Census market panel: raw log

## Villages

SHRUG village rows: 1991 564,855; 2001 527,920; 2011 588,973; Mission Antyodaya 521,223.
1991 villages with both the market flag and day fields: 405,251. Market flag = 1: 10.1%. Of those, a weekday market day: 85.7%; daily: 13.2%; fortnightly or monthly only: 0.3%; no frequency at all: 1.1%. Weekday day recorded where flag = 0: 28 villages.
1991 day fields missing while the flag is present: 107,373 villages (20.9% of flagged rows).
2011: weekly haat 15.9% of villages, mandi or regular market 12.3%, both 7.3%.
Mission Antyodaya 2019-20: the market item is ONE answer per village (mandi 1, regular market 2, weekly haat 3, none 4; question 71 of the MA 2020 data mapping), so a village never carries two market types. Villages with more than one market type in the SHRUG file: 82.
Villages in the 2011 directory with a district key: 588,973; 2011 rural population 886 million.

## Comparability of the measures (district level, 2011 population weights)

| pair | districts with >= 75% coverage | mean early | mean late | mean change | SD change | r(early, late) |
|---|---|---|---|---|---|---|
| haat | 435 | 0.132 | 0.251 | +0.118 | 0.174 | 0.48 |
| daily | 428 | 0.033 | 0.257 | +0.224 | 0.232 | 0.37 |
| anymkt | 505 | 0.180 | 0.363 | +0.183 | 0.216 | 0.48 |
| acs | 568 | 0.263 | 0.281 | +0.018 | 0.137 | 0.83 |
| haatMA | 412 | 0.258 | 0.227 | -0.031 | 0.173 | 0.44 |
| dailyMA | 412 | 0.218 | 0.154 | -0.064 | 0.202 | 0.33 |

States with median 1991-2011 market coverage below 0.75: JAMMU AND KASHMIR 0.00, LADAKH 0.00, CHANDIGARH 0.00, GUJARAT 0.00, HARYANA 0.00, THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU 0.09, ASSAM 0.18, MAHARASHTRA 0.23, ODISHA 0.23, ARUNACHAL PRADESH 0.41
States with median 2001-2011 credit-society coverage below 0.75: ODISHA 0.40, MIZORAM 0.70, MEGHALAYA 0.69, WEST BENGAL 0.13

## Analysis frame

Crop-record districts in the 606: 606. With both crop windows: 547. With the 1991-2011 haat change: 425. With the 2001-2011 credit-society change: 550. With the census irrigation change 2001-2011: 481.
Mean early-window values: D0 21.44, D0 at 0.5% 10.02, D1 5.042. Mean changes: dD0 -0.00, dD0_05 -0.57, dD1 -0.155.

## First-difference regressions (state FE; SE on 2011 parent; state-clustered t(G-1); wild cluster bootstrap by state, Webb, 9,999 draws)

| spec | outcome | term | n | b | SE parent | p parent | p state t(G-1) | p wild | G |
|---|---|---|---|---|---|---|---|---|---|
| haat + daily | count of crops | d_haat | 338 | +1.004 | 1.183 | 0.396 | 0.442 | 0.433 | 17 |
| haat + daily | count of crops | d_daily | 338 | -0.750 | 1.050 | 0.475 | 0.463 | 0.473 | 17 |
| haat + daily | count at 0.5% floor | d_haat | 338 | +0.246 | 0.670 | 0.714 | 0.735 | 0.728 | 17 |
| haat + daily | count at 0.5% floor | d_daily | 338 | -0.591 | 0.575 | 0.304 | 0.207 | 0.168 | 17 |
| haat + daily | effective number | d_haat | 338 | +0.624 | 0.455 | 0.170 | 0.290 | 0.296 | 17 |
| haat + daily | effective number | d_daily | 338 | -0.283 | 0.336 | 0.400 | 0.439 | 0.431 | 17 |
| haat + daily | count, drop6 | d_haat | 338 | +1.245 | 1.010 | 0.218 | 0.231 | 0.236 | 17 |
| haat + daily | count, drop6 | d_daily | 338 | -0.571 | 0.841 | 0.498 | 0.482 | 0.470 | 17 |
| haat + daily | count 0.5%, drop6 | d_haat | 338 | +0.147 | 0.677 | 0.828 | 0.821 | 0.823 | 17 |
| haat + daily | count 0.5%, drop6 | d_daily | 338 | -0.429 | 0.554 | 0.438 | 0.367 | 0.351 | 17 |
| haat + daily | effective, drop6 | d_haat | 338 | +0.612 | 0.408 | 0.133 | 0.235 | 0.244 | 17 |
| haat + daily | effective, drop6 | d_daily | 338 | -0.299 | 0.308 | 0.331 | 0.388 | 0.381 | 17 |
| haat only | count of crops | d_haat | 338 | +0.736 | 1.185 | 0.534 | 0.514 | 0.507 | 17 |
| haat only | count at 0.5% floor | d_haat | 338 | +0.034 | 0.643 | 0.957 | 0.960 | 0.961 | 17 |
| haat only | effective number | d_haat | 338 | +0.523 | 0.468 | 0.263 | 0.366 | 0.394 | 17 |
| haat only | count, drop6 | d_haat | 338 | +1.041 | 1.000 | 0.297 | 0.254 | 0.242 | 17 |
| haat only | count 0.5%, drop6 | d_haat | 338 | -0.006 | 0.663 | 0.992 | 0.992 | 0.992 | 17 |
| haat only | effective, drop6 | d_haat | 338 | +0.505 | 0.427 | 0.237 | 0.323 | 0.347 | 17 |
| any market | count of crops | d_anymkt | 357 | +0.245 | 0.841 | 0.771 | 0.786 | 0.781 | 16 |
| any market | count at 0.5% floor | d_anymkt | 357 | +0.209 | 0.542 | 0.700 | 0.732 | 0.740 | 16 |
| any market | effective number | d_anymkt | 357 | +0.199 | 0.292 | 0.496 | 0.638 | 0.670 | 16 |
| any market | count, drop6 | d_anymkt | 357 | +0.322 | 0.664 | 0.628 | 0.636 | 0.633 | 16 |
| any market | count 0.5%, drop6 | d_anymkt | 357 | +0.230 | 0.603 | 0.703 | 0.713 | 0.713 | 16 |
| any market | effective, drop6 | d_anymkt | 357 | +0.206 | 0.279 | 0.460 | 0.585 | 0.590 | 16 |
| credit societies | count of crops | d_acs | 431 | -1.071 | 0.982 | 0.275 | 0.249 | 0.253 | 22 |
| credit societies | count at 0.5% floor | d_acs | 431 | -1.288 | 0.628 | 0.040 | 0.112 | 0.258 | 22 |
| credit societies | effective number | d_acs | 431 | -0.701 | 0.448 | 0.118 | 0.097 | 0.271 | 22 |
| credit societies | count, drop6 | d_acs | 431 | +0.268 | 0.971 | 0.783 | 0.684 | 0.677 | 22 |
| credit societies | count 0.5%, drop6 | d_acs | 431 | -1.339 | 0.653 | 0.040 | 0.148 | 0.322 | 22 |
| credit societies | effective, drop6 | d_acs | 431 | -0.596 | 0.406 | 0.143 | 0.136 | 0.316 | 22 |
| haat + daily + credit | count of crops | d_haat | 337 | +1.077 | 1.204 | 0.371 | 0.427 | 0.432 | 17 |
| haat + daily + credit | count of crops | d_daily | 337 | -0.748 | 1.057 | 0.479 | 0.462 | 0.457 | 17 |
| haat + daily + credit | count of crops | d_acs | 337 | -1.002 | 1.523 | 0.511 | 0.512 | 0.569 | 17 |
| haat + daily + credit | count at 0.5% floor | d_haat | 337 | +0.322 | 0.681 | 0.637 | 0.668 | 0.669 | 17 |
| haat + daily + credit | count at 0.5% floor | d_daily | 337 | -0.574 | 0.574 | 0.318 | 0.216 | 0.190 | 17 |
| haat + daily + credit | count at 0.5% floor | d_acs | 337 | -1.048 | 0.964 | 0.277 | 0.454 | 0.514 | 17 |
| haat + daily + credit | effective number | d_haat | 337 | +0.723 | 0.459 | 0.115 | 0.270 | 0.288 | 17 |
| haat + daily + credit | effective number | d_daily | 337 | -0.251 | 0.328 | 0.443 | 0.492 | 0.502 | 17 |
| haat + daily + credit | effective number | d_acs | 337 | -1.353 | 0.617 | 0.028 | 0.180 | 0.315 | 17 |
| haat + daily + credit | count, drop6 | d_haat | 337 | +1.204 | 1.031 | 0.243 | 0.255 | 0.249 | 17 |
| haat + daily + credit | count, drop6 | d_daily | 337 | -0.616 | 0.848 | 0.468 | 0.447 | 0.439 | 17 |
| haat + daily + credit | count, drop6 | d_acs | 337 | +0.562 | 1.414 | 0.691 | 0.537 | 0.541 | 17 |
| haat + daily + credit | count 0.5%, drop6 | d_haat | 337 | +0.210 | 0.688 | 0.760 | 0.758 | 0.749 | 17 |
| haat + daily + credit | count 0.5%, drop6 | d_daily | 337 | -0.427 | 0.554 | 0.441 | 0.376 | 0.383 | 17 |
| haat + daily + credit | count 0.5%, drop6 | d_acs | 337 | -0.872 | 0.954 | 0.360 | 0.540 | 0.633 | 17 |
| haat + daily + credit | effective, drop6 | d_haat | 337 | +0.694 | 0.412 | 0.092 | 0.226 | 0.238 | 17 |
| haat + daily + credit | effective, drop6 | d_daily | 337 | -0.276 | 0.301 | 0.360 | 0.429 | 0.442 | 17 |
| haat + daily + credit | effective, drop6 | d_acs | 337 | -1.122 | 0.579 | 0.053 | 0.212 | 0.355 | 17 |
| haat + daily, irrigation 1991-2011 | count of crops | d_haat | 355 | +0.946 | 1.158 | 0.414 | 0.484 | 0.483 | 18 |
| haat + daily, irrigation 1991-2011 | count of crops | d_daily | 355 | -1.131 | 1.092 | 0.300 | 0.300 | 0.422 | 18 |
| haat + daily, irrigation 1991-2011 | count at 0.5% floor | d_haat | 355 | +0.162 | 0.643 | 0.801 | 0.816 | 0.819 | 18 |
| haat + daily, irrigation 1991-2011 | count at 0.5% floor | d_daily | 355 | -0.536 | 0.576 | 0.352 | 0.215 | 0.207 | 18 |
| haat + daily, irrigation 1991-2011 | effective number | d_haat | 355 | +0.481 | 0.464 | 0.300 | 0.405 | 0.419 | 18 |
| haat + daily, irrigation 1991-2011 | effective number | d_daily | 355 | -0.228 | 0.357 | 0.523 | 0.571 | 0.579 | 18 |
| haat + daily, irrigation 1991-2011 | count, drop6 | d_haat | 355 | +1.130 | 0.985 | 0.251 | 0.290 | 0.291 | 18 |
| haat + daily, irrigation 1991-2011 | count, drop6 | d_daily | 355 | -0.850 | 0.864 | 0.325 | 0.295 | 0.364 | 18 |
| haat + daily, irrigation 1991-2011 | count 0.5%, drop6 | d_haat | 355 | +0.048 | 0.647 | 0.941 | 0.938 | 0.936 | 18 |
| haat + daily, irrigation 1991-2011 | count 0.5%, drop6 | d_daily | 355 | -0.326 | 0.541 | 0.547 | 0.456 | 0.443 | 18 |
| haat + daily, irrigation 1991-2011 | effective, drop6 | d_haat | 355 | +0.455 | 0.422 | 0.280 | 0.368 | 0.376 | 18 |
| haat + daily, irrigation 1991-2011 | effective, drop6 | d_daily | 355 | -0.260 | 0.319 | 0.415 | 0.475 | 0.472 | 18 |
| haat + daily, with 1991 levels | count of crops | d_haat | 338 | +0.899 | 1.431 | 0.530 | 0.492 | 0.493 | 17 |
| haat + daily, with 1991 levels | count of crops | d_daily | 338 | -1.017 | 1.286 | 0.429 | 0.386 | 0.376 | 17 |
| haat + daily, with 1991 levels | count at 0.5% floor | d_haat | 338 | -0.768 | 0.797 | 0.335 | 0.420 | 0.447 | 17 |
| haat + daily, with 1991 levels | count at 0.5% floor | d_daily | 338 | +0.251 | 0.668 | 0.707 | 0.699 | 0.697 | 17 |
| haat + daily, with 1991 levels | effective number | d_haat | 338 | +0.312 | 0.541 | 0.565 | 0.611 | 0.621 | 17 |
| haat + daily, with 1991 levels | effective number | d_daily | 338 | -0.073 | 0.434 | 0.867 | 0.887 | 0.889 | 17 |
| haat + daily, with 1991 levels | count, drop6 | d_haat | 338 | +1.325 | 1.208 | 0.273 | 0.231 | 0.230 | 17 |
| haat + daily, with 1991 levels | count, drop6 | d_daily | 338 | -0.876 | 1.050 | 0.404 | 0.402 | 0.395 | 17 |
| haat + daily, with 1991 levels | count 0.5%, drop6 | d_haat | 338 | -0.844 | 0.773 | 0.275 | 0.316 | 0.357 | 17 |
| haat + daily, with 1991 levels | count 0.5%, drop6 | d_daily | 338 | +0.396 | 0.626 | 0.527 | 0.552 | 0.569 | 17 |
| haat + daily, with 1991 levels | effective, drop6 | d_haat | 338 | +0.301 | 0.480 | 0.531 | 0.581 | 0.598 | 17 |
| haat + daily, with 1991 levels | effective, drop6 | d_daily | 338 | -0.091 | 0.390 | 0.815 | 0.851 | 0.857 | 17 |
| haat + daily, no irrigation control | count of crops | d_haat | 384 | +0.747 | 1.040 | 0.473 | 0.516 | 0.519 | 22 |
| haat + daily, no irrigation control | count of crops | d_daily | 384 | -1.202 | 1.044 | 0.250 | 0.262 | 0.383 | 22 |
| haat + daily, no irrigation control | count at 0.5% floor | d_haat | 384 | +0.017 | 0.600 | 0.977 | 0.976 | 0.976 | 22 |
| haat + daily, no irrigation control | count at 0.5% floor | d_daily | 384 | -0.443 | 0.564 | 0.432 | 0.324 | 0.316 | 22 |
| haat + daily, no irrigation control | effective number | d_haat | 384 | +0.305 | 0.423 | 0.472 | 0.553 | 0.560 | 22 |
| haat + daily, no irrigation control | effective number | d_daily | 384 | -0.102 | 0.348 | 0.769 | 0.805 | 0.813 | 22 |
| haat + daily, no irrigation control | count, drop6 | d_haat | 384 | +0.822 | 0.897 | 0.359 | 0.376 | 0.368 | 22 |
| haat + daily, no irrigation control | count, drop6 | d_daily | 384 | -0.901 | 0.827 | 0.276 | 0.249 | 0.335 | 22 |
| haat + daily, no irrigation control | count 0.5%, drop6 | d_haat | 384 | -0.185 | 0.589 | 0.754 | 0.727 | 0.717 | 22 |
| haat + daily, no irrigation control | count 0.5%, drop6 | d_daily | 384 | -0.329 | 0.521 | 0.528 | 0.445 | 0.432 | 22 |
| haat + daily, no irrigation control | effective, drop6 | d_haat | 384 | +0.279 | 0.389 | 0.473 | 0.542 | 0.536 | 22 |
| haat + daily, no irrigation control | effective, drop6 | d_daily | 384 | -0.135 | 0.311 | 0.663 | 0.715 | 0.718 | 22 |

One-SD effects, main spec (haat + daily, all crops):
  d_haat on dD0: SD of change 0.145; one SD -> +0.146 (95% CI -0.191 to +0.484)
  d_daily on dD0: SD of change 0.220; one SD -> -0.165 (95% CI -0.618 to +0.288)
  d_haat on dD0_05: SD of change 0.145; one SD -> +0.036 (95% CI -0.155 to +0.227)
  d_daily on dD0_05: SD of change 0.220; one SD -> -0.130 (95% CI -0.378 to +0.118)
  d_haat on dD1: SD of change 0.145; one SD -> +0.091 (95% CI -0.039 to +0.221)
  d_daily on dD1: SD of change 0.220; one SD -> -0.062 (95% CI -0.207 to +0.083)

## Cross-check: the site's cross-section with the CENSUS 2011 haat in place of Mission Antyodaya

Correlation across the 606 of 2011 census haat reach with MA 2019-20 haat reach (same village set): 0.44; census 2011 mandi/regular with MA mandi or regular: 0.33.
