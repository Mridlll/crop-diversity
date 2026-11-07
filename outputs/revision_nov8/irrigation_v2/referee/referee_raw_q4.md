
## R5. Q4: long differences

Correlation of the 1991-2011 change in irrigated share with the change in log gross cropped area: to 2009-13 0.207, to 2015-19 0.133; with the 1991 level 0.009
Check: rebuilt relative-threshold change equals d_D0_05 to 2009-13: max abs diff 4.44e-16

| outcome | diversity to | sample | control | b dirr | state SE | p t(G-1) | p wild | n |
|---|---|---|---|---|---|---|---|---|
| D0 at 0.5% (as in note) | cs0913 | all | - | -1.755 | 0.639 | 0.012 | 0.021 | 454 |
| D0 at 0.5% | cs0913 | all | dlgca_cs0913 | -1.801 | 0.650 | 0.012 | 0.027 | 454 |
| count >= 0.5% of EARLY area (fixed ha) | cs0913 | all | - | -0.947 | 0.810 | 0.257 | 0.246 | 454 |
| count >= 0.5% of EARLY area (fixed ha) | cs0913 | all | dlgca_cs0913 | -1.657 | 0.753 | 0.040 | 0.043 | 454 |
| change in log GCA | cs0913 | all | - | 0.219 | 0.172 | 0.217 | 0.235 | 454 |
| D0 at 0.5% (as in note) | cs0913 | area-stable | - | -1.737 | 0.714 | 0.025 | 0.031 | 403 |
| D0 at 0.5% | cs0913 | area-stable | dlgca_cs0913 | -1.729 | 0.718 | 0.026 | 0.030 | 403 |
| count >= 0.5% of EARLY area (fixed ha) | cs0913 | area-stable | - | -1.292 | 0.747 | 0.099 | 0.107 | 403 |
| count >= 0.5% of EARLY area (fixed ha) | cs0913 | area-stable | dlgca_cs0913 | -1.501 | 0.798 | 0.075 | 0.072 | 403 |
| change in log GCA | cs0913 | area-stable | - | 0.071 | 0.088 | 0.429 | 0.428 | 403 |
| D0 at 0.5% (as in note) | w1517 | all | - | -1.466 | 0.860 | 0.104 | 0.116 | 447 |
| D0 at 0.5% | w1517 | all | dlgca_w1517 | -1.516 | 0.855 | 0.092 | 0.107 | 447 |
| count >= 0.5% of EARLY area (fixed ha) | w1517 | all | - | -1.057 | 1.159 | 0.373 | 0.369 | 447 |
| count >= 0.5% of EARLY area (fixed ha) | w1517 | all | dlgca_w1517 | -1.674 | 0.934 | 0.088 | 0.092 | 447 |
| change in log GCA | w1517 | all | - | 0.208 | 0.163 | 0.217 | 0.246 | 447 |
| D0 at 0.5% (as in note) | w1517 | area-stable | - | -1.479 | 0.848 | 0.097 | 0.113 | 352 |
| D0 at 0.5% | w1517 | area-stable | dlgca_w1517 | -1.474 | 0.855 | 0.101 | 0.117 | 352 |
| count >= 0.5% of EARLY area (fixed ha) | w1517 | area-stable | - | -1.350 | 0.920 | 0.159 | 0.157 | 352 |
| count >= 0.5% of EARLY area (fixed ha) | w1517 | area-stable | dlgca_w1517 | -1.413 | 0.925 | 0.143 | 0.127 | 352 |
| change in log GCA | w1517 | area-stable | - | 0.025 | 0.084 | 0.772 | 0.767 | 352 |

Leave-one-state-out, dD0_05 on dirr 1991-2011 (all districts):
- to cs0913: b range -2.02 to -1.32; max state p 0.028; states whose removal moves b most: BIHAR -1.32 (p 0.011); NAGALAND -2.02 (p 0.001); PUNJAB -1.98 (p 0.001); WEST BENGAL -1.90 (p 0.003)
- to w1517: b range -2.04 to -1.01; max state p 0.190; states whose removal moves b most: MADHYA PRADESH -2.04 (p 0.013); BIHAR -1.01 (p 0.190); NAGALAND -1.74 (p 0.046); HARYANA -1.21 (p 0.158)

- adding the 1991 level, to cs0913: b -1.948, state SE 0.733, t -2.66, G 21, n 454: p normal 0.008, p t(G-1) 0.015, p wild-cluster (Webb, 9999 draws) 0.024
- adding the early D0 at 0.5% (regression to the mean), to cs0913: b -1.576, state SE 0.701, t -2.25, G 21, n 454: p normal 0.025, p t(G-1) 0.036, p wild-cluster (Webb, 9999 draws) 0.036
- adding the 1991 level, to w1517: b -2.026, state SE 0.994, t -2.04, G 21, n 447: p normal 0.041, p t(G-1) 0.055, p wild-cluster (Webb, 9999 draws) 0.069
- adding the early D0 at 0.5% (regression to the mean), to w1517: b -1.181, state SE 0.630, t -1.88, G 21, n 447: p normal 0.061, p t(G-1) 0.075, p wild-cluster (Webb, 9999 draws) 0.078
- 1991 measured as reported total only, dD0_05 to cs0913: b -1.383, state SE 0.704, t -1.96, G 18, n 382: p normal 0.050, p t(G-1) 0.066, p wild-cluster (Webb, 9999 draws) 0.157
  same, without Odisha/Assam/Sikkim/Goa (where the variants differ): b -1.360, state SE 0.802, t -1.69, G 15, n 326: p normal 0.090, p t(G-1) 0.112, p wild-cluster (Webb, 9999 draws) 0.229
  main 1991 measure on that same reduced sample: b -1.958, state SE 0.794, t -2.46, G 18, n 398: p normal 0.014, p t(G-1) 0.025, p wild-cluster (Webb, 9999 draws) 0.041
- 1991 measured as reported total only, dD0_05 to w1517: b -2.652, state SE 0.518, t -5.12, G 18, n 375: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.011
  same, without Odisha/Assam/Sikkim/Goa (where the variants differ): b -2.799, state SE 0.590, t -4.74, G 15, n 319: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.023
  main 1991 measure on that same reduced sample: b -1.457, state SE 1.130, t -1.29, G 18, n 391: p normal 0.197, p t(G-1) 0.215, p wild-cluster (Webb, 9999 draws) 0.232
- 1991 measured as sum of sources only, dD0_05 to cs0913: b -0.619, state SE 1.206, t -0.51, G 21, n 454: p normal 0.608, p t(G-1) 0.613, p wild-cluster (Webb, 9999 draws) 0.643
  same, without Odisha/Assam/Sikkim/Goa (where the variants differ): b -1.952, state SE 0.795, t -2.46, G 18, n 398: p normal 0.014, p t(G-1) 0.025, p wild-cluster (Webb, 9999 draws) 0.040
  main 1991 measure on that same reduced sample: b -1.958, state SE 0.794, t -2.46, G 18, n 398: p normal 0.014, p t(G-1) 0.025, p wild-cluster (Webb, 9999 draws) 0.045
- 1991 measured as sum of sources only, dD0_05 to w1517: b -0.895, state SE 0.897, t -1.00, G 21, n 447: p normal 0.318, p t(G-1) 0.330, p wild-cluster (Webb, 9999 draws) 0.333
  same, without Odisha/Assam/Sikkim/Goa (where the variants differ): b -1.459, state SE 1.130, t -1.29, G 18, n 391: p normal 0.197, p t(G-1) 0.214, p wild-cluster (Webb, 9999 draws) 0.226
  main 1991 measure on that same reduced sample: b -1.457, state SE 1.130, t -1.29, G 18, n 391: p normal 0.197, p t(G-1) 0.215, p wild-cluster (Webb, 9999 draws) 0.227
- drop Rajasthan, dD0_05 to cs0913: b -1.694, state SE 0.671, t -2.52, G 20, n 422: p normal 0.012, p t(G-1) 0.021, p wild-cluster (Webb, 9999 draws) 0.034
- drop Rajasthan, dD0_05 to w1517: b -1.459, state SE 0.909, t -1.60, G 20, n 415: p normal 0.109, p t(G-1) 0.125, p wild-cluster (Webb, 9999 draws) 0.140
- drop Rajasthan + Madhya Pradesh, dD0_05 to cs0913: b -1.574, state SE 0.797, t -1.98, G 19, n 377: p normal 0.048, p t(G-1) 0.064, p wild-cluster (Webb, 9999 draws) 0.070
- drop Rajasthan + Madhya Pradesh, dD0_05 to w1517: b -2.072, state SE 0.872, t -2.38, G 19, n 370: p normal 0.017, p t(G-1) 0.029, p wild-cluster (Webb, 9999 draws) 0.047

BH across the 32 total-change slopes (4 outcomes x 2 census spans x 2 windows x 2 samples):
- d_D0_05_cs0913 1991_2011 cs0913 all: b -1.75, p 0.014/0.006, q 0.322/0.192
- d_D0_05_cs0913 1991_2011 cs0913 gross area within 2/3 to 1.5x: b -1.74, p 0.020/0.015, q 0.322/0.241
- d_D1_cs0913 2001_2011 cs0913 all: b 0.52, p 0.125/0.067, q 0.780/0.404
- d_D1_cs0913 2001_2011 cs0913 gross area within 2/3 to 1.5x: b 0.45, p 0.158/0.069, q 0.780/0.404
- d_D0_05_w1517 2001_2011 w1517 gross area within 2/3 to 1.5x: b -0.89, p 0.219/0.070, q 0.780/0.404
- d_D0_05_w1517 1991_2011 w1517 gross area within 2/3 to 1.5x: b -1.48, p 0.044/0.081, q 0.353/0.404
- 2001-2011 change (no 1991 repair involved), dD0_05 to cs0913: b -0.284, state SE 0.646, t -0.44, G 22, n 434: p normal 0.660, p t(G-1) 0.665, p wild-cluster (Webb, 9999 draws) 0.670
- 2001-2011 change (no 1991 repair involved), dD0_05 to w1517: b -0.529, state SE 0.536, t -0.99, G 22, n 426: p normal 0.324, p t(G-1) 0.335, p wild-cluster (Webb, 9999 draws) 0.315
